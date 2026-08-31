import asyncio
import os
import random

import httpx
from google import genai
from google.genai import errors, types

from llm import LLM


RETRYABLE_STATUS = frozenset({408, 429, 499, 500, 502, 503, 504})


class EmptyAnswer(RuntimeError):
    """Gemini returned a candidate with no text (safety block, or the thinking
    budget consumed the whole output allowance)."""


class Gemini(LLM):
    """Gemini 2.5 Flash on Vertex AI."""

    def __init__(self):
        self.__model = os.getenv("VERTEX_MODEL", "gemini-2.5-flash")
        self.__parallelism = int(os.getenv("VERTEX_PARALLELISM", "128"))
        self.__max_attempts = int(os.getenv("VERTEX_MAX_ATTEMPTS", "5"))
        self.__thinking_budget = int(os.getenv("VERTEX_THINKING_BUDGET", "0"))

        # Retries are invisible in the success rate, visible only in the tail.
        self.retries: dict[str, int] = {}

        # httpx keeps only 20 connections alive by default.
        connections = max(self.__parallelism * 2, 100)

        self.__client = genai.Client(
            vertexai=True,
            project=os.getenv("VERTEX_PROJECT"),
            # Per the brief; quota is allocated per region.
            location=os.getenv("VERTEX_LOCATION", "us-central1"),
            http_options=types.HttpOptions(
                # Forwarded to Vertex as a server-side deadline, not a local
                # read timeout, so expiry returns a retryable error.
                timeout=int(os.getenv("VERTEX_TIMEOUT_MS", "30000")),
                async_client_args={
                    "limits": httpx.Limits(
                        max_connections=connections,
                        max_keepalive_connections=connections,
                    ),
                },
            ),
        )

    def parallelism(self):
        return self.__parallelism

    async def ask_generic_question(self, system_prompt: str, question: str, temperature: float) -> LLM.SimpleResponse:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=self.__thinking_budget),
            response_logprobs=True,
            logprobs=1,
        )

        response = await self.__with_retries(
            lambda: self.__client.aio.models.generate_content(
                model=self.__model,
                contents=question,
                config=config,
            )
        )

        return self.__to_simple_response(response)

    async def __with_retries(self, call):
        for attempt in range(self.__max_attempts):
            try:
                return await call()
            except errors.APIError as e:
                if e.code not in RETRYABLE_STATUS or attempt == self.__max_attempts - 1:
                    raise
                self.__count_retry(str(e.code))
            except (httpx.TransportError, asyncio.TimeoutError) as e:
                if attempt == self.__max_attempts - 1:
                    raise
                self.__count_retry(type(e).__name__)

            # Full jitter: a quota trip fails every request at once, and
            # synchronised retries would re-trip it.
            await asyncio.sleep(random.uniform(0, min(8.0, 0.5 * 2 ** attempt)))

    def __count_retry(self, reason: str):
        self.retries[reason] = self.retries.get(reason, 0) + 1

    def __to_simple_response(self, response) -> LLM.SimpleResponse:
        usage = response.usage_metadata
        # Billed as output. Both are None, not 0, when absent.
        thoughts = usage.thoughts_token_count or 0
        answered = usage.candidates_token_count or 0

        answer = response.text
        if not answer:
            finish = response.candidates[0].finish_reason if response.candidates else None
            raise EmptyAnswer(f"no text in response (finish_reason={finish}, thinking_tokens={thoughts})")

        return LLM.SimpleResponse(
            answer=answer,
            input_tokens=usage.prompt_token_count or 0,
            output_tokens=answered + thoughts,
        )
