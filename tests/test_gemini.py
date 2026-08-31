"""Covers the parts of the Gemini provider that load testing found sharp edges in:
retry classification, thinking-token accounting, and empty answers."""

import os
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors

os.environ.setdefault("VERTEX_PROJECT", "test-project")

from llm import LLM
from llm.gemini import RETRYABLE_STATUS, EmptyAnswer, Gemini


def response(text="hi", prompt=10, candidates=20, thoughts=None, finish="STOP"):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=finish)],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt,
            candidates_token_count=candidates,
            thoughts_token_count=thoughts,
        ),
    )


def api_error(code):
    return errors.APIError(code, {"error": {"code": code, "message": "boom"}})


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setenv("VERTEX_MAX_ATTEMPTS", "3")
    g = Gemini()
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    return g


async def _no_sleep(_):
    return None


def stub(gemini, *outcomes):
    """Queue up per-attempt outcomes; exceptions are raised, values returned."""
    calls = iter(outcomes)

    async def generate_content(**_):
        outcome = next(calls)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    gemini._Gemini__client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)))
    return gemini


async def ask(gemini):
    return await gemini.ask_generic_question("system", "question", 0.7)


def test_deadline_codes_are_retryable():
    # The SDK reports our own timeout as a server deadline, not a local one.
    assert {499, 504} <= RETRYABLE_STATUS


@pytest.mark.asyncio
async def test_returns_answer_and_tokens():
    r = await ask(stub(Gemini(), response(text="Nike", prompt=11, candidates=22)))
    assert r == LLM.SimpleResponse(answer="Nike", input_tokens=11, output_tokens=22)


@pytest.mark.asyncio
async def test_thinking_tokens_count_as_output():
    # Vertex bills thoughts as output, so they must not vanish from the total.
    r = await ask(stub(Gemini(), response(candidates=20, thoughts=30)))
    assert r.output_tokens == 50


@pytest.mark.asyncio
async def test_retries_then_succeeds(gemini):
    stub(gemini, api_error(429), httpx.ConnectError("reset"), response(text="ok"))
    assert (await ask(gemini)).answer == "ok"
    assert gemini.retries == {"429": 1, "ConnectError": 1}


@pytest.mark.asyncio
async def test_does_not_retry_bad_request(gemini):
    # A temperature above 2.0 is a 400 and will fail identically forever.
    stub(gemini, api_error(400), response())
    with pytest.raises(errors.APIError):
        await ask(gemini)
    assert gemini.retries == {}


@pytest.mark.asyncio
async def test_gives_up_after_max_attempts(gemini):
    stub(gemini, *[api_error(503)] * 3)
    with pytest.raises(errors.APIError):
        await ask(gemini)


@pytest.mark.asyncio
async def test_truncated_response_raises_rather_than_returning_empty():
    # Thinking can eat the whole output allowance, leaving text="" and
    # candidates_token_count=None. Silently returning "" would poison the dataset.
    truncated = response(text="", candidates=None, thoughts=15, finish="MAX_TOKENS")
    with pytest.raises(EmptyAnswer, match="MAX_TOKENS"):
        await ask(stub(Gemini(), truncated))
