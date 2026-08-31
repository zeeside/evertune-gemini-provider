"""The two experiments that aren't concurrency runs, so their tables in FINDINGS.md
are reproducible too.

    python experiments.py determinism   # -> results/determinism-temperature-{0.0,0.7}.json
    python experiments.py thinking      # -> results/thinking-budget-sweep.json
"""

import asyncio
import collections
import json
import logging
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

from llm import Gemini
from loadtest import SYSTEM_PROMPT, workload

IN_PRICE, OUT_PRICE = 0.30 / 1e6, 2.50 / 1e6      # Vertex list price per token
QUESTION = "What are the top 5 running shoe brands?"


def orderings(answer: str) -> tuple:
    """Brand names in rank order — the signal the product consumes, not the prose."""
    found = []
    for line in answer.split("\n"):
        m = re.match(r"\s*\d+[.)]\s*\**([A-Za-z0-9 &'\-]+?)\**\s*[:\-—]", line)
        if m:
            found.append(m.group(1).strip())
    return tuple(found)


def summarise_determinism(answers: list[str], temperature: float) -> dict:
    ranks = collections.Counter(orderings(a) for a in answers)
    return {
        "temperature": temperature,
        "n": len(answers),
        "distinct_answer_strings": len(set(answers)),
        "distinct_brand_orderings": len(ranks),
        "orderings": {" > ".join(k): v for k, v in ranks.most_common()},
        "ranked_first": dict(collections.Counter(
            orderings(a)[0] for a in answers if orderings(a))),
        "answers": answers,
    }


async def determinism(n: int = 100):
    llm = Gemini()
    for temperature in (0.0, 0.7):
        answers = [r.answer.strip() for r in await asyncio.gather(
            *[llm.ask_generic_question(SYSTEM_PROMPT, QUESTION, temperature) for _ in range(n)])]
        out = summarise_determinism(answers, temperature)
        path = f"results/determinism-temperature-{temperature}.json"
        json.dump(out, open(path, "w"), indent=2)
        print(f"temperature={temperature}  strings={out['distinct_answer_strings']}  "
              f"orderings={out['distinct_brand_orderings']}  -> {path}")


async def thinking(n: int = 120, concurrency: int = 64):
    """Thinking tokens bill as output, so the budget is a cost lever, not just a latency one."""
    rows = []
    for budget in (0, 128, 512, -1):
        os.environ["VERTEX_THINKING_BUDGET"] = str(budget)
        llm, queue, lat = Gemini(), asyncio.Queue(), []
        for q in workload(n, seed=42):
            queue.put_nowait(q)
        tin = tout = 0

        async def worker():
            nonlocal tin, tout
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                started = time.perf_counter()
                r = await llm.ask_generic_question(SYSTEM_PROMPT, item, 0.7)
                lat.append(time.perf_counter() - started)
                tin += r.input_tokens
                tout += r.output_tokens

        await asyncio.gather(*[worker() for _ in range(concurrency)])
        lat.sort()
        pick = lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 2)
        rows.append({
            "thinking_budget": "dynamic" if budget == -1 else budget,
            "n": len(lat), "p50_s": pick(.5), "p95_s": pick(.95),
            "mean_input_tokens": round(tin / len(lat), 1),
            "mean_output_tokens": round(tout / len(lat), 1),
            "usd_per_1k_calls": round((tin / len(lat) * IN_PRICE
                                       + tout / len(lat) * OUT_PRICE) * 1000, 3),
        })
        print(f"  budget={rows[-1]['thinking_budget']:<8} p50={rows[-1]['p50_s']:<6} "
              f"p95={rows[-1]['p95_s']:<6} out={rows[-1]['mean_output_tokens']:<7} "
              f"${rows[-1]['usd_per_1k_calls']}/1k")
    json.dump(rows, open("results/thinking-budget-sweep.json", "w"), indent=2)
    print("-> results/thinking-budget-sweep.json")


if __name__ == "__main__":
    asyncio.run({"determinism": determinism, "thinking": thinking}[sys.argv[1]]())
