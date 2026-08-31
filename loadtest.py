"""Load harness for the LLM providers.

Fires a fixed number of brand-ranking questions through a provider at a given
concurrency and reports latency percentiles, throughput and a breakdown of what
failed. Set VERTEX_MAX_ATTEMPTS=1 to measure the raw error rate the provider
would otherwise hide behind retries.

    python loadtest.py --concurrency 32 --requests 200
    python loadtest.py --sweep 8,16,32,64,128 --requests 200
"""

import argparse
import asyncio
import itertools
import json
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass

import logging

warnings.filterwarnings("ignore")
logging.getLogger("google_genai.models").setLevel(logging.ERROR)  # silences a per-call AFC notice

from llm import Gemini, Together

PROVIDERS = {"gemini": Gemini, "together": Together}

SYSTEM_PROMPT = (
    "You are a market research analyst. Answer with a numbered list of brands, "
    "most prominent first, with one sentence of justification each."
)

# Roughly the shape of real traffic: the same question template fanned out over
# many categories, so caching and prompt reuse cannot flatter the numbers.
CATEGORIES = [
    "running shoes", "electric vehicles", "laptop computers", "credit cards",
    "streaming services", "coffee chains", "airlines", "smartphones",
    "athletic apparel", "meal kit delivery", "home insurance", "mattresses",
    "noise cancelling headphones", "project management software", "grocery chains",
    "hotel chains", "ride hailing apps", "cloud providers", "sunscreen", "pet food",
]
TEMPLATES = [
    "What are the top 5 {c} brands?",
    "If I asked a friend for a {c} recommendation, which brands would they name?",
    "Rank the leading {c} brands by how often people recommend them.",
]


@dataclass
class Result:
    latency: float
    ok: bool
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


def workload(n: int, seed: int = 0) -> list[str]:
    prompts = [t.format(c=c) for c, t in itertools.product(CATEGORIES, TEMPLATES)]
    rng = random.Random(seed)
    return [rng.choice(prompts) for _ in range(n)]


async def one(llm, question: str, temperature: float) -> Result:
    started = time.perf_counter()
    try:
        r = await llm.ask_generic_question(SYSTEM_PROMPT, question, temperature)
        return Result(time.perf_counter() - started, True,
                      input_tokens=r.input_tokens, output_tokens=r.output_tokens)
    except Exception as e:
        # Vertex errors carry the status in .code; fall back to the class name.
        code = getattr(e, "code", None)
        return Result(time.perf_counter() - started, False,
                      error=f"{type(e).__name__}:{code}" if code else type(e).__name__)


async def run(llm, questions: list[str], concurrency: int, temperature: float) -> tuple[list[Result], float]:
    queue = asyncio.Queue()
    for q in questions:
        queue.put_nowait(q)
    results: list[Result] = []

    async def worker():
        while True:
            try:
                q = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await one(llm, q, temperature))

    started = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(concurrency)])
    return results, time.perf_counter() - started


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((p / 100) * len(ordered) + 0.5)) - 1)]


# Latency here is bimodal when the stall mode bites, so a histogram and an
# explicit count of long requests say more than percentiles alone.
BUCKETS = [(0, 3), (3, 5), (5, 10), (10, 30), (30, 60), (60, float("inf"))]


def summarise(results: list[Result], elapsed: float, concurrency: int) -> dict:
    ok = [r for r in results if r.ok]
    latencies = [r.latency for r in ok]
    errors: dict[str, int] = {}
    for r in results:
        if not r.ok:
            errors[r.error] = errors.get(r.error, 0) + 1
    return {
        # Provenance: quota is allocated per region, so a number is only
        # meaningful alongside the endpoint that produced it.
        "region": os.getenv("VERTEX_LOCATION", "us-central1"),
        "model": os.getenv("VERTEX_MODEL", "gemini-2.5-flash"),
        "concurrency": concurrency,
        "requests": len(results),
        "ok": len(ok),
        "error_rate": round(1 - len(ok) / len(results), 4) if results else 0,
        "wall_s": round(elapsed, 2),
        "throughput_rps": round(len(ok) / elapsed, 2) if elapsed else 0,
        "output_tps": round(sum(r.output_tokens for r in ok) / elapsed, 1) if elapsed else 0,
        "p50_s": round(pct(latencies, 50), 2),
        "p90_s": round(pct(latencies, 90), 2),
        "p95_s": round(pct(latencies, 95), 2),
        "p99_s": round(pct(latencies, 99), 2),
        "max_s": round(max(latencies), 2) if latencies else 0,
        "mean_output_tokens": round(sum(r.output_tokens for r in ok) / len(ok), 1) if ok else 0,
        "stalls_over_10s": sum(1 for l in latencies if l > 10),
        "latency_buckets": {f"{lo}-{hi}s": sum(1 for l in latencies if lo <= l < hi) for lo, hi in BUCKETS},
        "errors": errors,
    }


HEADER = f"{'conc':>5} {'reqs':>5} {'ok':>5} {'err%':>6} {'rps':>7} {'out_tps':>8} {'p50':>6} {'p90':>6} {'p95':>6} {'p99':>6} {'max':>6}"


def render(s: dict) -> str:
    return (f"{s['concurrency']:>5} {s['requests']:>5} {s['ok']:>5} {s['error_rate'] * 100:>5.1f}% "
            f"{s['throughput_rps']:>7.2f} {s['output_tps']:>8.1f} {s['p50_s']:>6.2f} {s['p90_s']:>6.2f} "
            f"{s['p95_s']:>6.2f} {s['p99_s']:>6.2f} {s['max_s']:>6.2f}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="gemini", choices=PROVIDERS)
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=None, help="defaults to provider.parallelism()")
    ap.add_argument("--sweep", type=str, default=None, help="comma-separated concurrencies to sweep")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--json", type=str, default=None, help="write summaries to this file")
    args = ap.parse_args()

    llm = PROVIDERS[args.provider]()
    levels = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.concurrency or llm.parallelism()]

    print(f"provider={args.provider} model={os.getenv('VERTEX_MODEL', 'gemini-2.5-flash')} "
          f"location={os.getenv('VERTEX_LOCATION', 'us-central1')} attempts={os.getenv('VERTEX_MAX_ATTEMPTS', '5')} "
          f"thinking={os.getenv('VERTEX_THINKING_BUDGET', '0')}")
    print(HEADER)

    summaries = []
    for i, c in enumerate(levels):
        results, elapsed = await run(llm, workload(args.requests, seed=i), c, args.temperature)
        s = summarise(results, elapsed, c)
        summaries.append(s)
        print(render(s), flush=True)
        if s["errors"]:
            print(f"      errors: {s['errors']}", flush=True)
        retries = getattr(llm, "retries", None)
        if retries:
            s["retries"] = dict(retries)
            print(f"      retries: {dict(retries)}", flush=True)
            retries.clear()
        if i < len(levels) - 1:
            await asyncio.sleep(5)  # let per-minute quota buckets drain between rungs

    if args.json:
        with open(args.json, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
