# Assignment Findings: Gemini 2.5 Flash integration and load on Vertex

**Candidate**: Jonathan Awotwi

This document details my findings during the assignment. I worked on two iterations of the project:
First iteration: Focused on complex logging systems and experimental features.
Second iteration (this submission): Aimed for the simplest solution that directly addresses the assignment requirements.

The first attempt was manually steered and went a little too far with complex logging systems and experiments. The second relied more on Claude to generate code for the "simplest solution that answers what was asked for the assignment". Consequently, the load tests and technical evaluations assumed an "AI as judge" default.

| File | Description |
|:--|:--|
| `llm/gemini.py` | The provider |
| `loadtest.py` | Load harness |
| `tests/` | Edge cases the load testing found |
| `results/` | Raw JSON, each file tagged with the region that produced it |

Provided GCP project · `us-central1` per the brief · `gemini-2.5-flash` · macOS client on residential broadband · **18,772 live requests, zero failures** — including 900 with retries disabled (`VERTEX_MAX_ATTEMPTS=1`), so the zero is not retry-masked.

## Where I'd set the dial

Run each worker process at **concurrency 128**, and scale by adding processes rather than raising concurrency. One process sustained **62.6 req/s** over 3,000 requests; three concurrent processes summed to **128.7 req/s**. Past 256 in a single process, throughput falls and the extra requests only queue.

| conc | req/s | p50 | p95 | errors |
|-----:|------:|----:|-----:|:--|
| 16 | 6.86 | 1.79 | 3.10 | 0 |
| 32 | 11.41 | 1.82 | 3.03 | 0 |
| 64 | 22.05 | 1.81 | 3.20 | 0 |
| **128** | **62.56** | **1.78** | **3.03** | 0 |
| 512 | 34.17 | 9.62 | 31.24 | 0 |
| 1024 | 30.46 | 21.70 | 53.48 | 0 |
| 2048 | 33.17 | 37.06 | 107.08 | 0 · one 429, retried |

Throughput tops out near 63 req/s then *drops*, while latency climbs in step with concurrency — textbook saturation. **Nothing errors. It queues.** Error rate reports perfect health right up to a 37-second median, so alarms have to watch p95 drift, not error count.

Three processes at 128 each summed to 128.7 req/s with per-process p50 unchanged at 1.78s, so the ceiling is per-process, not Vertex's. Sharding is not free, though: p95 rose 3.03 → 4.8s and p99 3.86 → 8.1s. I did not isolate *which* per-process limit binds — the client pools `max(parallelism * 2, 100)` = 256 connections by default, and throughput plateaus at exactly that concurrency, so the pool is at least as likely a cause as the event loop. Either way I never reached Vertex's limit.

**A trap in my own method.** One rung reported p50 2.42s at concurrency 512; the dedicated run said 9.62s. The difference was 300 requests versus 2,048: with 300 the workers never fill, so the run measures ramp-up and calls it steady state. The table therefore drops every rung below a 2:1 requests-to-concurrency ratio, which excludes 192 and 256 from the sweep file. The 1024 and 2048 rows sit at exactly 2:1 — enough to show the trend, not enough to call them clean steady state.

## Does the answer hold still?

The product consumes brand order, so I measured whether *order* moves, not whether wording does. 100 identical calls per setting:

| temperature | distinct answer strings | distinct brand orderings | ranked #1 |
|:--|--:|--:|:--|
| 0.0 | 19 | **4** | Nike, 100/100 |
| 0.7 | 100 | **4** | Nike, 100/100 |

`temperature=0` is **not** deterministic — 19 strings from an identical prompt. But the signal is steadier than the prose: the same four orderings at both settings, with only positions three and four trading places. Temperature barely touches the ranking.

The exposure is elsewhere. Two orderings differ only in `Hoka` versus `Hoka One One` — one brand, two surface forms. Counting mentions needs entity normalisation more than it needs a low temperature.

## Model behaviour worth knowing

| Behaviour | Detail |
|:--|:--|
| Thinking is on unless disabled | Billed as output. **4.4x the cost, 3.1x the p50** against `thinking_budget=0` |
| Truncation returns success | `MAX_TOKENS` gives `text=''` *and* `candidatesTokenCount: None` — not 0 |
| The client timeout is a server deadline | Expiry arrives as 504, 499, or a local `ReadTimeout`, unpredictably. All retryable; my first attempt omitted 499 |
| `role="system"` is refused | `400 Content with system role is not supported` |
| Temperature is bounded | 0 to 2.0001 exclusive. 400s are not retried |
| Logprobs are free | Work with thinking on or off; no measurable cost |
| A refusal looks like an answer | `STOP`, with the refusal as ordinary text |

| thinking_budget | p50 | p95 | output tokens | $/1k calls |
|:--|--:|--:|--:|--:|
| **0** — shipped | **2.74** | **10.34** | **225** | **$0.58** |
| 128 | 3.07 | 7.58 | 311 | $0.79 |
| 512 | 5.75 | 14.22 | 590 | $1.49 |
| dynamic — Google's | 8.61 | 17.54 | 996 | $2.50 |

p95 is noisy at this sample size; p50 and token counts are the stable signal.

## Spend

All testing, both endpoints, ~35,000 requests: **~$25** at list price. Projected monthly at 1M calls: **~$575** with thinking off, **~$2,500** without. Estimated from the harness's token counts rather than billing, so treat as indicative. Testing is cheap; the default you ship is not.

## Why the code looks like this

| Choice | Reason |
|:--|:--|
| `system_instruction`, not a system message | Vertex refuses `role="system"` |
| Thinking off | 4.4x cost and 2.8x p95 buy nothing on this task |
| Retries inside the provider | `SimpleResponse` cannot say "retryable", and the caller cannot separate a transient 503 from a permanent 400 |
| Full jitter | Regional quota fails every request at once; lockstep retries re-trip it |
| Thinking counted in `output_tokens` | Billed as output. Overstates answer length — accepted |
| 30s deadline | ~10x p95. Frees a stuck worker instead of holding it |
| Empty answers raise | A silent `""` reads as "no brands named" |
| `retries` exposed as a counter | Absent from the success rate, present in the tail |
| `parallelism()` = 128 | Highest sustained rate before queueing starts |

## A wrong turn worth reporting

I ran ~16,000 requests against `global` before re-reading the brief, which specifies us-central1.

- Quota is regional, so those numbers described a pool nobody provisioned here. Discarded.
- They produced a phantom: in ~1 burst of 3, 2–6% of requests stalled 30–120s at an otherwise normal p50. I eliminated quota, pool sizing, auth stampede and cold start before changing region — on us-central1 it does not occur. The endpoint was the variable.
- Cost ~$9, and it is why every result file now records its region.

Separately, an A/B on timeouts showed a 6.5x gain that vanished once interleaved and repeated — the slow run was slow under both settings. Run-to-run spread here is wider than most effects worth chasing. Everything above was repeated.

## If this went to production

1. **Shard harder and find Vertex's real limit** — three processes did not reach it.
2. **Alarm on p95 drift, not errors** — this degrades silently; the harness emits p99 and a >10s count.
3. **Normalise brand entities** — `Hoka` and `Hoka One One` are one company.
4. **Price Provisioned Throughput** — pay-as-you-go capacity tracks account spend, which is not a guarantee.
5. **Score `thinking_budget=0` on the real evaluation** — ordering held over 200 calls, but that is one question.
6. **Pin the model version** — a floating alias turns a model roll into a silent data change.

Unknown: the real traffic shape. Burst profile, prompt length and latency budget would all move the operating point.
