# Running the load tests

Setup and usage for `loadtest.py`. The assignment README at the repository root is unchanged.

```bash
pip install -r requirements.txt
export VERTEX_PROJECT=<your-gcp-project>
```

These reproduce the runs behind `FINDINGS.md`. Add `--json results/<name>.json` to save a summary:

```bash
# soak — the headline number
python loadtest.py --concurrency 128 --requests 3000

# concurrency sweep
python loadtest.py --sweep 16,32,64,128,192,256 --requests 300
python loadtest.py --sweep 512,1024 --requests 2048
python loadtest.py --concurrency 2048 --requests 4096

# raw error rate, retries disabled
VERTEX_MAX_ATTEMPTS=1 python loadtest.py --sweep 128,256,512 --requests 300

# four repeated bursts at one level
python loadtest.py --sweep 128,128,128,128 --requests 300

# multi-process: run three of these at once and sum the throughput
python loadtest.py --concurrency 128 --requests 1000
```

The two non-concurrency experiments write their own files:

```bash
python experiments.py determinism   # temperature 0.0 and 0.7, 100 calls each
python experiments.py thinking      # thinking_budget 0 / 128 / 512 / dynamic
```

**Keep `--requests` several times `--concurrency`.** With fewer requests than workers the
queue never fills, so the run measures ramp-up rather than steady state — at concurrency
512, 300 requests reports p50 2.42s where 2,048 requests reports 9.62s.

## Arguments

| Argument | Default | What it does |
|---|---|---|
| `--provider` | `gemini` | Which provider to drive: `gemini` or `together`. |
| `--requests` | `100` | Total requests to send. In `--sweep` mode this is **per concurrency level**, not across the whole run. |
| `--concurrency` | `provider.parallelism()` (128 for Gemini) | Number of workers pulling from the queue, i.e. requests in flight at once. Ignored when `--sweep` is given. |
| `--sweep` | *(off)* | Comma-separated concurrency levels to run in turn, e.g. `16,32,64`. Overrides `--concurrency`, pauses 5s between levels to let per-minute quota drain, and varies the prompt seed per level so no level replays another's cache. |
| `--temperature` | `0.7` | Sampling temperature passed to the model. Gemini's valid range is 0 to 2.0; anything higher is a hard 400. |
| `--json` | *(off)* | Path to write the per-level summaries to as JSON. Printed output is unchanged. |

## Environment variables

`--provider gemini` reads its configuration from the environment. `VERTEX_PROJECT` is
required; the rest have working defaults and exist mainly so the load tests can vary one
knob at a time.

| Variable | Default | What it does |
|---|---|---|
| `VERTEX_PROJECT` | *(required)* | GCP project billed for the calls. |
| `VERTEX_LOCATION` | `us-central1` | Vertex region. The brief specifies us-central1, where the project's resources are provisioned; quota is allocated per region. |
| `VERTEX_MODEL` | `gemini-2.5-flash` | Model ID. |
| `VERTEX_PARALLELISM` | `128` | What `parallelism()` advertises to callers, and what the HTTP connection pool is sized from. |
| `VERTEX_MAX_ATTEMPTS` | `5` | Total attempts per request. Set to `1` to measure the raw error rate with retries disabled. |
| `VERTEX_THINKING_BUDGET` | `0` | Thinking token budget. `0` disables thinking; `-1` restores Google's dynamic default, which costs 4.4x more per call. |
| `VERTEX_TIMEOUT_MS` | `30000` | Per-attempt deadline. Forwarded to Vertex as a server-side deadline, so exceeding it returns a retryable error rather than hanging. |
