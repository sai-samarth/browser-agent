# OpenEnv + BrowserGym (MiniWoB) Setup Notes

## Initial Setup (local client project)
- Created a new project folder and initialized Python project tooling:
  - `uv init`
  - created a venv using Python 3.11 (`uv venv --python 3.11`)
    - OpenEnv mentions Python 3.10+; I assumed 3.11 is safe.
  - added `requests` to `pyproject.toml` (via `uv add requests`)
- Docker in WSL2:
  - Best setup is Docker Desktop on Windows, then enable WSL integration:
    - Docker Desktop → Settings → Resources → WSL Integration → enable your distro (Ubuntu, etc.)
  - Verify inside WSL:
    - `docker --version`
    - `docker ps`

---

## Attempt 1: Pull prebuilt OpenEnv BrowserGym image (failed)
Goal: run the OpenEnv BrowserGym Docker server quickly, without building locally.

Tried:
```bash
docker pull ghcr.io/openenv/browsergym-env:latest
```

Got:
```text
Error response from daemon: error from registry: denied
denied
```

Conclusion:
- The image name was wrong (or not accessible).
- Correct image is:
```bash
docker pull ghcr.io/meta-pytorch/openenv-browsergym-env:latest
```

To match OpenEnv docs naming conventions, tagged it locally as:
```bash
docker tag ghcr.io/meta-pytorch/openenv-browsergym-env:latest browsergym-env:latest
```

---

## Attempt 2: Run MiniWoB click-test on the prebuilt image (failed)
Tried to run:
```bash
docker run --rm -p 8000:8000 \
  -e BROWSERGYM_BENCHMARK="miniwob" \
  -e BROWSERGYM_TASK_NAME="click-test" \
  browsergym-env:latest
```

Error:
```text
ValueError: Failed to import BrowserGym benchmark 'miniwob': No module named 'browsergym.envs'
Make sure the package browsergym-miniwob is installed.
```

Initial assumption:
- Maybe MiniWoB package wasn’t installed inside the container.

Two ways forward:
- Quick/temporary: exec into the container and install packages (but you must redo it every new container).
- Better: create a derived Docker image that installs what you need.

---

## Attempt 3: Create a derived image for MiniWoB (built, then still failed)
Created a Dockerfile to install MiniWoB and required assets.

### 3.1 First derived Dockerfile idea
```dockerfile
FROM ghcr.io/meta-pytorch/openenv-browsergym-env:latest
RUN pip install --no-cache-dir -U browsergym browsergym-miniwob
```

### 3.2 Realization: MiniWoB needs extra setup beyond pip
MiniWoB++ requires:
- MiniWoB++ HTML assets repo (`miniwob-plusplus`)
- `MINIWOB_URL` pointing to those assets (file:// or http:// URL)
- a working Chromium runtime driven by Playwright

So I created a fuller `Dockerfile.browsergym-miniwob` (initial version):

```dockerfile
FROM ghcr.io/meta-pytorch/openenv-browsergym-env:latest

# Install benchmark package
RUN pip install --no-cache-dir -U browsergym-miniwob

# System deps for git (sometimes already present, but safe)
RUN apt-get update && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

# Clone MiniWoB++ at a pinned commit (reproducible)
RUN git clone https://github.com/Farama-Foundation/miniwob-plusplus.git /miniwob-plusplus \
 && cd /miniwob-plusplus \
 && git reset --hard 7fd85d71a4b60325c6585396ec4f48377d049838

# Playwright browser
RUN python -m playwright install --with-deps chromium

# Tell BrowserGym where MiniWoB HTML lives
ENV MINIWOB_URL="file:///miniwob-plusplus/miniwob/html/miniwob/"
```

### 3.3 Build failed: Playwright `--with-deps` broke on Debian
The build failed during:
```bash
python -m playwright install --with-deps chromium
```

Reason:
- Playwright dependency installer tried to install Ubuntu font packages that don’t exist in the container’s Debian repo:
  - `ttf-ubuntu-font-family` (missing)
  - `ttf-unifont` (missing / replaced by `fonts-unifont`)
- Result: `Failed to install browsers` (exit code 100)

### 3.4 Workaround: use system Chromium instead of Playwright-managed Chromium
Switched to a more robust approach:
- Avoid Playwright’s OS-specific dependency script (`install --with-deps`)
- Install system `chromium` + required libs via apt
- Set `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1`

This allowed the derived image to build successfully.

### 3.5 Still failed at runtime: `browsergym.envs` import
Even after installing `browsergym` + `browsergym-miniwob`, the container startup still failed with:
```text
No module named 'browsergym.envs'
```

Debugging inside the container showed:
- `browsergym` installed, but as a namespace package (printing `browsergym: None`)
- OpenEnv server code was trying to import `browsergym.envs.<benchmark>`
- The installed BrowserGym version’s module layout didn’t expose `browsergym.envs` in the way the server expected
- Conclusion: likely a mismatch between the OpenEnv server’s benchmark import path and the BrowserGym package layout/version inside this derived image.

At this point, patching the OpenEnv server import path was suggested, but instead I chose to follow the “official build from source” OpenEnv docs path.

---

## Attempt 4: Build OpenEnv + BrowserGym server images from source (worked)
Instead of using the prebuilt container as the base, I followed the OpenEnv repo instructions:

From the OpenEnv repository root:
```bash
docker build -t openenv-base:latest -f src/openenv/core/containers/images/Dockerfile .
```

Then build the BrowserGym environment image:
```bash
cd envs/browsergym_env
docker build -t browsergym-env:latest -f server/Dockerfile .
```

Run the server for MiniWoB training:
```bash
docker run --rm -p 8000:8000 \
  -e BROWSERGYM_BENCHMARK="miniwob" \
  -e BROWSERGYM_TASK_NAME="click-test" \
  browsergym-env:latest
```

Verification:
```bash
curl -s http://localhost:8000/docs | head -n 10
```

This returned the FastAPI Swagger UI HTML (so the server is running).

Also verified OpenAPI schema:
```bash
curl -s http://localhost:8000/openapi.json | head -n 5
```

This returned JSON with endpoints like `/reset`, `/step`, `/schema`, `/state`, `/health`.

---

## Key learnings
- Prebuilt images may not include all benchmark packages/config and can have version mismatches.
- MiniWoB needs MiniWoB++ assets + `MINIWOB_URL`, not just `pip install`.
- Playwright `--with-deps` can break depending on base OS; system Chromium is a safer workaround if needed.
- Building from source using the OpenEnv repo Dockerfiles produced a working BrowserGym server setup.

---

## Attempt 5: Build first rollout collector (scripted policy only)
Goal: get a minimal, reproducible data collection loop working before adding teacher API complexity.

Created:
- `scripts/collect_rollouts.py`
- `configs/rollout_config.yaml`

Collector design:
- Read all run parameters from YAML
- Connect to OpenEnv over WebSocket (`/ws`)
- Run `reset -> step -> step ...` for each episode
- Save outputs to:
  - `trajectory_steps.jsonl`
  - `episode_summaries.jsonl`
  - `resolved_config.yaml`

Initial run command:
```bash
uv run python scripts/collect_rollouts.py --config configs/rollout_config.yaml
```

---

## Attempt 6: Server instability while integrating rollouts (fixed)
While testing collector runs, BrowserGym server behavior was unstable (`/reset` and `/web/reset` intermittent 500s).

Observed stack traces included:
- thread/greenlet switching errors
- event loop context errors

Fix applied:
- Updated `envs/browsergym_env/server/browsergym_environment.py`
- Forced BrowserGym reset/step/close calls through one shared single-thread path
- Rebuilt and restarted container

Verification after fix:
- `/health` healthy
- `/reset` works
- `/web/reset` works

---

## Attempt 7: First smoke run data looked wrong (fixed)
Collector ran successfully, but logs were clearly wrong:
- actions were recorded
- but `goal/url/text` were empty
- rewards stayed `0.0`, `done=false`

Root cause:
- WebSocket response parsing mismatch in `collect_rollouts.py`
- OpenEnv returns WS observation payload as:
```json
{
  "type": "observation",
  "data": {
    "observation": {...},
    "reward": ...,
    "done": ...
  }
}
```
- Collector was treating `data` directly as the observation body.

Fix:
- Added normalization helper in collector to flatten WS payload
- Updated reset/step parsing to use normalized observation dict

Result:
- observation fields started logging correctly (`goal/url/text` populated)

---

## Attempt 8: Action string mismatch on MiniWoB click-test (fixed)
After parser fix, observations looked good, but episodes still failed:
- `last_action_error=true` for all steps
- no reward
- no terminal success

Tried selector variants:
- `click('button')`
- `click('Click Me!')`
- `click('[13]')`
- etc.

Found working action in this setup:
```text
click('13')
```

Updated config:
```yaml
policy:
  scripted_actions:
    click-test:
      - "click('13')"
```

---

## Smoke test status (current: passing)
Run:
- `data/rollouts/miniwob_smoke_run_20260215_155203`

Outcome:
- 3/3 episodes successful
- each episode solved in 1 step
- `cum_reward = 1.0`
- `final_done = true`
- `action_error_count = 0`

Conclusion:
- End-to-end MiniWoB rollout collection is now working for `click-test`
- Next planned step: teacher API integration for model-generated actions

---

## Attempt 9: Teacher API integration (GLM-5) (worked)
Goal: move from scripted actions to model-generated actions.

Implemented in collector:
- `policy.mode: teacher` support
- OpenAI-compatible teacher config in YAML (`base_url`, `model`, API key env var, sampling params)
- teacher metadata logging per step:
  - `teacher_model`
  - `teacher_used_fallback`
  - `teacher_usage`
  - optional `teacher_raw_response`

Initial teacher smoke result:
- `click-test` worked and returned reward.

---

## Attempt 10: Multi-task collection looked wrong (diagnosed)
When I added multiple tasks, logs showed:
- `task_name` changed
- but page URL/goal stayed on `click-test`

This meant data quality was bad (task label and actual environment state diverged).

Safety fix in collector:
- Added task/URL consistency check (`enforce_task_match: true`) to fail fast when requested task != loaded task.

---

## Attempt 11: BrowserGym server task switching fix (done)
Root cause:
- BrowserGym server accepted `reset(task_name=...)` but did not recreate the underlying gym env for the new task.

Fix:
- Patched `envs/browsergym_env/server/browsergym_environment.py` so reset switches/recreates env when task changes.
- Rebuilt and restarted `browsergym-env:latest`.

Verification:
- Reset now loads correct task pages (`click-test`, `click-button-sequence`, `click-scroll-list`) with matching URLs/goals.

---

## Attempt 12: Teacher behavior on harder tasks (mixed)
After task-switch fix:
- `click-test`: solved reliably.
- `click-button-sequence`: often repeated first click and failed.
- `click-scroll-list`: often hit max steps with low reward.

Observed pattern:
- many fallback `noop()` actions on harder task episodes.

Debug enabled:
- `save_teacher_debug: true`
- `save_pruned_html: true`

This made it easier to inspect exactly what teacher returned each step.

---

## Attempt 13: Teacher parsing + chat history improvements
To reduce fallback and repeated actions:
- Enabled thinking mode in teacher request.
- Increased `max_tokens`.
- Improved extraction to parse actions from richer response shapes (including reasoning-style outputs).

Major behavior change:
- Switched teacher prompting to true chat-history format:
  - fixed `system`
  - `user`: current observation/goal
  - `assistant`: chosen action
  - repeat each step
- Added configurable history window: `max_history_turns`.

Goal:
- make model aware of prior actions/states inside the same episode.

---

## Attempt 14: Multi-select issue in `click-scroll-list` (resolved in prompting)
Key nuance:
- plain click on second option deselects first option in multi-select listbox.

Validated working action style:
- use modifier clicks for additional selections, e.g.
  - `click('14')`
  - `click('18', modifiers=['Control'])`
  - `click('23')`

Prompt updates:
- system prompt now documents modifier-click usage.
- per-step user prompt adds multi-select hint when observation includes `multiselectable=True`.

---

## Attempt 15: Episode variation + latency diagnosis (measured)
Question checked:
- with `episodes_per_task > 1`, was the same exact case repeating?

Findings:
- `episode_idx` and `seed` were incrementing correctly (`0,1,2` in the test run).
- for `click-test`, goal/action pattern stays the same, but layout varies by seed
  (button position/size changed across episodes).

Latency diagnosis:
- previous per-step `latency_ms` was ~2.8s in teacher runs.
- root bottleneck was observation payload size, not JSONL logging:
  - `/reset` response size was ~10,224,040 bytes
  - response included huge raw `screenshot` array.

Collector improvements (for throughput debugging + sharding):
- Added latency breakdown fields in step logs:
  - `env_step_latency_ms`
  - `step_total_latency_ms`
  - `teacher_latency_ms` (teacher mode)
  - `reset_latency_ms`
- Added sharding/repro knobs in collection config:
  - `episode_start_index`
  - `seed_offset`

---

## Attempt 16: Disable screenshots reproducibly (done)
Requirement:
- disable screenshot collection in a reproducible way (no runtime container patching).

Approach:
- applied source-level changes in the OpenEnv BrowserGym server wrapper
  (`envs/browsergym_env/server/browsergym_environment.py`) and rebuilt image.
- restarted server from rebuilt `browsergym-env:latest`.

Validation on 2026-02-20:
- `POST /reset`: `screenshot: null`, payload ~693 bytes.
- `POST /web/reset`: `screenshot: null`, payload ~694 bytes.
- this confirms screenshot is now disabled at source in served observations.

Latency impact check (scripted smoke):
- run: `data/rollouts/latency_check_no_screenshot_20260220_190209`
- `latency_ms` (env step) min/avg/max:
  - `799.705 / 808.938 / 815.914`
- `reset_latency_ms` ~700ms.

Compared to prior baseline:
- earlier run average step latency was ~2846ms.
- new average step latency is ~809ms (about 3.5x faster).

---

## Attempt 17: Switch teacher model to mPLUG/GUI-Owl-1.5-8B-Think (done)
Change requested:
- move teacher from GLM-5 to `mPLUG/GUI-Owl-1.5-8B-Think` served via vLLM OpenAI-compatible endpoint on port `7999`.

Config change applied:
- teacher API base URL switched to local vLLM endpoint (`http://localhost:7999/v1`).
- teacher model updated to `mPLUG/GUI-Owl-1.5-8B-Think`.

Validation:
- collector runs completed with the new model and produced step/episode logs under new run dirs.

---

## Attempt 18: Empty teacher text in some steps (parsing fix)
Observed:
- some step rows had empty teacher response text despite valid completions.

What was done:
- checked raw response shape directly against the vLLM endpoint.
- updated collector extraction to robustly parse both:
  - answer text
  - reasoning text
- preserved both fields in JSONL outputs:
  - `teacher_response_answer`
  - `teacher_response_reasoning`

Result:
- answer extraction became stable across response shape variants.

---

## Attempt 19: Reasoning capture for think model (prompting update)
Issue:
- reasoning was still often empty even with a think model.

Prompt update:
- teacher prompt now explicitly requests:
  - internal reasoning wrapped in `<think>...</think>`
  - final action output separately
- collector splits `<think>` blocks into reasoning and answer fields.

Outcome on later smoke runs:
- reasoning started appearing in `teacher_response_reasoning`.
- no recurring "empty first response with one token then retry" behavior in the latest checks.

---

## Attempt 20: Collector output cleanup (done)
Goal:
- remove unwanted/debug-heavy fields from rollout outputs while keeping training-relevant data.

Code cleanup in `scripts/collect_rollouts.py`:
- removed dead helper: `_get_teacher_response_text_candidates`.
- removed step-level fields:
  - `env_step_latency_ms`
  - `step_total_latency_ms`
  - `reset_latency_ms`
  - `teacher_messages_sent`
  - `teacher_history_turns`
  - `teacher_attempt_count`
  - `teacher_used_empty_retry`
  - `teacher_finish_reason`
  - `teacher_raw_response`
  - `teacher_raw_message`
- removed summary field:
  - `reset_latency_ms`
- removed now-unused debug/timing plumbing tied to the deleted fields.

Kept intentionally:
- `teacher_response_answer`
- `teacher_response_reasoning`
- `teacher_latency_ms`
- `teacher_usage`
- `teacher_model`
- `teacher_used_fallback`
- core rollout fields (`task_name`, `episode_idx`, `seed`, `action_str`, observations, reward/done/errors, `latency_ms`).

---

## Current status (latest)
- Infrastructure is stable (server healthy, task switching works, screenshot disabled).
- Teacher now runs on local vLLM (`mPLUG/GUI-Owl-1.5-8B-Think` at port `7999`).
- JSONL parsing captures both answer and reasoning (with `<think>` split handling).
- Collector output schema is now leaner and better aligned for SFT dataset generation.
- Current focus: large-scale reproducible rollout generation (sharded episodes/seeds) with quality checks.
