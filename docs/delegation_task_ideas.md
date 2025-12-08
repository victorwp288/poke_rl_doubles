# Delegation Backlog – Small but Impactful Tasks

This backlog collects lightweight tasks (roughly 1–4 focused hours each) that keep the project healthy without demanding a full sprint. Each item notes the goal, suggested files, and what to verify before calling it done.

## Testing & Quality Assurance
- **Expand mask regression tests**  
  Goal: Add edge-case coverage for forced double switches and gimmick formats in `tests/test_action_repair.py`.  
  Files: `src/core/features.py`, `src/online/env.py`, `tests/test_action_repair.py`.  
  Done when the new tests fail before the fix and pass afterward.
- **Smoke test for imitation collector**  
  Goal: Create a short-lived asyncio test that validates `Recorder.rotate()` and `RecordingHeuristics` logging without hitting poke-env.  
  Files: `tools/imitation_collect.py`, `tests/`.  
  Done when pytest suite exercises recorder rotation and mask capture.
- **Config schema validation**  
  Goal: Add a test that loads `config/defaults.yaml` via `src/config.py` and checks required keys exist for `tools/online.py`.  
  Files: `src/config.py`, new test module under `tests/`.  
  Done when misconfigured YAML fails fast with a clear assertion.
- **Dataset loader fuzzing**  
  Goal: Introduce property-based tests (Hypothesis) for `_parse_payload()` to ensure malformed samples raise `ValueError`.  
  Files: `src/offline/dataset.py`, `pyproject.toml` (optional dependency), new test module.  
  Done when fuzzing catches invalid payloads and the test runs quickly.


## Tooling & Automation
- **Pre-commit hook sync**  
  Goal: Align `.pre-commit-config.yaml` with current Ruff/mypy versions and add `pyupgrade`.  
  Files: `.pre-commit-config.yaml`, `requirements.txt`.  
  Done when `pre-commit run --all-files` succeeds using locked versions.
- **Makefile or Justfile**  
  Goal: Introduce a simple task runner for frequent commands (install, format, test, run PPO).  
  Files: new `Makefile` or `justfile`, README snippet.  
  Done when developers can run `make test` and `make online MODE=warmstart`.
- **VS Code workspace settings**  
  Goal: Ship a pre-configured debugger/launch profile for `tools/online.py`.  
  Files: `.vscode/launch.json`, `.vscode/settings.json`.  
  Done when pressing Run in VS Code launches PPO training with env vars.

## Data Pipeline
- **Replay fetch dry-run mode**  
  Goal: Add a `--dry-run` flag to `tools/data_fetch.py` that lists targets without downloading.  
  Files: `tools/data_fetch.py`.  
  Done when the flag prints planned downloads and exits zero.
- **Human hints dataset sampler**  
  Goal: Provide a CLI to sample `data/processed/human_hints.jsonl` for manual review.  
  Files: new `tools/hints_sample.py` or extension to `data_parse.py`.  
  Done when `python tools/hints_sample.py --n 10` prints random entries.
- **Dataset size telemetry**  
  Goal: Log dataset count and mask sparsity when running `tools/imitation_merge.py`.  
  Files: `tools/imitation_merge.py`.  
  Done when merge output prints aggregated stats.

## Model & Training Improvements
- **KL schedule presets**  
  Goal: Add preset schedules (linear, cosine) to `KLRegularizedMaskablePPO` and expose them via config.  
  Files: `src/online/kl_ppo.py`, `tools/online.py`, `config/defaults.yaml`.  
  Done when modes can set `kl_schedule: cosine` and training logs confirm.
- **Normalization stats export**  
  Goal: Emit a CSV summary of `NormalizationStats` when loading BC weights.  
  Files: `src/online/init.py`, `tools/online.py`.  
  Done when the CSV appears under `outputs/models/`.
- **Reward metric plotting hook**  
  Goal: Extend `Gen9DoublesEnv` to expose reward components for TensorBoard logging.  
  Files: `src/online/env.py`, `tools/online.py`.  
  Done when PPO runs show individual reward curves.
- **Observation encoder benchmarking**  
  Goal: Add a script measuring encode speed and identifying hotspots using `timeit`.  
  Files: new `tools/profile_observations.py`.  
  Done when the script prints ops/sec on sample battles.

## Infrastructure & Ops
- **TensorBoard index HTML**  
  Goal: Generate a lightweight HTML file linking to all TensorBoard runs in `outputs/tensorboard/`.  
  Files: new script under `tools/`, README entry.  
  Done when `python tools/list_tensorboard.py` writes overview.html.
- **Log rotation guard**  
  Goal: Ensure `outputs/logs/online_env.log` rotates after N MB using Python’s `RotatingFileHandler`.  
  Files: `src/online/env.py`.  
  Done when long runs keep log size bounded.
- **GitHub Actions smoke job**  
  Goal: Add an optional CI job that installs dependencies and runs the smoke test.  
  Files: `.github/workflows/ci.yml` (new).  
  Done when push triggers the job and it passes locally.

## Miscellaneous Quick Wins
- **Config diff helper**  
  Goal: Write a small script to diff two YAML files (baseline vs override) and highlight changes.  
  Files: new `tools/config_diff.py`.  
  Done when the script prints key-level changes for two paths.
- **Progress bar standardisation**  
  Goal: Wrap long-running loops in `tools/imitation_batch.py` with `tqdm` and handle optional dependency gracefully.  
  Files: `tools/imitation_batch.py`, `requirements.txt` (optional extra).  
  Done when batch runs show progress bars without breaking headless runs.
- **Static typing sweep**  
  Goal: Add missing type hints in `tools/online.py` helper functions and ensure `mypy` stays happy.  
  Files: `tools/online.py`, `pyproject.toml` (if stricter options added).  
  Done when `mypy src tools` passes with the new annotations.

Adapt, split, or combine tasks as team members see fit; each is small enough for a partial day but meaningful enough to improve the project. Keep the list evolving by pruning completed items and adding new ones as gaps surface.
