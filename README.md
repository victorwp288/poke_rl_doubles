# RL Agent for Pokémon Showdown — Gen 9 Doubles (OU)

Start: Thu, 11 September 2025  
Hand‑in: Thu, 18 December 2025 at 10:00 CET

## Overview
PPO‑based agent built on Stable‑Baselines3 (PyTorch) that plays
**Generation 9 Doubles (OU)** via **poke‑env**. Training focuses on a local
Showdown simulator first; light online evaluation happens on a labeled bot
account with polite rate limits. We evaluate offline vs heuristics and via
ladder rating.

Little related experiment: https://github.com/victorwp288/poke-rl-demo

## Quick Start
Prereqs: Python 3.11. 

```bash

./init.sh

# fallback manual install
# pip install -r requirements.txt

# smoke‑test the environment
python tests/smoke_test_env.py

# collect imitation data (optional, adjust config/defaults.yaml first)
python tools/imitation_collect.py

# merge shard outputs into a single dataset (uses imitation_merge config)
python tools/imitation_merge.py

# orchestrate multi-shard imitation batches (see config/imitation_batches)
python tools/imitation_batch.py

# train the behavior cloning policy
python tools/offline.py

# evaluate the BC policy against bots
python tools/evaluate_bc.py

# run warmstart + scratch PPO back to back
python tools/run_ppo_compare.py

# inspect losses in TensorBoard
tensorboard --logdir outputs/tensorboard
```

## Development
- Lint/format/type‑check: `ruff format . && ruff check --fix . && mypy src`
- Web viewer: `python web/viewer_gradio.py`
- Optional: set up pre‑commit hooks: `pre-commit install`

## Project Structure
- `src/` — reusable library code (Python 3.11)
  - `src/core/` — shared helpers and environment wiring
  - `src/offline/` — dataset loader, policy, trainer
  - `src/online/` — reserved for PPO work
- `tools/` — simple entry points (train offline, sweep, data fetch/parse)
- `tests/` — smoke tests and simple unit tests
- `docs/` — notes and design docs
- `teams/` — example Pokémon team exports for experiments



## Configuration
- Defaults for offline, online, and data tooling live in `config/defaults.yaml`.
- The `online` block mirrors the offline BC architecture via `policy_hidden_dim` / `policy_hidden_layers` so weight warmstarts succeed by default.
- `imitation_batches`, `evaluation`, and `ppo_runs` blocks drive the orchestration helpers in `tools/`.
- Both offline and PPO runs now emit a best-performing checkpoint alongside the latest weights; adjust `best_policy_path` / `best_stats_path` if you need custom locations.
- Update the YAML to adjust paths, hyperparameters, or dataset sources.
- Offline trainer saves checkpoints under `outputs/models/` and TensorBoard events under `outputs/tensorboard/`.

## Offline Pipeline
- Collect imitation dataset: edit `config/defaults.yaml` and run `python tools/imitation_collect.py` for custom formats/teams.
- Pretrain behavior cloning policy: run `python tools/offline.py` to fit `BehaviorCloningPolicy`; per-epoch losses are printed and logged to TensorBoard when available.
- Monitor training: launch `tensorboard --logdir outputs/tensorboard` to inspect loss curves.
- Consume the policy: load `outputs/models/bc_policy.pt` into offline evaluation or PPO fine-tuning.

## Style & Conventions
- Formatting via Ruff: line length 100, double quotes, LF, 4‑space indent.
- Imports auto‑sorted (Ruff/isort). Library code in `src/`.
- Names: modules/files `snake_case.py`; functions/vars `snake_case`; classes
  `CamelCase`; constants `UPPER_SNAKE`.
- Add type hints for new public functions; run `mypy src` locally before PRs.

## Security & Server Etiquette
- Never commit secrets or account tokens. Prefer environment variables or
  plain Python config files under `data/sources/` when needed.
- Be polite with Showdown servers: rate-limit requests and prefer a local
  server for heavy training.

## Data & Storage Guidance
- Processed datasets live under `data/processed/` (e.g., `imitation.jsonl`, `human_hints.jsonl`).
- Raw replays stay in `data/raw/` (`downloaded/`, `showdown_logs/`).
- Model artifacts and run outputs go to `outputs/` (subfolders for `models/`, `tensorboard/`, `plots/`).
- Keep JSONL append-only until they approach ~1 GB, then consider caching in SQLite if needed.
