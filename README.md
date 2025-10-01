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

# install deps
pip install -r requirements.txt

# smoke‑test the environment
python tests/smoke_test_env.py

# collect imitation data (optional, edit defaults in tools/imitation_collect.py first)
python tools/imitation_collect.py

# train the behavior cloning policy
python tools/offline.py

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
- Offline defaults live in `src/offline/config.py`.
- You can edit the dataclass or pass overrides in `tools/offline.py` as needed.
- Offline trainer saves checkpoints under `outputs/models/` and TensorBoard events under `outputs/tensorboard/`.

## Offline Pipeline
- Collect imitation dataset: run `python tools/imitation_collect.py` (tweak `DEFAULT_SETTINGS` for custom formats/teams).
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


