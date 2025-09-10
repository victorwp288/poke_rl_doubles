# RL Agent for Pokémon Showdown — Gen 9 Doubles (OU)

Start: Thu, 11 September 2025  
Hand‑in: Thu, 18 December 2025 at 10:00 CET

## Overview
PPO‑based recurrent agent built on Stable‑Baselines3 (PyTorch) that plays
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
```

## Development
- Lint/format/type‑check: `ruff format . && ruff check --fix . && mypy src`
- Web viewer: `python web/viewer_gradio.py`
- Optional: set up pre‑commit hooks: `pre-commit install`

## Project Structure
- `src/` — reusable library code (Python 3.11)
- `scripts/` — CLI entry points (training/eval)
- `configs/` — YAML configs (see `configs/default.yml`)
- `tests/` — smoke tests and simple unit tests
- `web/` — lightweight Gradio viewer(not required)
- `docs/` — notes and design docs
- `teams/` — example Pokémon team exports for experiments



## Configuration
Default PPO and environment settings live in `configs/default.yml` (seed,
format, PPO hyper‑parameters, and model sizes). You can duplicate this file and
override values per experiment.

## Style & Conventions
- Formatting via Ruff: line length 100, double quotes, LF, 4‑space indent.
- Imports auto‑sorted (Ruff/isort). Library code in `src/`.
- Names: modules/files `snake_case.py`; functions/vars `snake_case`; classes
  `CamelCase`; constants `UPPER_SNAKE`.
- Add type hints for new public functions; run `mypy src` locally before PRs.

## Security & Server Etiquette
- Never commit secrets or account tokens. Prefer environment variables and
  `configs/*.yml`.
- Be polite with Showdown servers: rate‑limit requests and prefer a local
  server for heavy training.

