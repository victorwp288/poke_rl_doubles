# Essential Commands & File Map

## Terminal Cheat Sheet
- `pip install -r requirements.txt` — install Python deps (PyTorch, poke-env, tooling).
- `python tests/smoke_test_env.py` — verify environment, poke-env, and basic rendering.
- `python tools/imitation_collect.py` — generate imitation JSONL (edit `DEFAULT_SETTINGS` for format/teams).
- `python tools/offline.py` — train behavior cloning policy; logs go to stdout and `outputs/tensorboard/`.
- `tensorboard --logdir outputs/tensorboard` — inspect offline training losses.
- `ruff format .` / `ruff check --fix .` / `mypy src` — formatting, linting, typing (see more in README).

## Project Layout Overview
- `src/core/` — environment utilities (`env.py` exposes `action_space_size`, etc.).
- `src/offline/`
  - `config.py` — `OfflineConfig` dataclass for dataset/model paths, hyperparameters, TensorBoard directory.
  - `dataset.py` — JSONL loader, parsing `ImitationSample`, train/val split helpers.
  - `model.py` — `BehaviorCloningPolicy`, two-head MLP mirroring double-slot actions.
  - `trainer.py` — training loop: data loaders, masked cross entropy, metrics, checkpoint persistence.
- `src/utils/teambuilders.py` — helpers for Showdown team rotations and loading.
- `tools/`
  - `imitation_collect.py` — async poke-env self-play data recorder.
  - `offline.py` — thin CLI entry point that instantiates `OfflineConfig` and calls `train_offline`.
  - `offline_sweep.py` — simple hyperparameter sweep wrapper.
  - `data_fetch.py` / `data_parse.py` — replay ingestion pipeline.
- `data/`
  - `processed/` — append-only datasets (`imitation.jsonl`, `human_hints.jsonl`).
  - `raw/` — downloaded Showdown logs and metadata.
- `outputs/`
  - `models/` — saved PyTorch policies (e.g., `bc_policy.pt`).
  - `tensorboard/` — TensorBoard event logs for monitoring training.
  - `plots/` — sample visualizations.
- `teams/` — curated Showdown team exports for experiments.
- `docs/` — plans, notes, and this command reference.

