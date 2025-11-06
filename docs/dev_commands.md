# Essential Commands & File Map

## Terminal Cheat Sheet
- `./init.sh` — install project deps with the pinned versions.
- `pip install -r requirements.txt` — alternative manual install of Python deps when the init script is unavailable.
- `python tests/smoke_test_env.py` — verify environment, poke-env, and basic rendering.
- `python tools/imitation_collect.py` — generate imitation JSONL (set teams and counts in `config/defaults.yaml`).
- `python tools/imitation_merge.py` — concatenate shard outputs defined in `imitation_merge`.
- `python tools/imitation_batch.py` — launch the configured collector batches (`imitation_batches`).
- `python tools/offline.py` — train behavior cloning policy; uses `config/defaults.yaml` and logs to `outputs/tensorboard/`.
- `python tools/offline_sweep.py` — run BC sweeps; outputs land in `outputs/offline_sweep/<trial>/` with policy, stats, metrics.
- `python tools/online.py [mode]` — run Maskable PPO (`mode` options live in `config/defaults.yaml`).
- `policy_hidden_dim` / `policy_hidden_layers` keep the PPO MLP aligned with the offline BC model for warmstarts.
- `python tools/evaluate_bc.py` — load the BC checkpoint and play evaluation bouts (JSONL + optional TensorBoard).
- `python tools/run_ppo_compare.py` — execute the configured PPO modes sequentially (defaults to warmstart then scratch).
- `python tools/online_warmstart.py` / `python tools/online_scratch.py` — convenience wrappers that call `tools/online.py`.
- `tensorboard --logdir outputs/tensorboard` — inspect offline/online training losses, win rate, and per-battle stats.
- `ruff format .` / `ruff check --fix .` / `mypy src` — formatting, linting, typing (see more in README).

## Project Layout Overview
- `src/core/` — environment utilities (`env.py` exposes `action_space_size`, etc.).
- `src/offline/`
  - `dataset.py` — JSONL loader and validation helpers.
  - `model.py` — `BehaviorCloningPolicy`, two-head MLP for double-slot actions.
  - `trainer.py` — training loop: data loaders, masked cross entropy, metrics, checkpoint persistence.
- `src/online/`
  - `env.py` — poke-env doubles wrapper with reward stats, action masks, and helpers for multi-env logging.
  - `init.py` — utilities to load behavior cloning weights into MaskablePPO policies.
- `src/utils/teambuilders.py` — helpers for Showdown team rotations and loading.
- `tools/`
  - `imitation_collect.py` — async poke-env self-play data recorder.
  - `offline.py` — thin CLI entry point that reads defaults from `config/defaults.yaml` and calls `train_offline`.
  - `offline_sweep.py` — structured sweep runner that logs policies, stats, and metrics per trial.
  - `online.py` — Maskable PPO runner driven by YAML modes (scratch, warmstart, etc.).
  - `online_warmstart.py` / `online_scratch.py` — preset wrappers for quick runs.
  - `imitation_batch.py` — orchestrates multi-shard self-play collection based on `imitation_batches`.
  - `imitation_merge.py` — merges shard outputs into the dataset path for BC.
  - `evaluate_bc.py` — evaluates the BC checkpoint against baseline opponents.
  - `run_ppo_compare.py` — runs the configured PPO modes sequentially.
  - `data_fetch.py` / `data_parse.py` — replay ingestion pipeline.
- `data/`
  - `processed/` — append-only datasets (`imitation.jsonl`, `human_hints.jsonl`).
  - `raw/` — downloaded Showdown logs and metadata.
- `outputs/`
  - `models/` — saved PyTorch policies (e.g., `bc_policy.pt`, Maskable PPO checkpoints).
  - `tensorboard/` — TensorBoard event logs; online runs write train/eval win rate, KO stats, reward schedule trends.
  - `plots/` — sample visualizations.
- `teams/` — curated Showdown team exports for experiments.
- `docs/` — plans, notes, and this command reference.
