# Architecture

This document is the high-level map of the system. It describes the major components and the
contracts that must stay stable for training and evaluation parity.

## System Overview
- **Offline stage (behavior cloning)**: Collect masked state/action pairs, train a BC policy, and
  export normalization stats and checkpoints.
- **Online stage (maskable PPO)**: Use the BC checkpoint to warmstart PPO, train with action masks
  and optional KL regularization, and save policies plus VecNormalize stats.
- **Evaluation**: Load policies (and matching VecNormalize stats) to compare against heuristics,
  cross-play, and regression baselines.

## Core Contracts (Do Not Break)
- **Observation ordering**: `encode_observation` emits a fixed-order vector; reordering breaks
  dataset/checkpoint parity.
- **Action mask layout**: joint masks are flattened as `[slot0 | slot1]` with length `2 * act_size`.
- **Sanitize → repair**: raw actions are sanitized into bounds, then repaired to legal non-default
  actions when possible; info flags distinguish both steps.
- **VecNormalize parity**: evaluation must load the `<policy>_vecnorm.pkl` saved during training.
- **Warmstart scope**: only shared trunk + action heads transfer from BC into PPO.

## Module Map
- `src/core/`: observation encoding, action masking, feature constants.
- `src/offline/`: data collection (`collect/`), dataset parsing (`dataset/`), BC model + training
  (`model.py`, `train/`), BC evaluation (`eval_bc/`).
- `src/online/`: maskable env (`env.py`, `env_mask_*`), PPO model (`kl_ppo.py`), warmstart/loading
  (`policy/`), training/eval helpers (`train/`, `eval/`).
- `src/data/`: replay fetch/parse utilities for dataset enrichment.
- `tools/`: CLI entrypoints that glue configs to the above modules.

## Key Entry Points
- `tools/collect_dataset.py` — collect/merge/purge datasets or fetch/parse replays.
- `tools/offline_train.py` — BC training, sweeps, and BC-vs-bots evaluation.
- `tools/online.py` — PPO training (scratch/warmstart), grid/batch runs.
- `tools/eval_models.py` — PPO evaluation suite and comparisons.
