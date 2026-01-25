# Codebase Walkthrough (Quick Tour)

This is a short, file-level tour to orient readers quickly. For deeper context, see
`docs/ARCHITECTURE.md` and `docs/DATAFLOW.md`.

## Architecture Snapshot
- **Offline**: collect dataset → train BC → export checkpoint + stats.
- **Online**: build maskable env → optional warmstart → PPO training with masks.
- **Eval**: load policy (+ vecnorm if present) → run suites/compare/baselines.

## Key Directories
- `src/core/` — observation encoding (`observation/`), action masking (`action_mask.py`), constants.
- `src/offline/`
  - `collect/` — self-play dataset collection, batching, merging, purging.
  - `dataset/` — parsing + filters + scan utilities.
  - `model.py` — BC policy network.
  - `train/` — training CLI, grid, sweeps.
  - `eval_bc/` — BC vs bots evaluation.
- `src/online/`
  - `env.py` + `env_mask_*` — maskable env, sanitize/repair pipeline.
  - `policy/` — warmstart + policy loading.
  - `train/` — PPO init, callbacks, grid/batch orchestration.
  - `eval/` — evaluation suites and episodes.
  - `kl_ppo.py` — KL-regularized PPO extension.
- `src/data/` — replay fetch/parse utilities used by `collect_dataset`.
- `tools/` — CLI entrypoints that wire config to the modules above.
- `config/defaults.yaml` — single source of truth for defaults and modes.

## Core Contracts to Cite
- **Observation order**: `encode_observation` ordering is fixed across train/eval.
- **Mask order**: joint action masks are `[slot0 | slot1]` with length `2 * act_size`.
- **Sanitize → repair**: raw actions are clamped first, then repaired to legal non-defaults.
- **VecNormalize**: training saves `<policy>_vecnorm.pkl`; evaluation must load the same file.
- **Warmstart**: only shared trunk + action heads transfer from BC into PPO.

## Entry Points
- `tools/collect_dataset.py` — collect/merge/purge datasets or fetch/parse replays.
- `tools/offline_train.py` — BC training, sweeps, and BC-vs-bots eval.
- `tools/online.py` — PPO training, grid, and batch runs.
- `tools/eval_models.py` — PPO evaluation suites and comparisons.

## Oral Defense Quick Path
- Read `docs/ARCHITECTURE.md` + `docs/DATAFLOW.md`.
- Skim `src/core/observation/encoder.py` and `src/core/action_mask.py` for contracts.
- Skim `src/online/env_mask_repair.py` + `src/online/kl_ppo.py` for safety + regularization.
