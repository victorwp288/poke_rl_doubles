# Essential Commands & File Map

See also: [`codebase_overview.md`](./codebase_overview.md), [`ARCHITECTURE.md`](./ARCHITECTURE.md),
[`DATAFLOW.md`](./DATAFLOW.md), [`CONFIG.md`](./CONFIG.md), [`README.md`](../README.md).

## Terminal Cheat Sheet
- `./init.sh` — install project deps with the pinned versions.
- `pip install -r requirements.txt` — manual install alternative.
- `python tests/smoke_test_env.py` — verify environment, poke-env, and basic rendering.

- `python tools/collect_dataset.py [collect|batch|merge|purge|fetch|parse]` — dataset collection and
  replay ingestion; see `config/defaults.yaml` for defaults.

- `python tools/offline_train.py [train|grid|sweep|eval-bc]` — BC training, sweeps, and BC-vs-bots eval.

- `python tools/online.py [train] MODE [--override key=value ...]` — PPO training (scratch/warmstart).
- `python tools/online.py grid [--limit N] [modes ...]` — PPO grid search.
- `python tools/online.py batch [modes ...]` — run multiple PPO modes sequentially.

- `python tools/eval_models.py [compare|suite|ppo|bc] ...` — evaluation entrypoint (see `--help`).
- `python showdown_visual/run_showdown_battle.py --model PATH [--log-level summary|verbose]`
  — live local Showdown battle visualizer (prints spectate URL).

- `tensorboard --logdir outputs/tensorboard` — inspect offline/online training metrics.
- `ruff format .` / `ruff check --fix .` / `mypy src tools` — formatting, linting, typing.

## Project Layout Overview
- `src/core/` — observation encoding + action masks.
- `src/offline/`
  - `collect/` — dataset collection runners.
  - `dataset/` — parsing, filtering, scanning, and indexed access.
  - `model.py` — BC policy network.
  - `train/` — offline training CLI + sweeps.
  - `eval_bc/` — BC evaluation vs bots.
- `src/online/`
  - `env.py` + `env_mask_*` — maskable env, sanitize/repair pipeline.
  - `policy/` — warmstart + policy loading.
  - `train/` — PPO model init, callbacks, grid/batch.
  - `eval/` — PPO evaluation suite.
  - `kl_ppo.py` — KL-regularized PPO.
- `src/data/` — replay fetch + parse utilities used by `collect_dataset`.
- `tools/` — thin CLI entrypoints wiring config to the modules above.
- `showdown_visual/` — local Showdown battle visualizer adapter.
- `data/` — datasets and replay artifacts.
- `outputs/` — checkpoints, logs, tensorboard, eval summaries.
- `teams/` — curated Showdown team exports.
- `docs/` — architecture, dataflow, config guide, and walkthrough.

## Next steps

- For the **big-picture story + diagrams**: [`codebase_overview.md`](./codebase_overview.md)
- For the **contracts**: [`ARCHITECTURE.md`](./ARCHITECTURE.md)
- For the **pipelines**: [`DATAFLOW.md`](./DATAFLOW.md)
- For the **defaults/knobs**: [`CONFIG.md`](./CONFIG.md)
