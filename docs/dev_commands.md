# Essential Commands & File Map

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
- `data/` — datasets and replay artifacts.
- `outputs/` — checkpoints, logs, tensorboard, eval summaries.
- `teams/` — curated Showdown team exports.
- `docs/` — architecture, dataflow, config guide, and walkthrough.
