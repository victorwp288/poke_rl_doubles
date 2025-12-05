# Copilot Instructions for poke-rl-doubles

## Project Overview
A reinforcement learning agent for **Pokémon Showdown Gen 9 Doubles OU** using **Stable-Baselines3 PPO** and **behavior cloning**. The pipeline flows: imitation data collection → offline policy training → optional online PPO fine-tuning. All configuration lives in `config/defaults.yaml`; code is split into reusable `src/` libraries and CLI entry points in `tools/`.

## Architecture & Data Flow

### Three-Stage Training Pipeline
1. **Imitation Collection** (`tools/imitation_collect.py`): Async poke-env self-play against heuristic opponents (SimpleHeuristics, MaxBasePower, Random), encodes battles into JSONL tuples: `{observation, actions, mask}`.
2. **Offline BC Training** (`tools/offline.py` + `src/offline/`): Multi-layer MLP with attention-based slot fusion. Loads JSONL from `config/defaults.yaml[offline.dataset_path]`, outputs `bc_policy.pt` and stats.
3. **Online PPO** (`tools/online.py` + `src/online/`): Warmstart from BC checkpoint or train from scratch; uses KL regularization to prevent divergence from imitation prior.

### Key Cross-File Patterns
- **Configuration**: All settings in `config/defaults.yaml`, loaded once via `src/config.load()` (cached, thread-safe). Paths derived from config blocks: `offline`, `online`, `imitation_collect`, `imitation_batches`, `evaluation`, `ppo_runs`.
- **Feature Encoding**: `src/core/features.py` (393-dim dense vector) + `slot_action_mask()` used consistently in dataset loaders, imitation collectors, online environments, and evaluation. Action space always 1+6+4×5×(gimmicks+1) per gen.
- **Reward Composition**: Modular rewards in `src/online/env.py` (win/loss/draw + faint/HP/status/terrain bonuses). Can be overridden per PPO mode via `config/defaults.yaml[online.modes.<mode>.rewards]`.

## Critical Developer Workflows

### Local Training Loop (Quickstart)
```bash
./init.sh                          # Install deps once
python tests/smoke_test_env.py      # Verify poke-env + torch device
python tools/imitation_collect.py   # Generate imitation.jsonl (~1-5 min local)
python tools/offline.py             # Train BC (config drives batch size, epochs, learning rate)
tensorboard --logdir outputs/tensorboard  # Watch loss curves in real-time
```

### Add/Modify Teams
Teams are Showdown text exports in `teams/gen9dou_*.txt`. Consumed by:
- Imitation: `config/defaults.yaml[imitation_collect.our_team_path]` + `opponent_teams_dir`
- Online: `config/defaults.yaml[online.team_path]`
- Evaluation: `config/defaults.yaml[evaluation.bc_vs_bots.our_team_path]`

Rotation via `src/utils/teambuilders.RotatingTeambuilder(teams)` for diversity; constant teams via `constant_team_from_text(path.read_text())`.

### Hyperparameter Tuning
- **Offline sweeps**: `python tools/offline_sweep.py` uses `config/defaults.yaml[offline_sweeps]` (grid over `hidden_dim`, `learning_rate`, `dropout`, etc.). Outputs per-trial `models/<trial>/bc_policy.pt` + metrics CSV.
- **Online sweeps**: `python tools/online_sweep.py` grids over `config/defaults.yaml[online_sweeps]` (modes, learning rates, entropy, timesteps). Logs to `outputs/tensorboard/online_sweep/<trial>/`.

### Evaluation & Metrics
- **BC eval**: `python tools/evaluate_bc.py` loads `bc_policy.pt`, plays episodes vs opponents in `opponent_pool`, outputs JSONL + TensorBoard.
- **PPO compare**: `python tools/run_ppo_compare.py` runs configured modes sequentially; ties checkpoints to config.
- **TensorBoard**: Aggregates offline loss, online win rate, KL penalty schedule, reward components. Inspect via `tensorboard --logdir outputs/tensorboard`.

## Project-Specific Conventions

### File Locations & Semantics
- **Config-driven paths**: Every path (datasets, models, logs) is resolved from `config/defaults.yaml` keys; avoid hardcoded `/tmp` or relative paths beyond repo root.
- **Outputs**: `outputs/models/` for checkpoints, `outputs/tensorboard/` for logs, `outputs/eval/` for evaluation JSONL, `outputs/offline_sweep/<trial>/` for sweep artifacts.
- **Data**: `data/processed/` for final JSONL (append-only until ~1 GB), `data/raw/` for downloaded Showdown logs, `data/sources/` for metadata.

### Code Style & Type Hints
- **Formatting**: Ruff, line length 100, double quotes, LF, 4-space indent. Run `ruff format . && ruff check --fix . && mypy src` before commits.
- **Imports**: Isort auto-sort (first-party = `src`). Libraries in `src/`, CLIs in `tools/`, tests in `tests/`.
- **Type hints**: Public functions in `src/` must have hints; torch/numpy code benefits from locals. Run `mypy src` to validate.
- **Constants**: `UPPER_SNAKE` for config keys and magic numbers; `snake_case` for functions/vars, `CamelCase` for classes.

### Doubles-Specific Mechanics
- **Action space**: Two-slot tuple `(action_p1, action_p2)`, each 0..N where N = 1+6+4×5×(gimmicks+1). Always mask-constrained (forced switches, switch limits, fainted mons).
- **Observations**: `encode_observation(battle)` produces 393-dim vector covering: per-Pokémon HP/status/types, team composition, field conditions (weather, terrain, hazards, screens), turn count, legal action hints. Cached per battle object.
- **Masks**: `slot_action_mask(battle)` returns list of 2 lists, one per slot. Each entry is 1 if legal, 0 if not. Used in training, evaluation, and PPO masking.

### Environment Wrappers (poke-env Integration)
- **Doubles base**: `poke_env.environment.DoublesEnv` extended by `src/online/env.Gen9DoublesEnv` for observation encoding + reward accumulation.
- **Action masking**: `src/online/env.MaskableDoublesEnv` wraps base env, exposes `action_masks()`, repairs invalid actions, enforces step timeouts, logs to `outputs/logs/online_env.log`.
- **Vec wrappers**: `make_maskable_env()` constructs parallel environment via `SubprocVecEnv` (if parallel_battles > 1) or `DummyVecEnv` (single).

### Offline Training Details
- **Architecture**: `BehaviorCloningPolicy` in `src/offline/model.py` is MLP + LayerNorm input + MultiheadAttention over shared context → per-slot linear heads. Attention learns per-slot embeddings via trainable `slot_queries`.
- **Training loop** in `src/offline/trainer.py`: loads dataset, splits train/val, iterates epochs, computes masked cross-entropy per slot, applies label smoothing, logs per-epoch loss to TensorBoard.
- **Best checkpoint**: Tracked separately as `bc_policy_best.pt` + `bc_stats_best.json` if `best_policy_path` is set in config; enables model selection during sweeps.

### Online PPO Extensions
- **KL regularization**: `src/online/kl_ppo.KLRegularizedMaskablePPO` anneals KL penalty from `kl_coef_start` → `kl_coef_final` over `kl_anneal_steps` updates. Computed against frozen reference policy (either BC checkpoint or current policy after warmup).
- **Warmstart loader**: `src/online/init.py` maps BC checkpoint tensors (shared layers + heads) into PPO policy, preserving feature scaling. Validates obs dims, skips missing stats.
- **Step delay**: `src/online/env.py` can enforce minimum step interval via `step_delay` to slow down fast local simulators.

## Integration Points & Dependencies

### External Libraries
- **poke-env** 0.10.0: Doubles battles, Showdown server interaction, team parsing. Avoid direct `Battle` calls; use wrapper methods (`encode_observation`, `slot_action_mask`).
- **Stable-Baselines3 + sb3-contrib** (PPO, Maskable PPO): Provide RL algorithms. Custom KL penalty applied on top via monkey-patching `train()` method.
- **PyTorch** 2.3+: Model definitions, loss computation. Device auto-selection via `"cuda" if available else "cpu"`.
- **TensorBoard**: Logs via writer instantiated in `OfflineConfig.tensorboard_dir`, populated by `train_offline()` and PPO callbacks.

### Async Patterns
- **Imitation collection**: `asyncio` event loop in `imitation_collect.py` + `imitation_batch.py` to parallelize battles across multiple replicas + opponents.
- **Config caching**: Thread-safe loader in `src/config.py` with `_CACHE_LOCK` to support concurrent CLI invocations.

### Server Configuration
- **Localhost**: Default `http://localhost:8000` in config. Fast iteration; requires local Showdown server.
- **Public Showdown**: `https://play.pokemonshowdown.com` available via config override. Rate-limited (0.5 req/sec default).
- **Custom WebSocket**: URL parsing in `imitation_collect.py` converts HTTP → WS endpoints auto.

## Testing & Validation
- **Smoke test**: `python tests/smoke_test_env.py` verifies imports, torch device, connectivity, matplotlib/seaborn plotting.
- **Unit tests**: `tests/test_action_repair.py` isolates specific utilities.
- **No CI/CD runners visible**: Expect manual testing before commits; pre-commit hooks optional.

## Common Pitfalls & Workarounds
1. **Stale config cache**: If you edit `config/defaults.yaml`, restart Python process (cache is per-session).
2. **Action mask shape mismatches**: Always validate mask dimensions in environment wrappers vs. policy heads; `combine_slot_masks()` is defensive.
3. **Observation encoding changes**: If `FeatureConfig` constants drift, retrain BC models; PPO warmstarts may fail dimension checks.
4. **Async battle timeouts**: Increase `imitation_collect.battle_timeout` if Showdown server is slow; default 60s.
5. **Team parsing errors**: Showdown text format is strict; `RotatingTeambuilder` silently skips malformed entries—check logs if teams aren't loading.
