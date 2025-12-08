# Poke RL Doubles Codebase Walkthrough

This guide explains every file in the project so new and senior teammates can narrate the codebase with confidence. Each entry calls out what the file is for, how it works, and why it exists. Expect a focused 30‑minute read that you can revisit as a set of talking points.

## How to Use This Document
- Skim the **Architecture Snapshot** to understand the moving pieces.
- Use the directory sections (`src/`, `tools/`, `config/`, etc.) when you need a reminder of how parts connect.
- The **Representative Code Examples** highlight critical mechanics you can cite in discussions or onboarding sessions.

## Architecture Snapshot
- **Training loop**: imitation data → offline behavior cloning (`src/offline/`) → optional warm-start for online PPO (`src/online/` + `tools/online.py`).
- **Environment stack**: poke-env doubles wrapper with dense observation encoding and action masking (`src/core/` + `src/online/env.py`).
- **Supporting CLIs**: scripts in `tools/` orchestrate data collection, sweeps, and evaluation, all configured through `config/defaults.yaml`.
- **Assets**: `teams/` holds Showdown exports, `data/` stores datasets, `outputs/` captures checkpoints, logs, and metrics.
- **Documentation**: `docs/` provides command references, reading lists, incident writeups, and compute planning.

## Root-Level Files

### `init.sh`
- **What**: Bootstrap script for new machines.
- **How**: Upgrades `pip`, installs the core dependency set one package at a time (skipping the stdlib `asyncio` wheel), then installs optional web viewer and database extras.
- **Why**: Useful when provisioning remotes where `pip install -r requirements.txt` is not desirable; keeps dependency order explicit for reproducibility.

### `requirements.txt`
- **What**: Dependency manifest matching the packages installed by `init.sh`.
- **How**: Lists core libraries first (PyTorch, Stable-Baselines3 stack, poke-env) followed by tooling and optional extras.
- **Why**: Still the canonical source for dependency updates and compatible with existing workflows (`pip install -r requirements.txt`).
## Source Library (`poke_rl_doubles/src/`)

### `src/__init__.py`
- **What**: Marker file so `src/` is importable.
- **How**: Contains a single comment; no runtime code.
- **Why**: Allows `src` to be treated as a package when referenced from tooling or tests.

### `src/config.py`
- **What**: YAML-backed configuration loader with caching.
- **How**: Lazily reads `config/defaults.yaml` via `_read_config`, caches the dict behind `_CACHE`, and provides `load()` and `section()` helpers guarded by a threading lock. `load()` always returns a shallow copy so callers can mutate settings without polluting the global cache, while `section(name)` returns a defensive clone of a single section (or `{}` if missing).
- **Why**: Centralises configuration access so every CLI and module read consistent settings without re-parsing disk. The lock + cache prevent thrashing when multiple async collectors request config simultaneously, and the copy semantics make it safe to hand configs to subprocess factories.

### `src/core/env.py`
- **What**: Size calculation helper for the action space.
- **How**: `action_space_size()` lowercases the Showdown battle format, consults a `gimmicks_map` (Terastal counts per generation), and returns `1 + 6 + 4 * 5 * (gimmicks + 1)`—a deterministic bound on the double-slot action vector.
- **Why**: Online wrappers use it to dimension action masks before poke-env exposes real spaces. Having this cheap helper keeps vector sizes in sync across dataset parsing, PPO masks, and evaluation repair logic.

### `src/core/features.py`
- **What**: Observation encoding and action masking utilities.
- **How**:
  - `FeatureConfig` defines constants (weather slots, side conditions, per-mon feature sizes) plus ordering tuples that keep vector positions deterministic.
  - `ObservationEncoder` computes a 393-dim feature vector covering HP, status, terrain, team composition, coverage, and legal action counts. Helper methods such as `_global_state`, `_per_mon`, `_type_matchups`, and `_priority_flags` each focus on one aspect of the battlefield; the encoder also caches generation-specific constants via `_gen_data_cache`.
  - Functions `slot_action_mask()` and `combine_slot_masks()` convert poke-env legal orders into per-slot mask vectors, reusing `_legal_orders()` for fallback search when poke-env raises. The mask routine handles forced switches, detects default/“/choose pass” orders, and guarantees at least one legal action.
- **Why**: Stable-Baselines3 expects dense numeric observations and valid-action masks; consistent encoding is critical for both BC and PPO. These helpers also feed the imitation collector (so recorded masks match runtime masks) and the evaluation repair path.

### `src/utils/teambuilders.py`
- **What**: Helpers for reading and serving Pokémon Showdown team exports.
- **How**: Provides IO functions (`read_showdown_team`, `load_showdown_teams_from_dir`), a `RotatingTeambuilder` that parses text into battle-ready teams via poke-env base methods (`parse_packed_team`, `parse_showdown_team`), filters out malformed entries, and exposes `yield_team()` for thread-safe random selection. `constant_team_from_text` returns poke-env’s `ConstantTeambuilder`.
- **Why**: Training and evaluation scripts need deterministic or diverse teams without re-parsing in every CLI. Encapsulating the parsing logic here prevents repeated boilerplate and keeps team sanitisation consistent across collectors and opponents.

### `src/online/__init__.py`
- **What**: Export surface for online utilities.
- **How**: Defines `__all__` to re-export env helpers.
- **Why**: Simplifies imports elsewhere (`from src.online import make_maskable_env` etc.).

### `src/online/env.py`
- **What**: poke-env doubles environment wrappers with reward diagnostics and action masking.
- **How**:
  - Utility functions compute reward components (`_reward_metrics`, `_score_from_metrics`), build masks (`_battle_mask`), and generate cache keys for stat lookup. Helpers like `_team_hp_fraction` and `_status_count` are shared between reward shaping and logging.
  - `Gen9DoublesEnv` subclasses `DoublesEnv` to produce encoded observations via `encode_observation`, accumulate per-battle stats in `_score_cache`, and expose `set_rewards()` to live-tune reward weights. `latest_stats()` retrieves the last emitted metrics for dashboard purposes.
  - `MaskableDoublesEnv` wraps the base env to expose `action_masks`, sanitize invalid actions, repair unsafe inputs, enforce timeouts, and log to `outputs/logs/online_env.log`. It tracks step timing to honour `step_delay`, reshapes mask arrays defensively, and can generate fallback actions when both slots are forced.
  - `make_maskable_env()` constructs the composite env with optional reward overrides and step delays. It instantiates poke-env `AccountConfiguration`s for player/opponent, respects per-mode rate limits, and chooses between `SubprocVecEnv` and `DummyVecEnv` depending on parallel battle count.
- **Why**: Stable-Baselines3’s Maskable PPO needs reliable mask information and reward shaping to learn meaningful policies; logging and sanitisation guard against poke-env edge cases. Centralising this logic keeps PPO, evaluation, and collectors in sync when Showdown protocol changes.

### `src/online/init.py`
- **What**: Bridges offline checkpoints into online PPO policies.
- **How**: Loads PyTorch checkpoints, validates observation dimensions, maps shared linear layers, copies policy/value head tensors, and returns normalisation stats via `NormalizationStats`. Helper functions `_policy_device`, `_shared_layer_pairs`, `_head_tensors`, and `_apply_action_heads` isolate tensor selection so the loader can support future head variants without duplicating logic.
- **Why**: Enables PPO warm-starts from behaviour cloning without manual tensor surgery, preserving feature scaling learned offline. The module also gracefully handles missing stats files by falling back to metadata embedded in checkpoints.

### `src/online/kl_ppo.py`
- **What**: Custom Stable-Baselines3 policy implementing KL annealing.
- **How**: `KLRegularizedMaskablePPO` subclasses `MaskablePPO`, tracks KL reference policies, computes penalties against a frozen copy, and gradually anneals coefficients across updates while recording training stats. Methods `set_reference_policy`, `set_reference_from_current`, `_current_kl_coef`, and `_kl_penalty` wrap the book-keeping, while `train()` injects KL loss into the SB3 update loop, records entropy/policy/value stats, and respects early stopping if `target_kl` is exceeded.
- **Why**: Encourages policies to stay near the behaviour prior early in training, reducing divergence when masks or rewards shift. Having it live in-source allows tweaks (e.g., new schedules) without forking sb3-contrib.

### `src/offline/__init__.py`
- **What**: Export declarations for offline modules.
- **How**: Uses `__all__` to expose dataset/model/trainer symbols.
- **Why**: Simplifies consumer imports for CLI scripts.

### `src/offline/dataset.py`
- **What**: JSONL dataset loader and validator for imitation samples.
- **How**: Parses per-line payloads, converts observations/masks to lists, ensures two-slot actions obey masks, and optionally shuffles/bounds samples; `split_train_val()` slices off a validation subset. Helper functions `_as_float_list`, `_as_int_list`, `_as_mask`, and `_valid_actions` convert raw JSON (which may contain strings or nested lists) into strict numeric tensors while rejecting malformed records early.
- **Why**: Guarantees the behaviour cloning trainer receives consistent, legal data even when source JSON is noisy. Having these conversions close to file IO makes it easier to enforce invariants before samples hit PyTorch.

### `src/offline/model.py`
- **What**: Behaviour cloning network definition.
- **How**: `_shared_layers()` builds stacked linear → ReLU → Dropout blocks; `BehaviorCloningPolicy` normalises inputs with `LayerNorm`, runs shared MLP, applies multi-head attention to derive two slot-specific contexts, and emits logits for each slot via separate linear heads. Trainable `slot_queries` learn per-slot embeddings and `slot_norm` stabilises attention outputs.
- **Why**: Captures correlations between team slots while supporting shared feature processing; attention makes both action heads aware of the same context. LayerNorm on both inputs and slot projections combats scale drift between observation features and downstream heads.

### `src/offline/trainer.py`
- **What**: Main offline training loop plus utilities.
- **How**:
  - `OfflineConfig` normalises CLI/YAML inputs and resolves paths for checkpoints/stats while keeping raw data for provenance (`as_dict()`).
  - Helper functions build PyTorch datasets (`_SampleDataset`), collate batches with masks (`_batch_collate`), compute feature statistics (`_feature_stats`, `_binary_feature_flags`), apply normalisation noise, pick devices, and open TensorBoard writers.
  - `train_offline()` orchestrates sampling, normalisation, dataloaders, model initialisation, optimizer setup, per-epoch training/evaluation, checkpointing (current + best), and returns aggregated metrics. It also emits TensorBoard scalars, prints per-epoch summaries, and writes normalisation stats alongside checkpoints via `_write_stats`.
- **Why**: Provides a turnkey BC trainer that other scripts can call with minimal boilerplate, ensuring checkpoints and stats are always written in a consistent format. Encapsulation here keeps command-line entry points slim.

### `src/config.py`, `src/core/features.py`, and `src/offline/trainer.py` interplay
- Config sections define dataset paths and hyperparameters.
- Feature encoder outputs 393-length observations consumed by the BC and PPO policies.
- Trainer writes checkpoints consumed by `src/online/init.py` for warm-starting Maskable PPO.

## CLI & Automation Scripts (`poke_rl_doubles/tools/`)

### Offline training & sweeps
- **`tools/offline.py`**: Thin entry point that reads the `offline` config section, prints run metadata (dataset path, splits, epochs), and calls `train_offline()` directly. Serves as the canonical CLI invoked from docs and CI.
- **`tools/offline_grid.py`**: Executes grid sweeps over offline config parameters defined in `offline_sweeps`. Generates unique output directories via `_timestamp()`, honours an optional `--limit`, and writes JSON summaries to `outputs/sweeps/offline/`.
- **`tools/offline_sweep.py`**: Hard-coded trial list (`TRIALS`) for quick baselines. `run_trial()` merges overrides, launches `train_offline()`, and writes both metrics and error traces under `outputs/offline_sweep/<trial>/`.

### Online PPO orchestration
- **`tools/online.py`**: Core runner for Maskable PPO. `_mode_settings()` merges defaults with mode overrides, `_apply_overrides()` injects CLI overrides, `_build_env()` wires poke-env accounts, and `_load_bc_if_requested()` imports BC weights. The script configures callbacks (rolling checkpoints, EvalCallback + `CopyBestModelCallback`), seeds numpy/torch, resolves devices, and finally calls `model.learn()` with progress bar + TensorBoard naming.
- **`tools/online_grid.py`**: Sweeps PPO hyperparameters defined in `online_sweeps`, computing cartesian products of value lists, piping each combo to `run()`, and writing aggregate results under `outputs/sweeps/online/`. Uses `_prepare_overrides()` to keep overrides serialisable.
- **`tools/online_scratch.py`** / **`tools/online_warmstart.py`**: One-line convenience wrappers that call `run("scratch")` or `run("warmstart")`, letting teammates launch common modes without remembering flags.
- **`tools/run_ppo_compare.py`**: Sequentially runs configured modes (default warmstart then scratch), catching exceptions per mode for resilient dashboards and logging `[ppo]` prefixes for easy grepping.

### Imitation data pipeline
- **`tools/imitation_collect.py`**: Asynchronous collector that spins up a teacher player, records encoded observations plus legal action masks with `Recorder` rotation, and cycles through opponent pools. `Settings` wraps config-derived options, `RecordingHeuristics` extends poke-env heuristics to record every decision, and `play_dataset()` orchestrates battles with timeout protection and opponent cleanup.
- **`tools/imitation_batch.py`**: Launches multiple collectors concurrently based on `imitation_batches` config, handling output paths, seeding, and semaphore-based concurrency. `_build_settings()` resolves per-replica filenames and seeds, while `_run_batch()` awaits replicas and streams progress to stdout.
- **`tools/imitation_merge.py`**: Merges shard files (wildcards allowed) into a single JSONL dataset, streaming with a fixed chunk size (`CHUNK`) to handle large files without blowing RAM. `_resolve_source()` supports glob patterns and directories.

### Replay ingestion & parsing
- **`tools/data_fetch.py`**: Fetches replays from showdown using robots.txt-friendly HTTP requests, iterative rate limiting, and maintains an index of fetched artefacts. Helper functions normalise tokens (`normalize_token`), deduplicate IDs, and support best-effort user searches; the main loop respects robots.txt and writes downloaded files plus `index.json`.
- **`tools/data_parse.py`**: Parses downloaded replay blobs to extract turn-level hints (moves, switches) into structured JSONL entries for human-in-the-loop analysis. Regex patterns (`MOVE_LINE`, `SWITCH_LINE`, `TURN_LINE`) recognise log events, while `_hint_from_move`/`_hint_from_switch` build structured payloads containing turn, side, and tactic hints.

- **`tools/evaluate_bc.py`**: Loads the behaviour cloning checkpoint, builds opponents, sanitises/repairs invalid actions, logs diagnostics, and writes evaluation metrics. Contains robust mask handling for poke-env edge cases (`_repair_action`, `_sanitize_action`, `_joint_legal_actions`) and emits watchdog/timeouts to avoid hanging battles.

### Shared helpers inside `tools/online.py`
- `RollingCheckpointCallback` and `CopyBestModelCallback` manage checkpoint rotation and best-model syncing.
- `_mode_settings()` merges YAML defaults with mode overrides.
- `_build_env()` wraps env creation, wiring poke-env accounts, rate limits, and maskable wrappers.

## Configuration & Data Assets

### `poke_rl_doubles/config/defaults.yaml`
- **What**: Single source of truth for offline, online, imitation, evaluation, and sweep parameters.
- **How**: Structured by top-level sections (`offline`, `data_fetch`, `imitation_collect`, `online`, `imitation_batches`, `evaluation`, `ppo_runs`, `offline_sweeps`, `online_sweeps`). Each section mirrors CLI expectations, e.g., `online.modes` defines named PPO presets, `offline` pins hyperparameters, and sweep sections list value grids.
- **Why**: Every CLI reads from here, so changing the YAML updates collectors, trainers, and PPO runs without editing code. Keeping defaults centralised lets us diff experiment overrides against a single baseline.

### `poke_rl_doubles/data/`
- `sources/replays/`: Contains `README.md`, `ids.txt`, `urls.txt` for curated replay sources. `ids.txt` and `urls.txt` feed `tools/data_fetch.py`; the README reminds contributors how to gather replays safely.
- `processed/`: Destination for generated JSONL datasets (behaviour cloning data, parsed hints) created by collectors and parsers. Downstream training consumes `imitation.jsonl` from here.
- `raw/`: Holds downloaded replay artefacts; scripts assume append-only storage and rely on consistent naming (`<replay-id>.json|.log|.html`).

### `poke_rl_doubles/teams/`
- `README.md`: Describes available Gen9 Doubles OU teams with short strategic blurbs.
- `*.txt`: Showdown export text files representing baseline and variant teams (rain, sun, sand, trick room, hyper-offense, etc.). Each is plug-and-play with poke-env teambuilders.
- **Why**: Training and evaluations use these to diversify matchups or fix a baseline roster. Rotating opponents via these files prevents overfitting to a single archetype.

## Tests

### `poke_rl_doubles/tests/smoke_test_env.py`
- **What**: Manual integration test for environment setup.
- **How**: Imports required libraries (`REQUIRED_IMPORTS`), checks torch devices (`check_torch_devices`), attempts to connect to a Showdown server via `connect_showdown()`, and produces sample plots (`write_plots()`) to validate matplotlib.
- **Why**: Quick verification that the local environment, server configuration, and plotting libraries work before expensive runs. Ideal smoke test after provisioning new machines or containers.

### `poke_rl_doubles/tests/test_action_repair.py`
- **What**: Unit tests for action sanitisation and repair helpers.
- **How**: Builds stub environments, verifies forced-switch handling, ensures defaults are skipped when illegal, and tests `slot_action_mask()` edge cases. Covers the safety net around `_repair_action`, default moves, and forced switch fallback logic.
- **Why**: Guards critical mask logic so training/evaluation don’t regress when mask definitions change. If these tests fail, the PPO/evaluation runners likely produce illegal orders again.

## Documentation & Notes

### `poke_rl_doubles/docs/dev_commands.md`
- Cheat sheet of CLI commands, layout overview, environment variables, and logging destinations for day-to-day development. Useful first-stop reference when running collectors, trainers, or evaluation scripts.

### `poke_rl_doubles/docs/reading.md`
- Curated reading list covering poke-env internals, PPO references, and academic work on competitive Pokémon AI (plus adjacent RL achievements like AlphaStar). Helps new contributors ramp on domain-specific literature.

### `poke_rl_doubles/docs/evaluate_bc_fixes.md`
- Incident report capturing root cause and fixes for `evaluate_bc.py` hangs, including snippets for clamps, validation, diagnostics, and the observed failure modes. Serves as institutional memory for mask-related regressions.

### `poke_rl_doubles/docs/compute_estimate.md`
- Compute planning memo with local baseline measurements, cluster sizing recommendations, and time estimates for BC/PPO runs. Includes hardware option tables, utilisation assumptions, and success criteria for scaling up.

### `poke_rl_doubles/docs/codebase_walkthrough.md`
- This document. Keep it updated when modules move or new subsystems land; treat it as the living tour script for onboarding sessions.

### `poke_rl_doubles/docs/delegation_task_ideas.md`
- Curated backlog of bite-sized delegation candidates grouped by theme (testing, docs, tooling, data, training, infra). Handy when distributing chores or onboarding interns.

## Outputs & Working Directories
- `poke_rl_doubles/outputs/`: Structured outputs (`models/`, `logs/`, `tensorboard/`, `eval/`, `plots/`) where training and evaluation scripts write artefacts. Checkpoints land under `models/`, sweep summaries under `eval/` or `logs/`, and visualisations in `plots/`.
- `poke_rl_doubles/runs/`: Reserved for experiment metadata or future run logs (currently empty but a good home for hydra/wandb exports if adopted).
- Rolling log files such as `poke_rl_doubles/outputs/logs/online_env.log` are created on demand by the online environment wrapper, with rotation handled inside `MaskableDoublesEnv`.

## Representative Code Examples

### Observation encoding pipeline
```python
# src/core/features.py: ObservationEncoder.encode
def encode(self, battle):
    features = []
    features.extend(self._base_slots(battle))
    features.extend(self._global_state(battle))
    player_slots = list(battle.active_pokemon)
    opponent_slots = list(battle.opponent_active_pokemon)
    for mon in player_slots:
        features.extend(self._per_mon(mon, opponent_slots))
    for mon in opponent_slots:
        features.extend(self._per_mon(mon, player_slots))
    features.extend(self._type_matchups(battle, player_slots, opponent_slots))
    features.extend(self._priority_flags(player_slots, opponent_slots))
    features.extend(self._fake_out_flags(player_slots))
    features.extend(self._type_coverage(battle.team))
    features.extend(self._type_coverage(battle.opponent_team))
    features.extend(self._legal_action_counts(battle))
    return self._pad(features)
```
*Why it matters*: Demonstrates the feature ordering that every policy and dataset relies on; useful when auditing observation mismatches or extending features.

### Online action sanitisation
```python
# src/online/env.py: MaskableDoublesEnv._sanitize_action
def _sanitize_action(self, action):
    space = self._action_space()
    if space is None:
        return np.asarray(action), False

    vector = self._vector(action)
    slots = len(space.nvec)
    if vector.size < slots:
        vector = np.pad(vector, (0, slots - vector.size), constant_values=0)

    mask = self.get_action_mask()
    if mask is None:
        mask = np.ones(self._mask_shape, dtype=bool)
    mask = np.asarray(mask, dtype=bool).reshape(-1, self._act_size)

    cleaned = vector[:slots].copy()
    changed = False
    for idx, limit in enumerate(space.nvec):
        limit = int(limit)
        slot_mask = mask[idx] if idx < mask.shape[0] else np.ones(self._act_size, dtype=bool)
        legal = np.where(slot_mask)[0]
        if legal.size == 0:
            legal = np.arange(min(self._act_size, limit))
        choice = cleaned[idx] if idx < cleaned.size else 0
        if choice < 0 or choice >= limit or choice >= self._act_size or not slot_mask[choice]:
            cleaned[idx] = int(np.random.choice(legal))
            changed = True

    return cleaned.reshape(space.shape), changed
```
*Why it matters*: Protects the PPO runner from poke-env mis-specifications and BC warm-start glitches by always returning legal actions.

### Offline training loop
```python
# src/offline/trainer.py: train_offline (excerpt)
def train_offline(settings):
    config = OfflineConfig(settings)
    device = _pick_device(config.device)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    samples = load_samples(
        config.dataset_path,
        max_samples=config.max_samples,
        seed=config.seed,
        shuffle=config.shuffle,
    )
    train_samples, val_samples = split_train_val(samples, config.val_fraction)
    if not train_samples:
        raise ValueError("training split must contain samples")

    mean, std = _feature_stats(train_samples)
    binary_flags = _binary_feature_flags(train_samples)
    _normalize_samples(train_samples, mean, std)
    _normalize_samples(val_samples, mean, std)
    ...
```
*Why it matters*: Shows the end-to-end process—data loading, normalisation, dataloaders, model creation, logging, checkpointing—that any future trainer modification must respect.

### Configuration helper
```python
# src/config.py: section
def section(name, path=None):
    config = load(path)
    value = config.get(name, {})
    if isinstance(value, dict):
        return dict(value)
    return value
```
*Why it matters*: Central helper that every CLI uses to pull settings. The defensive copy ensures downstream code can mutate configs (e.g., add derived paths) without contaminating the global cache.

### Dataset validation
```python
# src/offline/dataset.py: _parse_payload
def _parse_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError
    observation = _as_float_list(payload.get("observation"))
    actions = _as_int_list(payload.get("action"))
    mask = _as_mask(payload.get("mask"))
    if len(actions) != 2:
        raise ValueError
    if not _valid_actions(actions, mask):
        raise ValueError
    return observation, actions, mask
```
*Why it matters*: Highlights the guard rails around raw JSONL data—every sample is checked before reaching PyTorch, keeping BC training stable even when collectors hiccup.

---

Keep this walkthrough close while onboarding collaborators or planning refactors—it ties together the why, how, and where of every file so you can reason about changes with context.
