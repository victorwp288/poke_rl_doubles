# Evaluation

This doc explains how evaluation works in this repo: which commands exist, what they measure, where
outputs go, and the most common pitfalls (especially vecnorm parity).

See also: [`codebase_overview.md`](./codebase_overview.md), [`ARCHITECTURE.md`](./ARCHITECTURE.md),
[`DATAFLOW.md`](./DATAFLOW.md), [`CONFIG.md`](./CONFIG.md), [`DATA_SOURCES.md`](./DATA_SOURCES.md),
[`dev_commands.md`](./dev_commands.md), [`README.md`](../README.md).

## What “evaluation” means here

Evaluation is “run a fixed number of battles under a fixed environment configuration” and summarize:

- win/loss/draw counts and win-rate
- episode reward (as defined by `src/online/env_rewards.py` and reward overrides)
- turn counts
- sanitize/repair counts (how often the safety pipeline had to intervene)

The goal is *comparability*: same environment settings, same opponent set, and consistent normalization
when VecNormalize is enabled.

## Entry point

- **CLI**: `python tools/eval_models.py ...`
- **Implementation**: `src/online/eval/cli.py`

The evaluation CLI has multiple subcommands; the default “legacy” mode is selected when you don’t
specify a subcommand.

## Common opponents

Evaluation uses a mix of scripted bots and policies:

- **Scripted**: `simple`, `maxbp`, `random` (from `poke-env` baselines)
- **Policy opponents**: can load a saved MaskablePPO checkpoint as an opponent (used for cross-play / mirror)

Opponent kinds are also referenced in config (`config/defaults.yaml`) under `evaluation.*`.

## Commands (subcommands)

### 1) Legacy (default)

If you run without a subcommand, the CLI uses the legacy evaluator:

```bash
python tools/eval_models.py \
  --policy scratch=outputs/models/maskable_ppo_scratch_best.zip \
  --policy warmstart=outputs/models/maskable_ppo_warmstart_best.zip \
  --episodes 200 \
  --opponents simple maxbp random \
  --mirror \
  --crossplay
```

Key flags:

- `--policy label=path` (repeatable)
- `--episodes N`: battles per (policy, opponent) pairing
- `--opponents simple|maxbp|random ...`: scripted baselines
- `--mirror`: play policy vs itself
- `--crossplay`: play each policy vs every other policy
- `--env-mode <mode>`: which `online.modes.<mode>` config entry to use for env settings
- `--override key=value`: patch settings, including `rewards.*` keys

Outputs: written under `outputs/eval/` (folder names/files depend on the mode and timestamp).

### 2) Suite (`suite`)

Runs a fixed “suite” against `{simple, maxbp, random, mirror}` and writes:

- `outputs/eval/eval_suite_<timestamp>.jsonl`
- `outputs/eval/eval_suite_<timestamp>_summary.json`

```bash
python tools/eval_models.py suite --policy outputs/models/maskable_ppo_warmstart.zip --episodes 50
```

### 3) PPO simple (`ppo`)

Runs a simplified PPO-vs-bots loop against scripted opponents (primarily used as a minimal sanity check):

```bash
python tools/eval_models.py ppo --policy outputs/models/maskable_ppo_scratch.zip --episodes 50 scratch
```

### 4) Compare (`compare`)

Runs a single-policy eval and writes:

- `outputs/eval/eval_<timestamp>.jsonl`
- `outputs/eval/summary_<timestamp>.csv`

```bash
python tools/eval_models.py compare --policy outputs/models/maskable_ppo_warmstart.zip --episodes 50 warmstart
```

### 5) BC (`bc`)

BC evaluation is implemented in the offline module:

- **CLI**: `python tools/eval_models.py bc ...`
- **Implementation**: `src/offline/eval_bc/cli.py`

This evaluates a BC checkpoint against scripted opponents using the BC inference wrapper and the same
core contracts (observation, mask order).

## VecNormalize parity (most common pitfall)

When VecNormalize is enabled for training (`online.use_vecnormalize: true`), training saves
`<policy_stem>_vecnorm.pkl` alongside the policy.

**Rule**: evaluation must load the matching vecnorm file for that policy, otherwise the observation
distribution changes and comparisons become misleading.

See `docs/ARCHITECTURE.md` and `docs/codebase_overview.md` for the parity contract.

## Recommended evaluation practice

- Keep `--episodes` large enough that win-rate noise is acceptable (the summary includes a Wilson CI in some paths).
- Always record:
  - policy checkpoint path(s)
  - config mode + overrides
  - opponent list
  - whether mirror/crossplay were enabled
- Treat “repair rate” as a health metric: a spike can indicate masking/action-space drift or instability.

## Debugging “weird eval”

- **Win-rate is high but reward is low**: reward shaping may not align with win outcome; inspect `env_rewards.py` and overrides.
- **Eval differs from training**: check vecnorm parity and env settings (`--env-mode`, overrides, team file).
- **Many repaired actions**: likely action mask mismatch, `act_size` mismatch, or invalid action mapping; see env logs and smoke tests.
