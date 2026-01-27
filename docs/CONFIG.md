# Config Guide (`config/defaults.yaml`)

This file is the single source of truth for offline/online training, collection, and evaluation.
CLI overrides patch these settings at runtime but should keep the same keys.

See also: [`codebase_overview.md`](./codebase_overview.md), [`ARCHITECTURE.md`](./ARCHITECTURE.md),
[`DATAFLOW.md`](./DATAFLOW.md), [`EVALUATION.md`](./EVALUATION.md), [`DATA_SOURCES.md`](./DATA_SOURCES.md),
[`dev_commands.md`](./dev_commands.md), [`README.md`](../README.md).

## Top-Level Sections
- **offline**: BC training hyperparameters, dataset paths, and normalization settings.
- **data_fetch / data_parse**: replay ingestion for the dataset tool.
- **imitation_collect**: online self-play collection parameters.
- **online**: PPO defaults and per-mode overrides (scratch / warmstart).
- **imitation_batches / imitation_merge**: batch collectors + dataset merging.
- **evaluation**: BC-vs-bots and PPO evaluation defaults.
- **offline_sweeps / online_sweeps**: grid search parameter lists.

## Critical Contracts
- **Warmstart alignment**: `online.policy_hidden_dim` / `policy_hidden_layers` should match
  `offline.hidden_dim` / `offline.hidden_layers` so BC weights map cleanly.
- **VecNormalize parity**: if `online.use_vecnormalize: true`, training saves
  `<policy>_vecnorm.pkl`; evaluation must load it for fair comparisons.
- **KL regularization**: `kl_coef_*` values are ignored unless a reference policy is set.

## Override Patterns
- `tools/online.py [mode] --override key=value` patches the `online` section.
- `tools/offline_train.py --dataset-path/--epochs/--device/...` overrides common `offline` keys.
- `tools/collect_dataset.py ...` reads `imitation_collect`, `imitation_batches`,
  `imitation_merge`, `data_fetch`, and `data_parse` depending on the subcommand.

## Next steps

- For the **pipelines and artifacts**: [`DATAFLOW.md`](./DATAFLOW.md)
- For the **command cheat sheet**: [`dev_commands.md`](./dev_commands.md)
- For the **big-picture story**: [`codebase_overview.md`](./codebase_overview.md)
