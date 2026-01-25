# Dataflow

This is the end-to-end flow of data through the project. Use it as a quick oral-defense script.

## 1) Offline data collection
```
Showdown battles
  → src/offline/collect/* (Recorder + heuristics/policy teacher)
  → data/processed/*.jsonl (observation, action, mask)
  → tools/collect_dataset.py collect|batch|merge|purge
```

## 2) Offline training (behavior cloning)
```
JSONL dataset
  → src/offline/dataset/* (parse + filter + scan)
  → src/offline/model.py (BC policy)
  → src/offline/train/* (train loop)
  → outputs/models/bc_policy*.pt + outputs/models/bc_stats*.json
```

## 3) Online training (maskable PPO)
```
make_maskable_env
  → observation encoding (src/core/observation)
  → action masks (src/core/action_mask + env_mask_* mixins)
  → sanitize → repair → fallback
  → KLRegularizedMaskablePPO (optional KL schedule)
  → outputs/models/maskable_ppo_*.zip + *_vecnorm.pkl
```

## 4) Evaluation
```
policy + optional vecnorm
  → tools/eval_models.py suite|compare|ppo|bc
  → src/online/eval/* (env + episodes + summaries)
  → outputs/eval/* summaries + jsonl
```

## Where parity is enforced
- **Observation order**: golden observation fixtures.
- **Mask order**: golden mask fixtures and env smoke tests.
- **VecNormalize**: train saves `<policy>_vecnorm.pkl`, eval loads the same file.
