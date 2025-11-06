#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import section  # noqa: E402
from src.offline import train_offline  # noqa: E402

TRIALS = [
    {
        "learning_rate": 5e-4,
        "batch_size": 1024,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "hidden_dim": 384,
        "hidden_layers": 3,
    },
    {
        "learning_rate": 5e-4,
        "batch_size": 2048,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "hidden_dim": 384,
        "hidden_layers": 4,
    },
    {
        "learning_rate": 1e-3,
        "batch_size": 1024,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "hidden_dim": 512,
        "hidden_layers": 3,
    },
    {
        "learning_rate": 1e-3,
        "batch_size": 2048,
        "weight_decay": 5e-5,
        "dropout": 0.2,
        "hidden_dim": 512,
        "hidden_layers": 4,
    },
]


def _trial_name(trial):
    return (
        f"lr{trial['learning_rate']:g}_bs{trial['batch_size']}_wd{trial['weight_decay']:g}_"
        f"drop{trial['dropout']:.2f}_hidden{trial['hidden_dim']}x{trial['hidden_layers']}"
    )


def _build_settings(base_settings, trial, workdir):
    settings = dict(base_settings)
    settings.update(trial)
    workdir.mkdir(parents=True, exist_ok=True)
    settings["policy_path"] = str(workdir / "bc_policy.pt")
    settings["tensorboard_dir"] = str(workdir / "tensorboard")
    settings["stats_path"] = str(workdir / "stats.json")
    return settings


def run_trial(base_settings, trial, root):
    trial_dir = root / _trial_name(trial)
    settings = _build_settings(base_settings, trial, trial_dir)
    print(
        "running trial -> "
        f"lr={trial['learning_rate']:g} batch={trial['batch_size']} "
        f"wd={trial['weight_decay']:g} dropout={trial['dropout']:.2f} "
        f"hidden={trial['hidden_dim']}x{trial['hidden_layers']}"
    )
    try:
        metrics = train_offline(settings)
        metrics_path = trial_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"stored metrics in {trial_dir}")
    except Exception as e:
        print(f"trial failed: {e}")
        error_path = trial_dir / "error.txt"
        error_path.write_text(str(e), encoding="utf-8")


def main():
    base = section("offline")
    output_root = Path("outputs/offline_sweep")
    output_root.mkdir(parents=True, exist_ok=True)
    for trial in TRIALS:
        run_trial(base, trial, output_root)


if __name__ == "__main__":
    main()
