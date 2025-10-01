#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.offline import OfflineConfig, train_offline  # noqa: E402


@dataclass
class Trial:
    learning_rate: float
    batch_size: int


BASE_SWEEP = [
    Trial(learning_rate=5e-4, batch_size=1024),
    Trial(learning_rate=5e-4, batch_size=2048),
    Trial(learning_rate=1e-3, batch_size=1024),
    Trial(learning_rate=1e-3, batch_size=2048),
]


def run_trial(base_config: OfflineConfig, trial: Trial) -> None:
    config = OfflineConfig(
        dataset_path=base_config.dataset_path,
        policy_path=base_config.policy_path,
        hints_path=base_config.hints_path,
        batch_size=trial.batch_size,
        learning_rate=trial.learning_rate,
        weight_decay=base_config.weight_decay,
        epochs=base_config.epochs,
        val_fraction=base_config.val_fraction,
        max_samples=base_config.max_samples,
        seed=base_config.seed,
        device=base_config.device,
        shuffle=base_config.shuffle,
        log_every=base_config.log_every,
    )
    print(f"running lr={trial.learning_rate} batch={trial.batch_size}")
    train_offline(config)


def main() -> None:
    base = OfflineConfig()
    for trial in BASE_SWEEP:
        run_trial(base, trial)


if __name__ == "__main__":
    main()
