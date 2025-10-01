# Config for offline training
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class OfflineConfig:
    dataset_path: Path = Path("data/processed/imitation.jsonl")
    policy_path: Path = Path("outputs/models/bc_policy.pt")
    hints_path: Path | None = Path("data/processed/human_hints.jsonl")
    tensorboard_dir: Path | None = Path("outputs/tensorboard")
    batch_size: int = 2048
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 50
    val_fraction: float = 0.1
    max_samples: int | None = None
    seed: int = 0
    device: str | None = None
    shuffle: bool = True
    log_every: int = 50

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
