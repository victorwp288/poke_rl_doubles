from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PolicySpec:
    label: str
    path: Path


@dataclass(frozen=True)
class OpponentSpec:
    label: str
    kind: str
    policy_path: Path | None = None


__all__ = ["OpponentSpec", "PolicySpec"]
