import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .filters import _drop_sample
from .parsing import _parse_payload


@dataclass
class ScanResult:
    path: Path
    offsets: list[int]
    mean: npt.NDArray[np.float32]
    std: npt.NDArray[np.float32]
    binary_flags: list[bool]
    count: int
    obs_dim: int
    action_dim: int


def scan_samples(
    dataset_path: str | Path,
    max_samples: int | None = None,
    seed: int = 0,
    shuffle: bool = True,
    filters: dict | None = None,
) -> ScanResult:
    """Scan a JSONL dataset in a streaming fashion to compute stats and byte offsets.

    This implementation keeps only running statistics in memory instead of all
    observations, while remaining API-compatible with the original helper.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"missing dataset: {path}")

    offsets: list[int] = []
    count = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    binary_flags: np.ndarray | None = None
    obs_dim: int | None = None
    action_dim: int | None = None

    with path.open("r", encoding="utf-8") as handle:
        while True:
            position = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                if _drop_sample(payload, filters):
                    continue
                observation, actions, mask = _parse_payload(payload)
            except Exception:
                continue

            obs_array = np.asarray(observation, dtype=np.float64)
            if obs_dim is None:
                obs_dim = int(obs_array.size)
                mean = np.zeros(obs_dim, dtype=np.float64)
                m2 = np.zeros(obs_dim, dtype=np.float64)
                binary_flags = np.ones(obs_dim, dtype=bool)
            if obs_array.size != obs_dim:
                continue

            count += 1
            # Welford's online algorithm for mean / variance
            assert mean is not None and m2 is not None
            delta = obs_array - mean
            mean += delta / count
            delta2 = obs_array - mean
            m2 += delta * delta2

            if binary_flags is not None:
                binary_flags &= np.isin(obs_array, (0.0, 1.0))

            offsets.append(position)
            if action_dim is None and mask:
                action_dim = len(mask[0])
            if max_samples and len(offsets) >= max_samples:
                break

    if not offsets or count == 0:
        raise ValueError(f"no usable records in {path}")

    if shuffle:
        random.Random(seed).shuffle(offsets)

    assert mean is not None and m2 is not None and obs_dim is not None
    variance = m2 / count if count > 0 else np.zeros_like(m2)
    std = np.sqrt(np.maximum(variance, 0.0))

    mean32 = mean.astype(np.float32)
    std32 = std.astype(np.float32)
    binary_list = binary_flags.tolist() if binary_flags is not None else []

    return ScanResult(
        path=path,
        offsets=offsets,
        mean=mean32,
        std=std32,
        binary_flags=list(binary_list),
        count=count,
        obs_dim=obs_dim or 0,
        action_dim=action_dim or 0,
    )


__all__ = ["ScanResult", "scan_samples"]
