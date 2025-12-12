import contextlib
import json
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset, get_worker_info


def _as_float_list(raw):
    if not isinstance(raw, list):
        raise ValueError
    return [float(value) for value in raw]


def _as_int_list(raw):
    if not isinstance(raw, list):
        raise ValueError
    return [int(value) for value in raw]


def _as_mask(raw):
    if not isinstance(raw, list):
        raise ValueError
    parsed: list[list[int]] = []
    width: int | None = None
    for slot in raw:
        if not isinstance(slot, list):
            raise ValueError
        slot_mask = [1 if int(entry) else 0 for entry in slot]
        if width is None:
            width = len(slot_mask)
        elif len(slot_mask) != width:
            raise ValueError
        parsed.append(slot_mask)
    if not parsed:
        raise ValueError
    return parsed


def _valid_actions(actions, mask):
    if len(actions) != len(mask):
        return False
    for choice, slot_mask in zip(actions, mask, strict=False):
        if choice < 0 or choice >= len(slot_mask):
            return False
        if not slot_mask[choice]:
            return False
    return True


def _drop_sample(payload, filters):
    if not filters:
        return False
    if filters.get("drop_timeouts") and payload.get("timeout"):
        return True
    if filters.get("drop_errors") and payload.get("error"):
        return True
    stats = payload.get("stats") if isinstance(payload, dict) else {}
    reward = payload.get("reward")
    if reward is None and isinstance(stats, dict):
        reward = stats.get("reward")
    if reward is not None:
        min_reward = filters.get("min_reward")
        if min_reward is not None and reward < min_reward:
            return True
    turn = None
    if isinstance(stats, dict):
        turn = stats.get("turn")
    if turn is None:
        turn = payload.get("turn")
    if turn is not None:
        min_turns = filters.get("min_turns")
        if min_turns is not None and turn < min_turns:
            return True
    result = payload.get("result") or (stats.get("result") if isinstance(stats, dict) else None)
    bad_results = filters.get("drop_results") or []
    if result in bad_results:
        return True
    return False


def _sample_weight(payload, cfg):
    if not cfg:
        return 1.0
    weight = 1.0
    stats = payload.get("stats") if isinstance(payload, dict) else {}
    result = payload.get("result") or (stats.get("result") if isinstance(stats, dict) else None)
    if result:
        weight *= float(cfg.get("by_result", {}).get(result, 1.0))
    opponent = payload.get("opponent")
    if opponent:
        weight *= float(cfg.get("by_opponent", {}).get(opponent, 1.0))
    reward = payload.get("reward")
    if reward is None and isinstance(stats, dict):
        reward = stats.get("reward")
    reward_scale = cfg.get("reward_scale")
    if reward is not None and reward_scale:
        weight += float(reward_scale) * float(reward)
    clamp_cfg = cfg.get("clamp") or {}
    w_min = clamp_cfg.get("min")
    w_max = clamp_cfg.get("max")
    if w_min is not None:
        weight = max(float(w_min), weight)
    if w_max is not None:
        weight = min(float(w_max), weight)
    return float(weight)


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


def load_samples(dataset_path, max_samples=None, seed=0, shuffle=True):
    """Legacy helper retained for compatibility with older tooling."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"missing dataset: {path}")
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                sample = _parse_payload(payload)
            except Exception:
                continue
            samples.append(sample)
            if max_samples and len(samples) >= max_samples:
                break
    if not samples:
        raise ValueError(f"no usable records in {path}")
    if shuffle:
        random.Random(seed).shuffle(samples)
    return samples


def split_train_val(samples, val_fraction):
    items = list(samples)
    if not items or val_fraction <= 0:
        return items, []
    if len(items) == 1:
        return items, []
    val_count = max(1, int(len(items) * val_fraction))
    val_count = min(val_count, len(items) - 1)
    val_split = items[:val_count]
    train_split = items[val_count:]
    return train_split, val_split


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


class IndexedJsonlDataset(Dataset):
    """Memory-efficient dataset that reads JSONL records on-demand using byte offsets."""

    def __init__(
        self,
        dataset_path: str | Path = None,
        offsets=None,
        mean=None,
        std=None,
        weight_cfg=None,
        path: str | Path = None,
    ):
        dataset_path = dataset_path or path
        self.path = Path(dataset_path)
        self.offsets = list(offsets)

        mean_arr = np.asarray(mean, dtype=np.float32)
        std_arr = np.asarray(std, dtype=np.float32)
        std_arr[std_arr <= 1e-6] = 1.0

        self._mean = mean_arr
        self._scales = std_arr
        self._handle: Any | None = None
        self._lock = threading.Lock()
        self._worker_handles: dict[int, Any] = {}
        self._weight_cfg = weight_cfg or {}

    def __len__(self) -> int:
        return len(self.offsets)

    def close(self) -> None:
        if self._handle is not None:
            with contextlib.suppress(Exception):
                self._handle.close()
            self._handle = None
        for handle in self._worker_handles.values():
            with contextlib.suppress(Exception):
                handle.close()
        self._worker_handles.clear()

    def _ensure_handle(self):
        if self._handle is None:
            self._handle = self.path.open("r", encoding="utf-8")
        return self._handle

    def _read_line(self, offset: int) -> str:
        worker = get_worker_info()
        if worker is not None:
            handle = self._worker_handles.get(worker.id)
            if handle is None:
                handle = self.path.open("r", encoding="utf-8")
                self._worker_handles[worker.id] = handle
            handle.seek(offset)
            return handle.readline()

        with self._lock:
            handle = self._ensure_handle()
            handle.seek(offset)
            return handle.readline()

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        offset = self.offsets[index]
        line = self._read_line(offset)
        if not line:
            raise ValueError(f"empty line at offset {offset}")

        payload = json.loads(line)
        observation, actions, mask = _parse_payload(payload)

        obs_array = np.asarray(observation, dtype=np.float32)
        obs_array = (obs_array - self._mean) / self._scales
        obs_tensor = torch.from_numpy(obs_array)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        weight = torch.tensor(_sample_weight(payload, self._weight_cfg), dtype=torch.float32)
        return obs_tensor, actions_tensor, mask_tensor, weight

    def __del__(self) -> None:
        self.close()
