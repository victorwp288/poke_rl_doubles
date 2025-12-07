import json
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Union, Any
import numpy as np
import numpy.typing as npt

def _as_float_list(raw):
    if not isinstance(raw, list):
        raise ValueError
    values = []
    for value in raw:
        values.append(float(value))
    return values


def _as_int_list(raw):
    if not isinstance(raw, list):
        raise ValueError
    values = []
    for value in raw:
        values.append(int(value))
    return values


def _as_mask(raw):
    if not isinstance(raw, list):
        raise ValueError
    parsed = []
    width = None
    for slot in raw:
        if not isinstance(slot, list):
            raise ValueError
        slot_mask = []
        for entry in slot:
            slot_mask.append(1 if int(entry) else 0)
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
    if not samples:
        raise ValueError(f"no usable records in {path}")
    if shuffle:
        random.Random(seed).shuffle(samples)
    if max_samples and max_samples > 0:
        samples = samples[:max_samples]
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

class ScanResult:
    def __init__(
        self,
        path: Path,
        offsets: List[int],
        mean: npt.NDArray[np.float32],
        std: npt.NDArray[np.float32],
        count: int,
        obs_dim: int,
        binary_flags: List[bool]
    ):
        self.path = path
        self.offsets = offsets
        self.mean = mean
        self.std = std
        self.count = count
        self.obs_dim = obs_dim
        self.binary_flags = binary_flags

def scan_samples(
    dataset_path: Union[str, Path],
    max_samples: Optional[int] = None,
    seed: int = 0,
    shuffle: bool = True
) -> ScanResult:
    """
    Scan a JSONL dataset to collect statistics and byte offsets without loading all data into memory.
    """

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    
    offsets = []
    observations = []
    obs_dim = None

    # First pass: collect offsets and observations
    with path.open("rb") as handle:
        offset = 0

        for raw_line_bytes in handle:
            line_start = offset
            offset += len(raw_line_bytes)
            line = raw_line_bytes.decode("utf-8").strip()

            if not line:
                continue

            try:
                payload = json.loads(line)
                obs, actions, mask = _parse_payload(payload)

                if obs_dim is None:
                    obs_dim = len(obs)
                elif len(obs) != obs_dim:
                    continue

                offsets.append(line_start)
                observations.append(obs)

                if max_samples and len(observations) >= max_samples:
                    break
            except Exception:
                continue
    
    if not observations:
        raise ValueError(f"No usable records in {path}")
    
    count = len(observations)
    obs_array = torch.tensor(observations, dtype=torch.float32)
    mean = obs_array.mean(dim=0)
    std = obs_array.std(dim=0, unbiased=False)
    binary_flags = []

    assert obs_dim is not None

    for col_idx in range(obs_dim):
        column = obs_array[:, col_idx]
        unique_vals = torch.unique(column)
        is_binary = all(val in [0.0, 1.0] for val in unique_vals.tolist())

        binary_flags.append(is_binary)

    if shuffle:
        indices = list(range(len(offsets)))
        random.Random(seed).shuffle(indices)
        offsets = [offsets[i] for i in indices]

    return ScanResult(
        path=path,
        offsets=offsets,
        mean=mean.numpy(),
        std=std.numpy(),
        count=count,
        obs_dim=obs_dim,
        binary_flags=binary_flags
    )

class IndexedJsonlDataset(Dataset):
    """
    Memory efficient dataset that reads JSONL files on-demand using byte offsets.
    """

    def __init__(
        self,
        path: Union[str, Path],
        offsets: List[int],
        mean: Union[npt.NDArray[np.float32], torch.Tensor],
        std: Union[npt.NDArray[np.float32], torch.Tensor],
        binary_flags: Optional[List[bool]] = None
    ):
        self.path = Path(path)
        self.offsets = offsets
        self.mean = torch.tensor(mean, dtype=torch.float32) if not isinstance(mean, torch.Tensor) else mean
        self.std = torch.tensor(std, dtype=torch.float32) if not isinstance(std, torch.Tensor) else std
        self.binary_flags = binary_flags
        self._file_handle = None

    def _ensure_file_open(self):
        if self._file_handle is None:
            self._file_handle = self.path.open("rb")

    def __len__(self) -> int:
        return len(self.offsets)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_file_open()

        # Seek byte offset
        assert self._file_handle is not None
        self._file_handle.seek(self.offsets[idx])

        line_bytes: bytes = self._file_handle.readline()
        line = line_bytes.decode("utf-8").strip()

        if not line:
            raise ValueError(f"Empty line at offset {self.offsets[idx]}")

        # Parse JSON
        payload = json.loads(line)
        obs, actions, mask = _parse_payload(payload)

        # Convert to tensors
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)

        # Normalize observations
        std_safe = torch.where(self.std > 1e-6, self.std, torch.ones_like(self.std))
        obs_normalized = (obs_tensor - self.mean) / std_safe

        return obs_normalized, actions_tensor, mask_tensor
    
    def __del__(self):
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception:
                pass