import contextlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from .filters import _sample_weight
from .parsing import _parse_payload


class IndexedJsonlDataset(Dataset):
    """Memory-efficient dataset that reads JSONL records on-demand using byte offsets."""

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        offsets=None,
        mean=None,
        std=None,
        weight_cfg=None,
        path: str | Path | None = None,
    ):
        dataset_path = dataset_path or path
        if dataset_path is None:
            raise ValueError("dataset_path is required")
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

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        state["_worker_handles"] = {}
        state["_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._handle = None
        self._worker_handles = {}
        self._lock = threading.Lock()


__all__ = ["IndexedJsonlDataset"]
