import contextlib
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tests.helpers_dataset import create_tiny_jsonl_dataset

from src.offline.dataset import IndexedJsonlDataset, scan_samples


def test_dataset_single_worker(tmp_path: Path):
    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path)

    dataset = IndexedJsonlDataset(
        path=scan.path, offsets=scan.offsets, mean=scan.mean, std=scan.std
    )

    loader = DataLoader(dataset, batch_size=2, num_workers=0, shuffle=False)

    batch_count = 0
    for obs_batch, actions_batch, mask_batch, weight_batch in loader:
        batch_count += 1
        assert obs_batch.shape[1] == 4
        assert actions_batch.shape[1] == 2
        assert mask_batch.shape[1:] == (2, 4)
        assert weight_batch.shape[0] == obs_batch.shape[0]

    assert batch_count == 3


def test_dataset_multiple_workers(tmp_path: Path):
    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=10)
    scan = scan_samples(dataset_path)

    dataset = IndexedJsonlDataset(
        path=scan.path, offsets=scan.offsets, mean=scan.mean, std=scan.std
    )

    with contextlib.suppress(Exception):
        torch.multiprocessing.set_sharing_strategy("file_system")
    loader = DataLoader(dataset, batch_size=3, num_workers=2, shuffle=False)

    batches_loaded = 0
    try:
        for obs_batch, actions_batch, mask_batch, weight_batch in loader:
            batches_loaded += 1
            assert obs_batch.shape[0] <= 3
            assert obs_batch.shape[1] == 4
            assert actions_batch.shape == (obs_batch.shape[0], 2)
            assert mask_batch.shape == (obs_batch.shape[0], 2, 4)
            assert weight_batch.shape[0] == obs_batch.shape[0]
            if batches_loaded >= 1:
                break
    except RuntimeError as exc:
        if "torch_shm_manager" in str(exc):
            import pytest

            pytest.skip("torch_shm_manager not permitted in this environment")
        raise

    assert batches_loaded > 0


def test_dataset_all_samples_accessible(tmp_path: Path):
    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path)

    dataset = IndexedJsonlDataset(
        path=scan.path, offsets=scan.offsets, mean=scan.mean, std=scan.std
    )

    for i in range(len(dataset)):
        obs, actions, mask, weight = dataset[i]
        assert obs.shape == (4,)
        assert actions.shape == (2,)
        assert mask.shape == (2, 4)
        assert torch.is_tensor(obs)
        assert torch.is_tensor(actions)
        assert torch.is_tensor(mask)
        assert torch.is_tensor(weight)
