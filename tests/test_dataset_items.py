import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tests.helpers_dataset import create_tiny_jsonl_dataset

from src.offline.dataset import IndexedJsonlDataset, scan_samples


def test_dataset_length(tmp_path: Path):
    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path, shuffle=False)
    dataset = IndexedJsonlDataset(scan.path, scan.offsets, scan.mean, scan.std)
    assert len(dataset) == len(samples)


def test_dataset_single_item_shapes(tmp_path: Path):
    _, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path)
    dataset = IndexedJsonlDataset(scan.path, scan.offsets, scan.mean, scan.std)

    obs, action, mask, weight = dataset[0]
    assert obs.shape == (4,)
    assert action.shape == (2,)
    assert mask.shape == (2, 4)
    assert weight is not None
    assert float(weight) == 1.0


def test_dataset_normalization(tmp_path: Path):
    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path, shuffle=False)
    dataset = IndexedJsonlDataset(scan.path, scan.offsets, scan.mean, scan.std)

    obs, _, _, _ = dataset[0]
    expected = (np.array(samples[0]["observation"]) - scan.mean) / scan.std
    assert np.allclose(obs, expected, rtol=1e-5, atol=1e-7)
