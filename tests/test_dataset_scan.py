import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tests.helpers_dataset import create_tiny_jsonl_dataset

from src.offline.dataset import scan_samples


def test_scan_basic_statistics(tmp_path: Path):
    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path)

    observations = np.array([s["observation"] for s in samples], dtype=np.float32)
    expected_mean = observations.mean(axis=0)
    expected_std = observations.std(axis=0, ddof=0)

    assert np.allclose(scan.mean, expected_mean, rtol=1e-5, atol=1e-7)
    assert np.allclose(scan.std, expected_std, rtol=1e-5, atol=1e-7)


def test_scan_metadata(tmp_path: Path):
    _, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path)

    assert scan.count == 5
    assert len(scan.offsets) == 5
    assert scan.obs_dim == 4
    assert scan.path == dataset_path


def test_scan_binary_flags(tmp_path: Path):
    _, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path)

    expected_flags = [True, False, False, True]
    assert len(scan.binary_flags) == 4
    assert scan.binary_flags == expected_flags


def test_scan_deteministic_shuffling_same_seed(tmp_path: Path):
    _, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)

    scan1 = scan_samples(dataset_path, seed=7, shuffle=True)
    scan2 = scan_samples(dataset_path, seed=7, shuffle=True)

    assert scan1.offsets == scan2.offsets


def test_scan_different_seeds(tmp_path: Path):
    _, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)

    scan1 = scan_samples(dataset_path, seed=7, shuffle=True)
    scan2 = scan_samples(dataset_path, seed=99, shuffle=True)

    assert scan1.offsets != scan2.offsets


def test_scan_no_shuffle(tmp_path: Path):
    _, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path, shuffle=False)
    assert scan.offsets[0] == 0
    assert scan.offsets == sorted(scan.offsets)
