import json
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List, Tuple, Dict, Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.offline.dataset import scan_samples, IndexedJsonlDataset

def create_tiny_jsonl_dataset(
    tmp_path: Path,
    num_samples: int = 5
) -> Tuple[List[Dict[str, Any]], Path]:
    """
    Helper that creates a tiny JSONL dataset for testing.
    """

    samples = []

    # - Feature 0: always binary
    # - Feature 1: non-binary continuous values
    # - Feature 2: another non-binary value
    # - Feature 3: always binary

    for i in range(num_samples):
        observation = [
            float(i % 2),           # Binary: alternates 0, 1, 0, 1, 0
            float(i * 0.5 + 0.1),   # Non-binary: 0.1, 0.6, 1.1, 1.6, 2.1
            float(i + 10.0),        # Non-binary: 10.0, 11.0, 12.0, 13.0, 14.0
            1.0 if i > 2 else 0.0   # Binary: 0, 0, 0, 1, 1
        ]

        action = [0, 1]

        mask = [
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]

        sample_dict = {
            "observation": observation,
            "action": action,
            "mask": mask
        }

        samples.append(sample_dict)

    # Write to JSONL file
    dataset_path = tmp_path / "test_dataset.jsonl"

    with dataset_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    return samples, dataset_path

def test_scan_basic_statistics(tmp_path: Path):
    """
    Test that scan_samples computes correct mean and std.
    """

    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path)

    # Manually compute expected mean and std
    observations = np.array([s["observation"] for s in samples], dtype=np.float32)
    expected_mean = observations.mean(axis=0)
    expected_std = observations.std(axis=0, ddof=0)

    # Check statistics match within tolerance
    assert np.allclose(scan.mean, expected_mean, rtol=1e-5, atol=1e-7)
    assert np.allclose(scan.std, expected_std, rtol=1e-5, atol=1e-7)

def test_scan_metadata(tmp_path: Path):
    """
    Test that scan_samples returns correct metadata.
    """

    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path)

    # Check count
    assert scan.count == 5

    # Check offsets length
    assert len(scan.offsets) == 5

    # Check observation dimension
    assert scan.obs_dim == 4

    # Check path is stored correctly
    assert scan.path == dataset_path

def test_scan_binary_flags(tmp_path: Path):
    """
    Test that scan_samples correctly identifies binary features.
    """

    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path)

    # Expected binary flags: [True, False, False, True]
    # Features 0 and 3 are binary
    # Features 1 and 2 are non-binary
    expected_flags = [True, False, False, True]

    assert len(scan.binary_flags) == 4
    assert scan.binary_flags == expected_flags

def test_scan_deteministic_shuffling_same_seed(tmp_path: Path):
    """
    Test that same seed produces identical shuffle order.
    """

    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)

    # Scan with same seed twice
    scan1 = scan_samples(dataset_path, seed=7, shuffle=True)
    scan2 = scan_samples(dataset_path, seed=7, shuffle=True)

    assert scan1.offsets == scan2.offsets

def test_scan_different_seeds(tmp_path: Path):
    """
    Test that different seeds produce different shuffle orders.
    """

    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)

    # Scan with different seeds
    scan1 = scan_samples(dataset_path, seed=56, shuffle=True)
    scan2 = scan_samples(dataset_path, seed=785, shuffle=True)

    assert scan1.offsets != scan2.offsets

def test_scan_no_shuffle(tmp_path: Path):
    """
    Test that shuffle=False preserves file order.
    """

    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path, shuffle=False)

    assert len(scan.offsets) == 5

    # Offsets should be increasing for sequential file reads
    for i in range(len(scan.offsets) - 1):
        assert scan.offsets[i] < scan.offsets[i + 1]

def test_dataset_length(tmp_path: Path):
    """
    Test that dataset reports correct length.
    """

    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path)

    dataset = IndexedJsonlDataset(
        path=scan.path,
        offsets=scan.offsets,
        mean=scan.mean,
        std=scan.std
    )

    assert len(dataset) == 5

def test_dataset_single_item_shapes(tmp_path: Path):
    """
    Test that individual samples have expected shapes.
    """

    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path)

    dataset = IndexedJsonlDataset(
        path=scan.path,
        offsets=scan.offsets,
        mean=scan.mean,
        std=scan.std
    )

    # Get first sample
    obs, actions, mask = dataset[0]

    # Check observation shape
    assert obs.shape == (4,)

    # Check actions shape
    assert actions.shape == (2,)

    # Check mask shape
    assert mask.shape == (2, 4)

def test_dataset_normalization(tmp_path: Path):
    """
    Test that observations are normalized using mean/std.
    """

    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path, shuffle=False)

    dataset = IndexedJsonlDataset(
        path=scan.path,
        offsets=scan.offsets,
        mean=scan.mean,
        std=scan.std
    )

    # Get first sample
    obs, _, _ = dataset[0]

    # Manually compute normalized observation for first sample
    raw_obs = np.array(samples[0]["observation"], dtype=np.float32)
    expected_normalized = (raw_obs - scan.mean) / np.where(scan.std > 1e-6, scan.std, 1.0)

    assert np.allclose(obs.numpy(), expected_normalized, rtol=1e-5, atol=1e-7)

def test_dataset_single_worker(tmp_path: Path):
    """
    Test dataset with single DataLoader worker.
    """

    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path)

    dataset = IndexedJsonlDataset(
        path=scan.path,
        offsets=scan.offsets,
        mean=scan.mean,
        std=scan.std
    )

    # Create DataLoader with single worker
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        shuffle=False
    )

    # Iterate through batches
    batch_count = 0

    for obs_batch, actions_batch, mask_batch in loader:
        batch_count += 1

        # Check batch shapes
        assert obs_batch.shape[1] == 4
        assert actions_batch.shape[1] == 2
        assert mask_batch.shape[1:] == (2, 4)

    # Should have 3 batches (5 samples / batch_size 2 = 3 batches)
    assert batch_count == 3

def test_dataset_multiple_workers(tmp_path: Path):
    """
    Test dataset with multiple DataLoader workers
    """

    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=10)
    scan = scan_samples(dataset_path)

    dataset = IndexedJsonlDataset(
        path=scan.path,
        offsets=scan.offsets,
        mean=scan.mean,
        std=scan.std
    )

    # Create DataLoader with multiple workers
    loader = DataLoader(
        dataset,
        batch_size=3,
        num_workers=2,
        shuffle=False
    )

    # Iterate through at least one batch without exceptions
    batches_loaded = 0

    for obs_batch, actions_batch, mask_batch in loader:
        batches_loaded += 1

        # Verify batch shapes are correct
        assert obs_batch.shape[0] <= 3
        assert obs_batch.shape[1] == 4
        assert actions_batch.shape == (obs_batch.shape[0], 2)
        assert mask_batch.shape == (obs_batch.shape[0], 2, 4)

        # Break after first batch
        if batches_loaded >= 1:
            break

    assert batches_loaded > 0

def test_dataset_all_samples_accessible(tmp_path: Path):
    """
    Test that all samples can be accessed individually.
    """

    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=5)
    scan = scan_samples(dataset_path)

    dataset = IndexedJsonlDataset(
        path=scan.path,
        offsets=scan.offsets,
        mean=scan.mean,
        std=scan.std
    )

    # Try accessing all samples
    for i in range(len(dataset)):
        obs, actions, mask = dataset[i]

        # Basic sanity checks
        assert obs.shape == (4,)
        assert actions.shape == (2,)
        assert mask.shape == (2, 4)
        assert torch.is_tensor(obs)
        assert torch.is_tensor(actions)
        assert torch.is_tensor(mask)

def test_empty_file(tmp_path: Path):
    """
    Test handling of empty JSONL file.
    """

    dataset_path = tmp_path / "empty.jsonl"

    dataset_path.touch()

    try:
        scan_samples(dataset_path)
        assert False, "Should have raised an exception for empty file"
    except (ValueError, FileNotFoundError):
        pass

def test_invalid_json(tmp_path: Path):
    """
    Test handling of invalid JSON lines.
    """

    dataset_path = tmp_path / "invalid.jsonl"

    with dataset_path.open("w") as f:
        f.write('{"observation": [1.0, 2.0, 3.0, 4.0], "action": [0, 1], "mask": [[1,1,1,1],[1,1,1,1]]}\n')
        f.write('this is not valid json\n')
        f.write('{"observation": [3.0, 4.0, 5.0, 6.0], "action": [1, 0], "mask": [[1,1,1,1],[1,1,1,1]]}\n')

    # Should skip invalid lines and process valid ones
    scan = scan_samples(dataset_path)

    assert scan.count == 2

def test_max_samples_limit(tmp_path: Path):
    """
    Test max_samples parameter limits dataset size.
    """
    
    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=10)
    scan = scan_samples(dataset_path, max_samples=5)

    assert scan.count == 5
    assert len(scan.offsets) == 5