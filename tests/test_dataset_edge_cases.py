import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tests.helpers_dataset import create_tiny_jsonl_dataset

from src.offline.dataset import scan_samples


def test_empty_file(tmp_path: Path):
    dataset_path = tmp_path / "empty.jsonl"
    dataset_path.touch()
    with pytest.raises(ValueError):
        scan_samples(dataset_path)


def test_invalid_json(tmp_path: Path):
    dataset_path = tmp_path / "invalid.jsonl"
    dataset_path.write_text("{not a json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        scan_samples(dataset_path)


def test_max_samples_limit(tmp_path: Path):
    samples, dataset_path = create_tiny_jsonl_dataset(tmp_path, num_samples=10)
    scan = scan_samples(dataset_path, max_samples=3)
    assert scan.count == 3
