import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.offline.dataset import scan_samples


def _fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


def test_scan_samples_matches_golden():
    dataset_path = _fixtures_dir() / "dataset_small.jsonl"
    result = scan_samples(dataset_path, shuffle=False)

    golden = json.loads((_fixtures_dir() / "golden_scan.json").read_text(encoding="utf-8"))

    np.testing.assert_allclose(result.mean, golden["mean"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(result.std, golden["std"], rtol=0, atol=1e-6)
    assert result.binary_flags == golden["binary_flags"]
    assert result.count == golden["count"]
    assert result.obs_dim == golden["obs_dim"]
    assert result.action_dim == golden["action_dim"]
