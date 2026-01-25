import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tests.helpers_battle import build_dummy_battle

from src.core.observation import encode_observation


def _fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


def test_encode_observation_matches_golden():
    battle = build_dummy_battle()
    obs = np.asarray(encode_observation(battle), dtype=np.float32)
    fixture_path = _fixtures_dir() / "golden_observation.npy"
    expected = np.load(fixture_path).astype(np.float32)
    assert obs.shape == expected.shape
    np.testing.assert_array_equal(obs, expected)
