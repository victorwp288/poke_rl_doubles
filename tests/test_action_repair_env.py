#!/usr/bin/env python3

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from gymnasium import spaces

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.online.env import MaskableDoublesEnv


def _make_stub_env():
    env = object.__new__(MaskableDoublesEnv)
    env._act_size = 3
    env.action_space = spaces.MultiDiscrete([3, 3])
    env.env = type("Base", (), {})()
    env.env.battle1 = object()
    return env


def test_maps_to_default_strict_detection():
    env = _make_stub_env()
    strict_calls = []

    def fake_action_to_order(vector, battle, fake=False, strict=False):
        strict_calls.append(strict)
        flat = tuple(np.asarray(vector, dtype=int).reshape(-1).tolist())
        if flat == (0, 0):
            if strict:
                raise ValueError("strict conversion rejected default")
            return type("Dummy", (), {"order": "/choose default"})()
        if strict:
            return "/choose move heatwave"
        return type("Dummy", (), {"order": "/choose move heatwave"})()

    with patch("src.online.env.DoublesEnv.action_to_order", side_effect=fake_action_to_order):
        assert env._maps_to_default(np.array([0, 0]), strict=True) is True
        assert env._maps_to_default(np.array([1, 0]), strict=True) is False

    assert any(strict_calls), "Expected strict flag usage in action_to_order"


def test_safe_action_fallback_skips_default_candidate():
    env = _make_stub_env()
    slot_masks = np.ones((2, 3), dtype=bool)

    def fake_joint_candidates(self, masks):
        return [np.array([0, 0]), np.array([1, 2])]

    def fake_maps_to_default(self, vector, strict=False):
        flat = tuple(np.asarray(vector, dtype=int).reshape(-1).tolist())
        return flat == (0, 0)

    with (
        patch.object(MaskableDoublesEnv, "_joint_candidates", fake_joint_candidates),
        patch.object(MaskableDoublesEnv, "_maps_to_default", fake_maps_to_default),
    ):
        choice = env._safe_action_fallback(slot_masks)

    assert choice is not None
    flattened = np.asarray(choice, dtype=int).reshape(-1)
    assert flattened.tolist() == [1, 2]


def test_safe_action_fallback_returns_none_when_only_default():
    env = _make_stub_env()
    slot_masks = np.ones((2, 3), dtype=bool)

    def fake_joint_candidates(self, masks):
        return [np.array([0, 0])]

    def fake_maps_to_default(self, vector, strict=False):
        return True

    with (
        patch.object(MaskableDoublesEnv, "_joint_candidates", fake_joint_candidates),
        patch.object(MaskableDoublesEnv, "_maps_to_default", fake_maps_to_default),
    ):
        choice = env._safe_action_fallback(slot_masks)

    assert choice is None


def test_repair_action_uses_strict_fallback():
    env = _make_stub_env()
    slot_masks = np.ones((2, 3), dtype=bool)

    fallback_vector = np.array([1, 2])

    def fake_joint_candidates(self, masks):
        return [fallback_vector]

    def fake_mask_allows(vector, masks):
        flat = np.asarray(vector, dtype=int).reshape(-1)
        return bool(np.array_equal(flat, fallback_vector))

    def fake_within_space(vector):
        return True

    def fake_action_to_order(vector, battle, fake=False, strict=True):
        if strict:
            raise ValueError("strict failure")
        return type("Dummy", (), {"order": "/choose move heatwave"})()

    env._mask_allows_action = fake_mask_allows  # type: ignore[attr-defined]
    env._within_action_space = fake_within_space  # type: ignore[attr-defined]

    with (
        patch.object(MaskableDoublesEnv, "_joint_candidates", fake_joint_candidates),
        patch("src.online.env.DoublesEnv.action_to_order", side_effect=fake_action_to_order),
    ):
        repaired = env._repair_action(np.array([5, 5]), slot_masks)

    assert repaired is not None
    assert np.asarray(repaired, dtype=int).reshape(-1).tolist() == fallback_vector.tolist()


def test_repair_action_forced_switch_fallback():
    env = _make_stub_env()
    slot_masks = np.ones((2, 3), dtype=bool)

    fallback_vector = np.array([1, 2])

    def fake_action_to_order(vector, battle, fake=False, strict=True):
        flat = tuple(np.asarray(vector, dtype=int).reshape(-1).tolist())
        if flat == (0, 0):
            return "/choose default"
        if flat == tuple(fallback_vector.tolist()):
            return "/choose switch rillaboom, switch tornadus"
        raise ValueError("unexpected vector")

    with (
        patch.object(
            MaskableDoublesEnv,
            "_forced_switch_vectors",
            return_value=[fallback_vector.copy()],
        ),
        patch("src.online.env.DoublesEnv.action_to_order", side_effect=fake_action_to_order),
    ):
        repaired = env._repair_action(np.array([0, 0]), slot_masks)

    assert repaired is not None
    flattened = np.asarray(repaired, dtype=int).reshape(-1)
    assert flattened.tolist() == fallback_vector.tolist()


def test_repair_action_allows_default_when_no_alternative():
    env = _make_stub_env()
    slot_masks = np.ones((2, 3), dtype=bool)

    def fake_action_to_order(vector, battle, fake=False, strict=True):
        return "/choose default"

    env._mask_allows_action = lambda vector, masks: True  # type: ignore[attr-defined]

    with (
        patch.object(MaskableDoublesEnv, "_forced_switch_vectors", return_value=[]),
        patch("src.online.env.DoublesEnv.action_to_order", side_effect=fake_action_to_order),
        patch.object(MaskableDoublesEnv, "_maps_to_default", return_value=True),
    ):
        repaired = env._repair_action(np.array([5, 5]), slot_masks)

    assert repaired is not None
    assert np.asarray(repaired, dtype=int).reshape(-1).tolist() == [2, 2]
