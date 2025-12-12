#!/usr/bin/env python3
# Tests for the action repair helpers in MaskableDoublesEnv

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from gymnasium import spaces

sys.path.append(str(Path(__file__).resolve().parents[1]))

from poke_env.player.battle_order import SingleBattleOrder

from src.core.features import slot_action_mask
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

    env._mask_allows = fake_mask_allows  # type: ignore[attr-defined]
    env._within_space = fake_within_space  # type: ignore[attr-defined]

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

    env._mask_allows = lambda vector, masks: True  # type: ignore[attr-defined]

    with (
        patch.object(MaskableDoublesEnv, "_forced_switch_vectors", return_value=[]),
        patch("src.online.env.DoublesEnv.action_to_order", side_effect=fake_action_to_order),
        patch.object(MaskableDoublesEnv, "_maps_to_default", return_value=True),
    ):
        repaired = env._repair_action(np.array([5, 5]), slot_masks)

    assert repaired is not None
    assert np.asarray(repaired, dtype=int).reshape(-1).tolist() == [2, 2]


def test_slot_action_mask_forced_switch_indices():
    battle = type("Battle", (), {"force_switch": [True, False]})()
    switch_a = SingleBattleOrder("/choose switch rillaboom")
    switch_b = SingleBattleOrder("/choose switch ironhands")
    default_second = SingleBattleOrder("/choose default")

    def fake_legal_orders(_, slot):
        if slot == 0:
            return [switch_a, switch_b]
        return [default_second]

    def fake_order_to_action(order, battle_obj, fake=False, strict=False):
        first = getattr(order.first_order, "order", None)
        mapping = {
            "/choose switch rillaboom": 5,
            "/choose switch ironhands": 9,
        }
        value = mapping.get(first, 0)
        return np.asarray([value, 0], dtype=int)

    def fake_order_to_action_individual(order, battle_obj, strict, slot):
        mapping = {
            "/choose switch rillaboom": 5,
            "/choose switch ironhands": 9,
        }
        token = getattr(order, "order", None)
        return mapping.get(token, 0)

    with (
        patch("src.core.features._legal_orders", side_effect=fake_legal_orders),
        patch("src.core.features.DoublesEnv.order_to_action", side_effect=fake_order_to_action),
        patch(
            "src.core.features.DoublesEnv._order_to_action_individual",
            side_effect=fake_order_to_action_individual,
        ),
    ):
        mask = slot_action_mask(battle, 0, 12)

    assert mask[0] == 0
    assert mask[5] == 1
    assert mask[9] == 1
    assert sum(mask) == 2


def test_slot_action_mask_returns_copy_on_cache_miss():
    battle = SimpleNamespace()

    with (
        patch("src.core.features._mask_state_key", return_value="token"),
        patch("src.core.features._build_slot_action_mask", side_effect=[[1, 0], [0, 1]]),
    ):
        mask = slot_action_mask(battle, 0, 2)

    cached = battle._slot_action_cache["masks"][0]
    assert mask == [1, 0]
    assert mask is not cached

    mask[0] = 0
    assert cached[0] == 1


def test_slot_action_mask_returns_copy_on_cache_hit():
    battle = SimpleNamespace(
        _slot_action_cache={"token": "token", "act_size": 2, "masks": [[1, 0], [0, 1]]}
    )

    with patch("src.core.features._mask_state_key", return_value="token"):
        mask = slot_action_mask(battle, 0, 2)

    cached = battle._slot_action_cache["masks"][0]
    assert mask == [1, 0]
    assert mask is not cached

    mask[0] = 0
    assert cached[0] == 1


def test_slot_action_mask_forced_single_switch_defaults():
    class DummyMon:
        def __init__(self, name):
            self.base_species = name

    battle = type(
        "Battle",
        (),
        {
            "force_switch": [True, True],
            "available_switches": [[DummyMon("A")], [DummyMon("B")]],
        },
    )()

    with patch("src.core.features._legal_orders", return_value=[]):
        mask_a = slot_action_mask(battle, 0, 12)
        mask_b = slot_action_mask(battle, 1, 12)

    assert mask_a[0] == 1
    assert sum(mask_a) == 1
    assert mask_b[0] == 1
    assert sum(mask_b) == 1
