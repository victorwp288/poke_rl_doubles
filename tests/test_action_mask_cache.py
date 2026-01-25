#!/usr/bin/env python3

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from poke_env.player.battle_order import SingleBattleOrder

from src.core.action_mask import slot_action_mask


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
        patch("src.core.action_mask._legal_orders", side_effect=fake_legal_orders),
        patch("src.core.action_mask.DoublesEnv.order_to_action", side_effect=fake_order_to_action),
        patch(
            "src.core.action_mask.DoublesEnv._order_to_action_individual",
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
        patch("src.core.action_mask._mask_state_key", return_value="token"),
        patch("src.core.action_mask._build_slot_action_mask", side_effect=[[1, 0], [0, 1]]),
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

    with patch("src.core.action_mask._mask_state_key", return_value="token"):
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

    with patch("src.core.action_mask._legal_orders", return_value=[]):
        mask_a = slot_action_mask(battle, 0, 12)
        mask_b = slot_action_mask(battle, 1, 12)

    assert mask_a[0] == 1
    assert sum(mask_a) == 1
    assert mask_b[0] == 1
    assert sum(mask_b) == 1
