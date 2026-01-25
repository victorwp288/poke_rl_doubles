import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from poke_env.player.battle_order import SingleBattleOrder
from tests.helpers_mask_battle import build_mask_battle

from src.core.action_mask import combine_slot_masks, slot_action_mask


def _fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


def test_slot_action_mask_matches_golden(monkeypatch):
    battle = build_mask_battle()
    act_size = 12

    order_a = SingleBattleOrder("/choose move tackle")
    order_b = SingleBattleOrder("/choose switch allya")
    order_c = SingleBattleOrder("/choose move protect")

    def fake_legal_orders(_, slot):
        if slot == 0:
            return [order_a, order_b]
        return [order_c]

    def fake_order_to_action(order, battle_obj, fake=False, strict=False):
        first = getattr(order.first_order, "order", None)
        second = getattr(order.second_order, "order", None)
        mapping = {
            "/choose move tackle": 2,
            "/choose switch allya": 7,
            "/choose move protect": 4,
        }
        return np.asarray([mapping.get(first, 0), mapping.get(second, 0)], dtype=int)

    def fake_order_to_action_individual(order, battle_obj, strict, slot):
        mapping = {
            "/choose move tackle": 2,
            "/choose switch allya": 7,
            "/choose move protect": 4,
        }
        return mapping.get(getattr(order, "order", None), 0)

    monkeypatch.setattr("src.core.action_mask._legal_orders", fake_legal_orders)
    monkeypatch.setattr("src.core.action_mask.DoublesEnv.order_to_action", fake_order_to_action)
    monkeypatch.setattr(
        "src.core.action_mask.DoublesEnv._order_to_action_individual",
        fake_order_to_action_individual,
    )

    mask_a = slot_action_mask(battle, 0, act_size)
    mask_b = slot_action_mask(battle, 1, act_size)
    combined = combine_slot_masks(mask_a, mask_b).astype(np.uint8)

    fixture_path = _fixtures_dir() / "golden_slot_mask.npy"
    expected = np.load(fixture_path).astype(np.uint8)
    assert combined.shape == expected.shape
    np.testing.assert_array_equal(combined, expected)
