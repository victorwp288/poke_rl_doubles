from src.core.action_mask import combine_slot_masks, slot_action_mask


def _battle_cache_key(battle):
    return battle.battle_tag, getattr(battle, "player_role", "player")


def _joint_action_mask(battle, act_size):
    mask_a = slot_action_mask(battle, 0, act_size)
    mask_b = slot_action_mask(battle, 1, act_size)
    # Joint mask uses slot0 then slot1 order; keep consistent with policy action layout.
    return combine_slot_masks(mask_a, mask_b)


__all__ = ["_battle_cache_key", "_joint_action_mask"]
