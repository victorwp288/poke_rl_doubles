"""
Action mask construction for doubles.

Summary:
- Computes which discrete action IDs are legal *per slot* given the current battle state.
- In doubles, legality depends on joint choices; we join per-slot legal orders and map them back to
  per-slot action indices.
- The training/eval contract is that joint masks are concatenated as `[slot0 | slot1]`.

Related parity tests: `tests/test_feature_golden_mask.py`.
"""

from collections.abc import Sequence

import numpy as np
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.battle_order import DefaultBattleOrder, DoubleBattleOrder, SingleBattleOrder

MaskVector = list[int]
MaskLike = Sequence[int]


def _legal_orders(battle, slot):
    orders = []
    if slot < 0 or slot >= 2:
        return orders
    active = battle.active_pokemon
    mon = active[slot] if slot < len(active) else None
    switches = battle.available_switches[slot] if slot < len(battle.available_switches) else []
    force_switch = getattr(battle, "force_switch", [False, False])
    if force_switch[slot]:
        for switch in switches:
            orders.append(SingleBattleOrder(switch))
        return orders
    if mon is None:
        return orders

    moves = battle.available_moves[slot] if slot < len(battle.available_moves) else []
    available_z_moves = set(getattr(mon, "available_z_moves", []) or [])
    for move in moves:
        try:
            targets = battle.get_possible_showdown_targets(move, mon)
        except Exception:
            targets = [0]
        if not targets:
            targets = [0]
        for target in targets:
            orders.append(SingleBattleOrder(move, move_target=target))
            if battle.can_mega_evolve[slot]:
                orders.append(SingleBattleOrder(move, move_target=target, mega=True))
            if battle.can_z_move[slot] and move in available_z_moves:
                orders.append(SingleBattleOrder(move, move_target=target, z_move=True))
            if battle.can_dynamax[slot]:
                orders.append(SingleBattleOrder(move, move_target=target, dynamax=True))
            if battle.can_tera[slot]:
                orders.append(SingleBattleOrder(move, move_target=target, terastallize=True))

    if not battle.trapped[slot]:
        for switch in switches:
            orders.append(SingleBattleOrder(switch))
    return orders


def slot_action_mask(battle, slot, act_size) -> MaskVector:
    cache = getattr(battle, "_slot_action_cache", None)
    token = _mask_state_key(battle)
    if (
        cache
        and cache.get("token") == token
        and cache.get("act_size") == act_size
        and cache.get("masks")
    ):
        cached_masks = cache["masks"]
        if slot < len(cached_masks) and cached_masks[slot] is not None:
            return list(cached_masks[slot])

    masks = [
        _build_slot_action_mask(battle, 0, act_size),
        _build_slot_action_mask(battle, 1, act_size),
    ]
    battle._slot_action_cache = {"token": token, "act_size": act_size, "masks": masks}
    return list(masks[slot])


def _build_slot_action_mask(battle, slot, act_size) -> MaskVector:
    legal_actions = set()
    if slot < 0 or slot >= 2:
        return [1] * act_size

    force_flags = getattr(battle, "force_switch", [False, False])
    force_active = bool(force_flags[slot]) if slot < len(force_flags) else False
    available_switches = getattr(battle, "available_switches", [])
    both_forced_single = (
        len(force_flags) >= 2
        and all(bool(flag) for flag in force_flags[:2])
        and all(
            len(available_switches[idx]) == 1 if idx < len(available_switches) else False
            for idx in range(2)
        )
    )

    def _is_default_single(single_order):
        if single_order is None:
            return True
        if isinstance(single_order, DefaultBattleOrder):
            return True
        if isinstance(getattr(single_order, "order", None), str):
            normalized = single_order.order.strip().lower()
            return normalized in {"/choose default", "/choose pass"}
        return False

    def _maybe_add(action_value):
        if action_value is None:
            return
        try:
            value = int(action_value)
        except Exception:
            return
        if 0 <= value < act_size:
            legal_actions.add(value)

    def _collect_joint_orders():
        try:
            orders_a = _legal_orders(battle, 0)
        except Exception:
            orders_a = []
        try:
            orders_b = _legal_orders(battle, 1)
        except Exception:
            orders_b = []

        try:
            joint_orders = DoubleBattleOrder.join_orders(orders_a, orders_b)
        except Exception:
            joint_orders = []

        if not joint_orders:
            joint_orders = [DoubleBattleOrder(first_order=None, second_order=None)]

        for joint in joint_orders:
            single = joint.first_order if slot == 0 else joint.second_order
            if force_active and _is_default_single(single):
                continue
            try:
                vector = DoublesEnv.order_to_action(joint, battle, fake=False, strict=True)
            except Exception:
                try:
                    vector = DoublesEnv.order_to_action(joint, battle, fake=False, strict=False)
                except Exception:
                    continue
            arr = np.asarray(vector, dtype=int).reshape(-1)
            if slot >= arr.size:
                continue
            _maybe_add(arr[slot])

    def _collect_individual_orders():
        try:
            single_orders = _legal_orders(battle, slot)
        except Exception:
            single_orders = []
        for order in single_orders:
            if force_active and _is_default_single(order):
                continue
            try:
                raw_action = DoublesEnv._order_to_action_individual(order, battle, True, slot)
            except Exception:
                try:
                    raw_action = DoublesEnv._order_to_action_individual(order, battle, False, slot)
                except Exception:
                    continue
            _maybe_add(raw_action)

    _collect_joint_orders()
    _collect_individual_orders()

    if both_forced_single and force_active:
        legal_actions = {0}
    elif not legal_actions:
        if force_active:
            return [0] * act_size
        legal_actions.add(0)

    mask = [0] * act_size
    for action_raw in legal_actions:
        action_index = int(action_raw)
        if 0 <= action_index < act_size:
            mask[action_index] = 1
    return mask


def combine_slot_masks(mask_a: MaskLike, mask_b: MaskLike) -> np.ndarray:
    # Contract: concatenate (slot0, slot1) to shape (2 * act_size,).
    # Changing this order breaks mask semantics in training/eval.
    # Parity is enforced by `tests/test_feature_golden_mask.py` (golden_slot_mask.npy).
    arr_a = np.asarray(list(mask_a), dtype=np.uint8)
    arr_b = np.asarray(list(mask_b), dtype=np.uint8)
    return np.concatenate((arr_a, arr_b), axis=0)


def _mask_state_key(battle):
    def _bool_tuple(values):
        return tuple(bool(entry) for entry in (values or [])[:2])

    def _moves_signature(all_moves):
        signature = []
        iterable = list(all_moves)[:2] if isinstance(all_moves, Sequence) else []
        for slot_moves in iterable:
            slot_sig = []
            for move in slot_moves or []:
                slot_sig.append(
                    (
                        getattr(move, "id", None),
                        bool(getattr(move, "disabled", False)),
                        int(getattr(move, "current_pp", 0) or 0),
                    )
                )
            signature.append(tuple(slot_sig))
        while len(signature) < 2:
            signature.append(())
        return tuple(signature)

    def _switch_signature(all_switches):
        signature = []
        iterable = list(all_switches)[:2] if isinstance(all_switches, Sequence) else []
        for slot_switches in iterable:
            slot_sig = []
            for mon in slot_switches or []:
                slot_sig.append(
                    (
                        getattr(mon, "species", getattr(mon, "base_species", None)),
                        bool(getattr(mon, "fainted", False)),
                    )
                )
            signature.append(tuple(slot_sig))
        while len(signature) < 2:
            signature.append(())
        return tuple(signature)

    return (
        getattr(battle, "turn", 0),
        _bool_tuple(getattr(battle, "force_switch", [])),
        _bool_tuple(getattr(battle, "trapped", [])),
        _bool_tuple(getattr(battle, "can_mega_evolve", [])),
        _bool_tuple(getattr(battle, "can_z_move", [])),
        _bool_tuple(getattr(battle, "can_dynamax", [])),
        _bool_tuple(getattr(battle, "can_tera", [])),
        _moves_signature(getattr(battle, "available_moves", [])),
        _switch_signature(getattr(battle, "available_switches", [])),
    )


__all__ = [
    "_build_slot_action_mask",
    "_legal_orders",
    "_mask_state_key",
    "combine_slot_masks",
    "slot_action_mask",
]
