import itertools

import numpy as np
from gymnasium import spaces
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.battle_order import DoubleBattleOrder

from src.core.action_mask import _legal_orders


def _joint_legal_actions(env):
    maskable_env = env.envs[0]
    base_env = getattr(maskable_env, "base_env", None)
    battle = getattr(base_env, "battle1", None) if base_env is not None else None
    if battle is None:
        return []
    slot_orders = []
    for idx in range(2):
        try:
            orders = _legal_orders(battle, idx)
        except Exception:
            orders = []
        slot_orders.append(orders)
    candidates = []
    joined = DoubleBattleOrder.join_orders(slot_orders[0], slot_orders[1])
    for order in joined:
        try:
            action = DoublesEnv.order_to_action(order, battle, fake=False, strict=True)
        except Exception:
            continue
        candidates.append([int(value) for value in action])
    if not candidates:
        candidates.append([0, 0])
    return candidates


def _wrap_action(pair, num_slots):
    candidate = np.asarray(pair, dtype=int)
    if candidate.size < num_slots:
        candidate = np.pad(candidate, (0, num_slots - candidate.size), constant_values=0)
    return candidate.reshape(1, num_slots)


def _first_valid_action(env):
    maskable_env = env.envs[0]
    base_env = getattr(maskable_env, "base_env", None)
    battle = getattr(base_env, "battle1", None) if base_env is not None else None
    action_space = getattr(maskable_env, "action_space", None)
    if battle is None or not isinstance(action_space, spaces.MultiDiscrete):
        return None
    nvec = [int(n) for n in action_space.nvec]
    combos = itertools.product(*[range(limit) for limit in nvec])
    for combo in combos:
        action_arr = np.asarray(combo, dtype=np.int64)
        try:
            DoublesEnv.action_to_order(action_arr, battle, fake=False, strict=True)
        except Exception:
            continue
        return action_arr.reshape(1, -1)
    return None


def _is_action_valid(action, env):
    maskable_env = env.envs[0]
    base_env = getattr(maskable_env, "base_env", None)
    battle = getattr(base_env, "battle1", None) if base_env is not None else None
    action_space = getattr(maskable_env, "action_space", None)
    if battle is None or not isinstance(action_space, spaces.MultiDiscrete):
        return False
    vector = np.asarray(action, dtype=np.int64).reshape(-1)
    if vector.size == 0:
        return False
    # check bounds first
    nvec = [int(n) for n in action_space.nvec]
    if len(vector) != len(nvec):
        return False
    for value, limit in zip(vector, nvec, strict=False):
        if value < 0 or value >= limit:
            return False
    try:
        order = DoublesEnv.action_to_order(vector, battle, fake=False, strict=True)
        if order is None:
            return False
    except Exception:
        return False
    return True


__all__ = [
    "_first_valid_action",
    "_is_action_valid",
    "_joint_legal_actions",
    "_wrap_action",
]
