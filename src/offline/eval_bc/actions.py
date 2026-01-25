import random

import numpy as np
from gymnasium import spaces
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.battle_order import DoubleBattleOrder


def _safe_action_mask(env):
    try:
        return env.envs[0].action_masks()
    except Exception:
        return None


def _mask_view(env, mask):
    mask_arr, act_size, action_space = _ensure_mask(mask, env)
    if mask_arr is not None and act_size is not None:
        return mask_arr, act_size, action_space
    fallback = _safe_action_mask(env)
    return _ensure_mask(fallback, env)


def _mask_allows(action, env, mask_arr, act_size):
    if mask_arr is None or act_size is None:
        return True
    vector = np.asarray(action, dtype=int).reshape(-1)
    try:
        slot_masks = mask_arr.reshape(-1, act_size)
    except ValueError:
        return True
    action_space = getattr(env.envs[0], "action_space", None)
    nvec = [int(n) for n in getattr(action_space, "nvec", [])]
    for idx, value in enumerate(vector):
        if idx >= slot_masks.shape[0]:
            break
        if idx < len(nvec) and (value < 0 or value >= nvec[idx]):
            return False
        if value < 0 or value >= slot_masks.shape[1]:
            return False
        if not slot_masks[idx, value]:
            return False
    return True


def _current_battle(env):
    maskable_env = env.envs[0]
    base_env = getattr(maskable_env, "base_env", None)
    return getattr(base_env, "battle1", None) if base_env is not None else None


def _describe_single_order(order):
    if order is None:
        return None
    payload = {
        "mega": bool(getattr(order, "mega", False)),
        "z": bool(getattr(order, "z_move", False)),
        "dynamax": bool(getattr(order, "dynamax", False)),
        "tera": bool(getattr(order, "terastallize", False)),
        "target": int(getattr(order, "move_target", 0)),
    }
    raw = getattr(order, "order", None)
    if raw is None:
        payload["kind"] = "none"
    else:
        raw_type = type(raw).__name__
        payload["kind"] = raw_type
        species = getattr(raw, "species", None)
        if species:
            payload["species"] = str(species)
        move_id = getattr(raw, "id", None)
        if move_id:
            payload["move"] = str(move_id)
    return payload


def _describe_action(env, action):
    battle = _current_battle(env)
    if battle is None:
        return None
    vector = np.asarray(action, dtype=int).reshape(-1)
    detail = {
        "vector": [int(x) for x in vector.tolist()],
    }
    try:
        order = DoublesEnv.action_to_order(vector, battle, fake=False, strict=True)
    except AssertionError as exc:
        detail["error"] = str(exc)
        try:
            order = DoublesEnv.action_to_order(vector, battle, fake=False, strict=False)
        except Exception as inner_exc:
            detail["fallback_error"] = str(inner_exc)
            return detail
    if isinstance(order, DoubleBattleOrder):
        detail["first"] = _describe_single_order(order.first_order)
        detail["second"] = _describe_single_order(order.second_order)
    else:
        detail["order"] = type(order).__name__
    return detail


def _ensure_mask(mask, env):
    action_space = getattr(env.envs[0], "action_space", None)
    if not isinstance(action_space, spaces.MultiDiscrete):
        return None, None, action_space
    arr = None
    if mask is not None:
        arr = np.asarray(mask, dtype=bool)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
    num_slots = len(action_space.nvec)
    act_size = getattr(env.envs[0], "_act_size", None)
    if act_size is None:
        act_size = int(action_space.nvec[0]) if num_slots else None
    if act_size is None and arr is not None and num_slots:
        act_size = arr.shape[1] // num_slots
    return arr, act_size, action_space


def _sanitize_action(action, mask, env):
    mask_arr, act_size, action_space = _ensure_mask(mask, env)
    if not isinstance(action_space, spaces.MultiDiscrete) or mask_arr is None or act_size is None:
        return action, None
    raw = np.asarray(action[0]).astype(int).flatten()
    num_slots = len(action_space.nvec)
    nvec = [int(n) for n in action_space.nvec]
    if raw.size < num_slots:
        raw = np.pad(raw, (0, num_slots - raw.size), constant_values=0)
    slot_masks = mask_arr.reshape(-1, act_size)
    cleaned = []
    for slot in range(num_slots):
        slot_mask = slot_masks[slot]
        legal = np.where(slot_mask)[0]
        if legal.size == 0:
            legal = np.arange(min(act_size, nvec[slot]))
        choice = raw[slot] if slot < raw.size else 0
        # enforce hard bounds from action space
        if choice < 0 or choice >= nvec[slot]:
            choice = 0
        # then check mask
        if choice < 0 or choice >= act_size or not slot_mask[choice]:
            choice = int(random.choice(legal.tolist()))
        cleaned.append(choice)
    action[0] = np.array(cleaned, dtype=int)
    return action, slot_masks.reshape(1, -1)


__all__ = [
    "_current_battle",
    "_describe_action",
    "_describe_single_order",
    "_ensure_mask",
    "_mask_allows",
    "_mask_view",
    "_safe_action_mask",
    "_sanitize_action",
]
