import random

import numpy as np
from gymnasium import spaces

from .actions import _current_battle, _ensure_mask
from .debug import _action_to_list, _log_invalid_action
from .repair_candidates import (
    _first_valid_action,
    _is_action_valid,
    _joint_legal_actions,
    _wrap_action,
)


def _repair_action(env, action, mask):
    mask_arr, act_size, action_space = _ensure_mask(mask, env)
    if not isinstance(action_space, spaces.MultiDiscrete):
        _log_invalid_action("no_multidiscrete", env, action)
        return None
    if mask_arr is None or act_size is None:
        try:
            fresh_arr, fresh_act_size, fresh_space = _ensure_mask(env.envs[0].action_masks(), env)
        except Exception:
            fresh_arr = None
            fresh_act_size = None
            fresh_space = action_space
        if isinstance(fresh_space, spaces.MultiDiscrete):
            action_space = fresh_space
        if fresh_arr is not None:
            mask_arr = fresh_arr
        if fresh_act_size is not None:
            act_size = fresh_act_size
    if mask_arr is None or act_size is None:
        _log_invalid_action("no_mask_available", env, action)
        return None
    num_slots = len(action_space.nvec)
    nvec = [int(n) for n in action_space.nvec]
    battle = _current_battle(env)
    force_flags = list(getattr(battle, "force_switch", [])) if battle is not None else []
    if force_flags and all(bool(x) for x in force_flags):
        switches = getattr(battle, "available_switches", [])
        has_switch_option = False
        summary = []
        for idx, flag in enumerate(force_flags):
            slot_summary = {"slot": idx, "flag": bool(flag)}
            options = switches[idx] if idx < len(switches) else []
            names = []
            healthy = False
            for mon in options or []:
                species = getattr(mon, "species", str(mon))
                names.append(str(species))
                if not getattr(mon, "fainted", False):
                    healthy = True
            slot_summary["options"] = names
            slot_summary["healthy_available"] = healthy
            summary.append(slot_summary)
            if healthy:
                has_switch_option = True
        if not has_switch_option:
            default_action = np.full((1, num_slots), -2, dtype=int)
            _log_invalid_action(
                "force_switch_default",
                env,
                default_action,
                note={"reason": "no_healthy_switch_targets", "summary": summary},
            )
            return default_action
    raw = np.asarray(action[0]).astype(int).flatten()
    if raw.size < num_slots:
        raw = np.pad(raw, (0, num_slots - raw.size), constant_values=0)
    # clamp raw to valid bounds
    for idx in range(len(raw)):
        if idx < len(nvec):
            raw[idx] = max(0, min(raw[idx], nvec[idx] - 1))
    if mask_arr is None:
        try:
            fresh_mask, _, _ = _ensure_mask(env.envs[0].action_masks(), env)
        except Exception:
            fresh_mask = None
        mask_arr = fresh_mask
    try:
        slot_masks = mask_arr.reshape(-1, act_size)
    except Exception:
        slot_masks = None

    def allowed(pair):
        if slot_masks is None:
            return True
        for idx, value in enumerate(pair):
            if idx >= len(nvec) or value < 0 or value >= nvec[idx]:
                return False
            if value >= act_size or not slot_masks[idx, value]:
                return False
        return True

    # first try joint legal actions since they're most likely to work
    legal_pairs = _joint_legal_actions(env)
    if legal_pairs:
        random.shuffle(legal_pairs)
        for pair in legal_pairs[:20]:  # limit attempts
            if not allowed(pair):
                continue
            candidate = _wrap_action(pair, num_slots)
            valid = _is_action_valid(candidate, env)
            if valid:
                _log_invalid_action(
                    "joint_candidate",
                    env,
                    candidate,
                    note={"candidate": _action_to_list(candidate), "valid": True},
                )
                return candidate

    # try simple fallbacks
    fallbacks = []
    if num_slots == 1:
        fallbacks.append([0])
    else:
        fallbacks.extend(
            [
                [raw[0], 0],
                [0, raw[1]],
                [0, 0],
            ]
        )
    for pair in fallbacks:
        if not allowed(pair):
            continue
        candidate = _wrap_action(pair, num_slots)
        valid = _is_action_valid(candidate, env)
        if valid:
            _log_invalid_action(
                "fallback_candidate",
                env,
                candidate,
                note={"candidate": _action_to_list(candidate), "valid": True},
            )
            return candidate

    # brute force search
    fallback = _first_valid_action(env)
    if fallback is not None:
        _log_invalid_action(
            "bruteforce_candidate",
            env,
            fallback,
            note={"candidate": _action_to_list(fallback), "valid": True},
        )
        return fallback

    # last resort - return the first legal pair or [0, 0]
    if legal_pairs:
        candidate = _wrap_action(legal_pairs[0], num_slots)
        _log_invalid_action(
            "last_resort_legal",
            env,
            candidate,
            note={"candidate": _action_to_list(candidate)},
        )
        return candidate

    default_action = np.zeros((1, num_slots), dtype=int)
    _log_invalid_action(
        "last_resort_default",
        env,
        default_action,
        note={"candidate": _action_to_list(default_action)},
    )
    return default_action


__all__ = ["_repair_action"]
