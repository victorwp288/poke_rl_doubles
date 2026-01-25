import json

import numpy as np

from .actions import _describe_action


def _action_to_list(action):
    if action is None:
        return None
    arr = np.asarray(action).astype(int).reshape(-1)
    return [int(x) for x in arr]


def _battle_debug_info(env):
    info = {}
    maskable_env = env.envs[0]
    base_env = getattr(maskable_env, "base_env", None)
    battle = getattr(base_env, "battle1", None) if base_env is not None else None
    if battle is None:
        return info
    force_switch = getattr(battle, "force_switch", None)
    if force_switch is not None:
        info["force_switch"] = [bool(x) for x in force_switch]
    info["turn"] = int(getattr(battle, "turn", 0) or 0)
    info["trapped"] = [bool(x) for x in getattr(battle, "trapped", [False, False])]
    moves = []
    for idx in range(2):
        slot_moves = []
        if idx < len(battle.available_moves):
            slot_moves = [getattr(move, "id", str(move)) for move in battle.available_moves[idx]]
        moves.append(slot_moves)
    switches = []
    for idx in range(2):
        slot_switches = []
        if idx < len(battle.available_switches):
            slot_switches = [
                getattr(mon, "species", str(mon)) for mon in battle.available_switches[idx]
            ]
        switches.append(slot_switches)
    info["available_moves"] = moves
    info["available_switches"] = switches
    info["can_mega"] = [bool(x) for x in getattr(battle, "can_mega_evolve", [False, False])]
    info["can_z"] = [bool(x) for x in getattr(battle, "can_z_move", [False, False])]
    info["can_dyna"] = [bool(x) for x in getattr(battle, "can_dynamax", [False, False])]
    info["can_tera"] = [bool(x) for x in getattr(battle, "can_tera", [False, False])]
    return info


def _log_invalid_action(stage, env, action, note=None):
    payload = {
        "stage": stage,
        "action": _action_to_list(action),
        "battle": _battle_debug_info(env),
        "note": note,
        "order": _describe_action(env, action),
    }
    try:
        message = json.dumps(payload)
    except TypeError:
        message = str(payload)
    print(f"[debug] invalid_action {message}", flush=True)


__all__ = ["_action_to_list", "_battle_debug_info", "_log_invalid_action"]
