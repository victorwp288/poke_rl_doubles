#!/usr/bin/env python3
import argparse
import itertools
import json
import random
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from poke_env import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ServerConfiguration,
    ShowdownServerConfiguration,
)
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.baselines import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.player.battle_order import DoubleBattleOrder
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import section  # noqa: E402
from src.core.features import _legal_orders  # noqa: E402
from src.online.env import make_maskable_env  # noqa: E402
from src.online.init import configure_action_head, load_behavior_clone_weights  # noqa: E402
from src.utils.teambuilders import read_showdown_team  # noqa: E402

BASELINE_PLAYERS = {
    "simple": SimpleHeuristicsPlayer,
    "maxbp": MaxBasePowerPlayer,
    "random": RandomPlayer,
}

DEFAULT_WATCHDOG_SECONDS = 45.0


def _unique_username(base):
    prefix = (base or "bot")[:11]
    suffix = uuid.uuid4().hex[:6]
    return f"{prefix}{suffix}"


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


def _step_with_repair(env, action, mask, max_attempts=5):
    candidate = np.asarray(action, dtype=int)
    attempt = 0
    current_mask = mask
    last_error = None
    while attempt < max_attempts:
        mask_arr, act_size, _ = _mask_view(env, None)
        if mask_arr is None or act_size is None:
            mask_arr, act_size, _ = _mask_view(env, current_mask)
        if not _mask_allows(candidate, env, mask_arr, act_size):
            _log_invalid_action(
                "mask_reject",
                env,
                candidate,
                note={"attempt": attempt, "candidate": _action_to_list(candidate)},
            )
            repaired = _repair_action(env, candidate, mask_arr)
            if repaired is None:
                _log_invalid_action(
                    "repair_unavailable",
                    env,
                    candidate,
                    note={"attempt": attempt, "candidate": _action_to_list(candidate)},
                )
                break
            candidate = np.asarray(repaired, dtype=int)
            current_mask = mask_arr
            attempt += 1
            continue
        if not _is_action_valid(candidate, env):
            _log_invalid_action(
                "preflight_invalid",
                env,
                candidate,
                note={"attempt": attempt, "candidate": _action_to_list(candidate)},
            )
            repaired = _repair_action(env, candidate, mask_arr)
            if repaired is None:
                _log_invalid_action(
                    "repair_unavailable",
                    env,
                    candidate,
                    note={"attempt": attempt, "candidate": _action_to_list(candidate)},
                )
                break
            candidate = np.asarray(repaired, dtype=int)
            current_mask = mask_arr
            attempt += 1
            continue
        try:
            return env.step(candidate)
        except AssertionError as exc:
            if "invalid action" not in str(exc):
                raise
            stage = "env_step_failure" if attempt == 0 else "repair_failure"
            _log_invalid_action(
                stage,
                env,
                candidate,
                note={"attempt": attempt, "candidate": _action_to_list(candidate)},
            )
            last_error = exc
            current_mask = None
            repaired = _repair_action(env, candidate, None)
            if repaired is None:
                _log_invalid_action(
                    "repair_unavailable",
                    env,
                    candidate,
                    note={"attempt": attempt, "candidate": _action_to_list(candidate)},
                )
                break
            _log_invalid_action(
                "repair_attempt",
                env,
                repaired,
                note={"attempt": attempt + 1, "candidate": _action_to_list(repaired)},
            )
            candidate = np.asarray(repaired, dtype=int)
            current_mask = None
            attempt += 1
    raise AssertionError("invalid action (repair failed)") from last_error


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


def _resolve_server_configuration(url):
    token = (url or "").strip()
    if not token:
        return LocalhostServerConfiguration
    lowered = token.lower()
    if lowered in {"showdown", "https://play.pokemonshowdown.com"}:
        return ShowdownServerConfiguration
    if lowered in {"local", "localhost", "http://localhost:8000"}:
        return LocalhostServerConfiguration
    websocket_url = token
    if websocket_url.startswith("http://"):
        websocket_url = "ws://" + websocket_url[len("http://") :]
    elif websocket_url.startswith("https://"):
        websocket_url = "wss://" + websocket_url[len("https://") :]
    elif not websocket_url.startswith(("ws://", "wss://")):
        websocket_url = f"ws://{websocket_url}"
    if not websocket_url.endswith("/websocket"):
        websocket_url = websocket_url.rstrip("/") + "/websocket"
    return ServerConfiguration(websocket_url, ShowdownServerConfiguration.authentication_url)


def _policy_kwargs():
    online_cfg = section("online")
    hidden_dim = int(online_cfg.get("policy_hidden_dim", 512))
    hidden_layers = int(online_cfg.get("policy_hidden_layers", 2))
    return {
        "net_arch": {
            "pi": [hidden_dim] * hidden_layers,
            "vf": [hidden_dim] * hidden_layers,
        },
        "activation_fn": torch.nn.ReLU,
        "ortho_init": False,
    }


def _build_opponent(kind, battle_format, server_cfg):
    factory = BASELINE_PLAYERS.get(kind)
    if factory is None:
        raise ValueError(f"unknown opponent '{kind}'")
    base_label = (kind or factory.__name__).replace(" ", "")
    username = _unique_username(base_label)
    account = AccountConfiguration(username, None)
    return factory(
        account_configuration=account,
        battle_format=battle_format,
        max_concurrent_battles=1,
        server_configuration=server_cfg,
    )


def _evaluate(model, eval_cfg, server_cfg, team_text):
    episodes = int(eval_cfg.get("episodes", 10))
    opponent_pool = eval_cfg.get("opponent_pool") or [eval_cfg.get("opponent", "simple")]
    battle_format = eval_cfg.get("battle_format", "gen9doublesou")
    watchdog_seconds = float(eval_cfg.get("watchdog_seconds", DEFAULT_WATCHDOG_SECONDS))
    max_steps = int(eval_cfg.get("max_steps", 300))
    rewards = []
    outcomes = []
    details = []
    per_opponent = {}

    for episode in range(episodes):
        opponent_kind = opponent_pool[episode % len(opponent_pool)]
        opp_bucket = per_opponent.setdefault(
            opponent_kind,
            {"wins": 0, "losses": 0, "draws": 0, "timeouts": 0, "errors": 0, "rewards": []},
        )
        print(
            f"[eval] episode {episode + 1}/{episodes} opponent={opponent_kind}",
            flush=True,
        )
        opponent = _build_opponent(opponent_kind, battle_format, server_cfg)
        env = DummyVecEnv(
            [
                lambda opp=opponent: make_maskable_env(
                    opponent=opp,
                    battle_format=battle_format,
                    team=team_text,
                    server_configuration=server_cfg,
                )
            ]
        )
        obs = env.reset()
        try:
            mask_arr, _, _ = _ensure_mask(env.envs[0].action_masks(), env)
        except Exception:
            mask_arr = None
        mask = mask_arr
        done = False
        total = 0.0
        last_info = {}
        step_count = 0
        episode_timeout = False
        episode_error = None
        last_progress = time.monotonic()
        last_turn = None
        while not done:
            if watchdog_seconds and (time.monotonic() - last_progress) > watchdog_seconds:
                elapsed = time.monotonic() - last_progress
                print(
                    f"[eval]   watchdog timeout after {elapsed:.1f}s without progress",
                    flush=True,
                )
                episode_timeout = True
                break
            action_mask = mask
            action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
            # hard clamp to action space bounds
            action_space = env.envs[0].action_space
            if isinstance(action_space, spaces.MultiDiscrete):
                nvec = action_space.nvec
                for i in range(len(action[0])):
                    if i < len(nvec):
                        action[0][i] = np.clip(action[0][i], 0, nvec[i] - 1)
            action, sanitized_mask = _sanitize_action(action, action_mask, env)
            try:
                obs, reward, done_arr, info = _step_with_repair(env, action, sanitized_mask)
            except AssertionError as exc:
                episode_error = str(exc)
                print(
                    f"[eval]   aborting episode {episode + 1}: {exc}",
                    flush=True,
                )
                break
            last_progress = time.monotonic()
            total += float(reward[0])
            done = bool(done_arr[0])
            # If the battle is finished but done is False, force termination.
            base_env = getattr(env, "envs", [None])[0]
            base_env = getattr(base_env, "base_env", base_env)
            battle = getattr(base_env, "battle1", None) if base_env is not None else None
            if info and isinstance(info, list | tuple) and info[0]:
                last_info = info[0]
                if "action_mask" in last_info:
                    mask_arr, _, _ = _ensure_mask(last_info["action_mask"], env)
                    mask = mask_arr
            else:
                try:
                    mask_arr, _, _ = _ensure_mask(env.envs[0].action_masks(), env)
                    mask = mask_arr
                except Exception:
                    mask = None
            if isinstance(last_info, dict):
                stats = last_info.get("battle_stats")
                if isinstance(stats, dict):
                    turn = stats.get("turn")
                    if turn is not None and turn != last_turn:
                        last_progress = time.monotonic()
                        last_turn = turn
                    result = stats.get("result")
                    if result:
                        done = True
            if battle is not None and getattr(battle, "finished", False):
                done = True
            step_count += 1
            if max_steps and step_count >= max_steps:
                print(f"[eval]   max_steps reached ({max_steps}), forcing episode end", flush=True)
                done = True
            if step_count % 5 == 0 or done:
                turn = None
                if isinstance(last_info, dict):
                    stats = last_info.get("battle_stats")
                    if isinstance(stats, dict):
                        turn = stats.get("turn")
                turn_text = f" turn={int(turn)}" if turn is not None else ""
                print(
                    f"[eval]   step={step_count}{turn_text} total_reward={total:.3f}",
                    flush=True,
                )
        env.close()
        stats = (last_info.get("battle_stats") if isinstance(last_info, dict) else None) or {}
        result = stats.get("result", "unknown")
        if episode_timeout:
            result = "timeout"
        if episode_error is not None and not episode_timeout:
            result = "error"
        rewards.append(total)
        outcomes.append(result)
        opp_bucket["rewards"].append(total)
        details.append(
            {
                "episode": episode,
                "opponent": opponent_kind,
                "result": result,
                "reward": total,
                "stats": stats,
                "timeout": episode_timeout,
                "error": episode_error,
                "steps": step_count,
            }
        )
        if result == "win":
            opp_bucket["wins"] += 1
        elif result == "draw":
            opp_bucket["draws"] += 1
        elif result == "timeout":
            opp_bucket["timeouts"] += 1
        elif result == "error":
            opp_bucket["errors"] += 1
        else:
            opp_bucket["losses"] += 1
    return rewards, outcomes, details, per_opponent


def build_arg_parser(defaults):
    parser = argparse.ArgumentParser(description="Evaluate BC policy vs bots")

    parser.add_argument("--episodes", type=int, default=defaults.get("episodes"))
    parser.add_argument("--opponent", type=str, default=defaults.get("opponent"))
    parser.add_argument(
        "--opponent-pool",
        type=str,
        default=",".join(defaults.get("opponent_pool", [])),
        help="Comma-separated list",
    )
    parser.add_argument("--battle-format", type=str, default=defaults.get("battle_format"))
    parser.add_argument("--tensorboard-dir", type=Path, default=defaults.get("tensorboard_dir"))
    parser.add_argument("--output-path", type=Path, default=defaults.get("output_path"))
    parser.add_argument("--summary-path", type=Path, default=defaults.get("summary_path"))
    parser.add_argument("--checkpoint", type=Path, default=defaults.get("checkpoint"))
    parser.add_argument("--stats-path", type=Path, default=defaults.get("stats_path"))
    parser.add_argument("--our-team-path", type=Path, default=defaults.get("our_team_path"))
    parser.add_argument("--server-url", type=str, default=defaults.get("server_url"))
    parser.add_argument(
        "--gate-best-on-win-rate",
        action="store_true",
        default=defaults.get("gate_best_on_win_rate", False),
    )
    parser.add_argument("--best-metrics-path", type=Path, default=defaults.get("best_metrics_path"))
    parser.add_argument("--best-policy-path", type=Path, default=defaults.get("best_policy_path"))
    parser.add_argument("--best-stats-path", type=Path, default=defaults.get("best_stats_path"))

    return parser


def merge_cli_overrides(defaults, args):
    settings = defaults.copy()

    for key in [
        "episodes",
        "opponent",
        "battle_format",
        "tensorboard_dir",
        "output_path",
        "summary_path",
        "checkpoint",
        "stats_path",
        "our_team_path",
        "server_url",
        "best_metrics_path",
        "best_policy_path",
        "best_stats_path",
    ]:
        value = getattr(args, key)
        if value is not None:
            settings[key] = value

    if args.opponent_pool:
        settings["opponent_pool"] = [x.strip() for x in args.opponent_pool.split(",") if x.strip()]
    if getattr(args, "gate_best_on_win_rate", False):
        settings["gate_best_on_win_rate"] = True

    return settings


def load_defaults():
    cfg = section("evaluation") or {}
    bc = cfg.get("bc_vs_bots", {}) or {}

    return {
        "episodes": bc.get("episodes", 10),
        "opponent": bc.get("opponent", "simple"),
        "opponent_pool": bc.get("opponent_pool", ["simple", "maxbp", "random"]),
        "battle_format": bc.get("battle_format", "gen9doublesou"),
        "tensorboard_dir": Path(bc.get("tensorboard_dir", "outputs/tensorboard/eval")),
        "output_path": Path(bc.get("output_path", "outputs/eval/bc_vs_bots.jsonl")),
        "summary_path": bc.get("summary_path"),
        "checkpoint": Path(bc.get("checkpoint", "outputs/models/bc_policy.pt")),
        "stats_path": Path(bc.get("stats_path", "outputs/models/bc_stats.json")),
        "our_team_path": Path(bc.get("our_team_path", "teams/gen9dou_fixed.txt")),
        "server_url": bc.get("server_url", "http://localhost:8000"),
        "gate_best_on_win_rate": bc.get("gate_best_on_win_rate", False),
        "best_metrics_path": bc.get("best_metrics_path"),
        "best_policy_path": bc.get("best_policy_path"),
        "best_stats_path": bc.get("best_stats_path"),
    }


def main():
    defaults = load_defaults()
    parser = build_arg_parser(defaults)
    args = parser.parse_args()
    eval_cfg = merge_cli_overrides(defaults, args)

    if not eval_cfg:
        raise RuntimeError("evaluation.bc_vs_bots not configured")
    offline_cfg = section("offline")
    checkpoint = Path(
        eval_cfg.get("checkpoint") or offline_cfg.get("policy_path", "outputs/models/bc_policy.pt")
    )
    stats_path = eval_cfg.get("stats_path") or offline_cfg.get("stats_path")
    stats_path = Path(stats_path) if stats_path else None
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing checkpoint {checkpoint}")
    tensorboard_dir = eval_cfg.get("tensorboard_dir")
    writer = None
    if tensorboard_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(Path(tensorboard_dir)))
        except Exception as exc:
            print(f"[warn] tensorboard unavailable: {exc}")
    opponent_kind = eval_cfg.get("opponent", "simple")
    battle_format = eval_cfg.get("battle_format", "gen9doublesou")
    collect_cfg = section("imitation_collect")
    team_path = Path(
        eval_cfg.get("our_team_path")
        or (collect_cfg.get("our_team_path") if isinstance(collect_cfg, dict) else None)
        or "teams/gen9dou_fixed.txt"
    )
    team_text = read_showdown_team(team_path)
    server_url = eval_cfg.get("server_url")
    if not server_url and isinstance(collect_cfg, dict):
        server_url = collect_cfg.get("server_url")
    server_cfg = _resolve_server_configuration(server_url)

    def _env_factory():
        opponent = _build_opponent(opponent_kind, battle_format, server_cfg)
        return make_maskable_env(
            opponent=opponent,
            battle_format=battle_format,
            team=team_text,
            server_configuration=server_cfg,
        )

    env = DummyVecEnv([_env_factory])
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        tensorboard_log=None,
        verbose=0,
        policy_kwargs=_policy_kwargs(),
    )

    offline_cfg = section("offline") or {}
    head_hidden = eval_cfg.get("policy_head_mlp_dim") or offline_cfg.get("head_mlp_dim") or 512
    configure_action_head(model.policy, head_hidden)

    # verify action space before loading weights
    action_space = env.envs[0].action_space
    if isinstance(action_space, spaces.MultiDiscrete):
        print(f"[info] action_space={action_space.nvec.tolist()}", flush=True)

    load_behavior_clone_weights(
        policy=model.policy,
        checkpoint_path=checkpoint,
        stats_path=stats_path,
    )
    env.close()
    model.policy.eval()

    # verify action distribution dimensions
    action_dist = model.policy.action_dist
    if hasattr(action_dist, "action_dims"):
        print(f"[info] action_dims={action_dist.action_dims}", flush=True)

    rewards, outcomes, details, per_opponent = _evaluate(model, eval_cfg, server_cfg, team_text)
    wins = sum(1 for result in outcomes if result == "win")
    losses = sum(1 for result in outcomes if result == "loss")
    draws = sum(1 for result in outcomes if result == "draw")
    mean_reward = float(np.mean(rewards)) if rewards else 0.0
    total_episodes = len(outcomes) or 1
    win_rate = wins / total_episodes
    print(
        f"[eval] episodes={len(outcomes)} win={wins} loss={losses} draw={draws} "
        f"reward_mean={mean_reward:.3f} win_rate={win_rate:.3f}",
        flush=True,
    )
    out_path = Path(eval_cfg.get("output_path", "outputs/eval/bc_eval.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in details:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "episodes": len(outcomes),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "mean_reward": mean_reward,
        "per_opponent": {},
    }
    for opponent_kind, bucket in per_opponent.items():
        opp_total = (
            bucket["wins"]
            + bucket["losses"]
            + bucket["draws"]
            + bucket["timeouts"]
            + bucket["errors"]
        ) or 1
        opp_mean_reward = float(np.mean(bucket["rewards"])) if bucket["rewards"] else 0.0
        summary["per_opponent"][opponent_kind] = {
            "wins": bucket["wins"],
            "losses": bucket["losses"],
            "draws": bucket["draws"],
            "timeouts": bucket["timeouts"],
            "errors": bucket["errors"],
            "win_rate": bucket["wins"] / opp_total,
            "mean_reward": opp_mean_reward,
        }
    summary_path = Path(
        eval_cfg.get("summary_path") or out_path.with_name(f"{out_path.stem}_summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] summary -> {summary_path}", flush=True)
    if writer is not None:
        writer.add_scalar("eval/win_rate", win_rate, 0)
        writer.add_scalar("eval/loss_rate", losses / total_episodes, 0)
        writer.add_scalar("eval/draw_rate", draws / total_episodes, 0)
        writer.add_scalar("eval/reward_mean", mean_reward, 0)
        writer.flush()
        writer.close()

    if eval_cfg.get("gate_best_on_win_rate"):
        best_metrics_path = Path(
            eval_cfg.get("best_metrics_path") or out_path.with_name(f"{out_path.stem}_best.json")
        )
        best_policy_path = eval_cfg.get("best_policy_path")
        best_stats_path = eval_cfg.get("best_stats_path")
        previous_best = None
        if best_metrics_path.exists():
            try:
                previous = json.loads(best_metrics_path.read_text(encoding="utf-8"))
                previous_best = float(previous.get("win_rate", 0.0))
            except Exception:
                previous_best = None
        improved = previous_best is None or win_rate > previous_best
        if improved:
            best_metrics_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[eval] new best win_rate={win_rate:.3f} -> {best_metrics_path}", flush=True)
            if best_policy_path:
                try:
                    shutil.copy2(checkpoint, Path(best_policy_path))
                    print(f"[eval] copied checkpoint to {best_policy_path}", flush=True)
                except Exception as exc:
                    print(f"[warn] failed to copy best policy: {exc}", flush=True)
            if best_stats_path and stats_path and Path(stats_path).exists():
                try:
                    shutil.copy2(Path(stats_path), Path(best_stats_path))
                    print(f"[eval] copied stats to {best_stats_path}", flush=True)
                except Exception as exc:
                    print(f"[warn] failed to copy best stats: {exc}", flush=True)


if __name__ == "__main__":
    main()
