from pathlib import Path
from typing import Any

import numpy as np
from poke_env import AccountConfiguration
from poke_env.player.baselines import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.online.env import make_maskable_env
from src.online.opponents import PolicyOpponentPlayer, _unique_username

from .settings import _step_delay
from .specs import OpponentSpec


def _make_opponent(spec: OpponentSpec, *, server_cfg, battle_format: str, act_size: int):
    opp_account = AccountConfiguration(_unique_username("EvalOpp"), None)
    common = dict(
        account_configuration=opp_account,
        battle_format=battle_format,
        max_concurrent_battles=1,
        server_configuration=server_cfg,
    )
    if spec.kind == "simple":
        return SimpleHeuristicsPlayer(**common)
    if spec.kind == "maxbp":
        return MaxBasePowerPlayer(**common)
    if spec.kind == "random":
        return RandomPlayer(**common)
    if spec.kind == "policy":
        if spec.policy_path is None:
            raise ValueError("policy opponent requires policy_path")
        return PolicyOpponentPlayer(model_path=spec.policy_path, act_size=act_size, **common)
    raise ValueError(f"unknown opponent kind: {spec.kind}")


def _make_eval_env(opponent, settings: dict[str, Any], team_text: str, server_cfg):
    player_account = AccountConfiguration(_unique_username("EvalOur"), None)
    env = make_maskable_env(
        opponent=opponent,
        battle_format=settings["battle_format"],
        rewards=settings["rewards"],
        team=team_text,
        account_configuration1=player_account,
        server_configuration=server_cfg,
        step_delay=_step_delay(settings),
        console_log_mode="off",
        console_log_interval_sec=settings.get("console_log_interval_sec", 5.0),
    )
    return DummyVecEnv([lambda: env])


def _maybe_wrap_vecnormalize(venv, policy_path: Path):
    vec_path = policy_path.with_name(f"{policy_path.stem}_vecnorm.pkl")
    if vec_path.exists():
        # Eval always loads <stem>_vecnorm.pkl and disables reward normalization for parity.
        # Must match the training vecnorm stats to make scores comparable.
        wrapped = VecNormalize.load(str(vec_path), venv)
        wrapped.training = False
        wrapped.norm_reward = False
        return wrapped, vec_path
    return venv, None


def _find_mask_env(venv):
    current = venv
    while hasattr(current, "venv"):
        current = current.venv
    if hasattr(current, "envs") and current.envs:
        return current.envs[0]
    return current


def _ensure_mask_shape(mask, act_size: int):
    if mask is None:
        return None
    arr = np.asarray(mask, dtype=np.int64).reshape(-1)
    if arr.size == act_size * 2:
        return arr
    if arr.size == act_size:
        # Expand single-slot mask into joint (slot0, slot1) layout.
        return np.concatenate((arr, arr), axis=0)
    return None


def _sanitize_action(raw_action, mask, act_size: int):
    if raw_action is None:
        return None
    vector = np.asarray(raw_action, dtype=np.int64).reshape(-1)
    if vector.size < 2:
        return raw_action
    if mask is None:
        return raw_action
    mask = _ensure_mask_shape(mask, act_size)
    if mask is None:
        return raw_action
    slot_masks = mask.reshape(2, act_size)
    sanitized = vector.copy()
    changed = False
    for slot in range(2):
        if slot >= sanitized.size:
            break
        action_idx = int(sanitized[slot])
        if action_idx < 0 or action_idx >= act_size:
            sanitized[slot] = 0
            changed = True
            continue
        if slot_masks[slot, action_idx] == 0:
            sanitized[slot] = 0
            changed = True
    if changed:
        return sanitized
    return raw_action


def _action_payload(action):
    if action is None:
        return None
    arr = np.asarray(action, dtype=int).reshape(-1)
    return [int(x) for x in arr]


__all__ = [
    "_action_payload",
    "_ensure_mask_shape",
    "_find_mask_env",
    "_make_eval_env",
    "_make_opponent",
    "_maybe_wrap_vecnormalize",
    "_sanitize_action",
]
