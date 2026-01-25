import contextlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.core.env import action_space_size
from src.online.environment import _ensure_server_available, _resolve_server_configuration

from .env import (
    _find_mask_env,
    _make_eval_env,
    _make_opponent,
    _maybe_wrap_vecnormalize,
    _sanitize_action,
)
from .paths import _utc_timestamp
from .policy import _load_policy, _policy_action
from .specs import OpponentSpec, PolicySpec


def _episode_outcome(info):
    if not info:
        return None
    stats = info.get("battle_stats")
    if not stats:
        return None
    result = stats.get("result")
    if result in {"win", "loss", "draw"}:
        return result
    return None


def _episode_turns(info):
    if not info:
        return None
    stats = info.get("battle_stats")
    if not stats:
        return None
    turns = stats.get("turn")
    if turns is None:
        turns = stats.get("turns")
    if turns is None:
        return None
    try:
        return int(turns)
    except Exception:
        return None


def _collect_episode(
    *,
    policy: PolicySpec,
    policy_model,
    opponent: OpponentSpec,
    settings: dict[str, Any],
    overrides: dict[str, Any] | None,
    server_cfg,
    act_size: int,
    max_steps: int,
    team_text: str,
    per_episode_path: Path | None,
    replays_path: Path | None,
    render: bool,
    step_sleep: float,
) -> dict[str, Any]:
    opponent_player = _make_opponent(
        opponent,
        server_cfg=server_cfg,
        battle_format=settings["battle_format"],
        act_size=act_size,
    )
    env = _make_eval_env(opponent_player, settings, team_text, server_cfg)
    env, vec_path = _maybe_wrap_vecnormalize(env, policy.path)
    policy_model = _load_policy(policy.path, env, act_size)
    mask_env = _find_mask_env(env)

    episode = 0
    battles = 0
    wins = losses = draws = 0
    total_rewards = []
    total_turns = []
    sanitized_count = 0
    repaired_count = 0

    obs = env.reset()
    battle = None
    if hasattr(mask_env, "current_battle"):
        battle = mask_env.current_battle
    if battle is None and hasattr(mask_env, "battle"):
        battle = mask_env.battle

    while episode < max_steps:
        action = _policy_action(policy_model, obs)
        mask = None
        if hasattr(mask_env, "action_masks"):
            try:
                mask = mask_env.action_masks()
            except Exception:
                mask = None
        sanitized_action = _sanitize_action(action, mask, act_size)
        if sanitized_action is not None and not np.array_equal(sanitized_action, action):
            sanitized_count += 1
            action = sanitized_action
        obs, reward, done, info = env.step(action)
        info = info[0] if info else {}
        if info.get("repaired_action"):
            repaired_count += 1

        if render:
            with contextlib.suppress(Exception):
                env.render()

        if step_sleep > 0:
            time.sleep(step_sleep)

        if done[0]:
            battles += 1
            outcome = _episode_outcome(info)
            if outcome == "win":
                wins += 1
            elif outcome == "loss":
                losses += 1
            elif outcome == "draw":
                draws += 1

            total_rewards.append(float(reward[0]))
            turns = _episode_turns(info)
            if turns is not None:
                total_turns.append(turns)

            if replays_path is not None:
                try:
                    stats = info.get("battle_stats", {}) if info else {}
                    replay = stats.get("replay") if isinstance(stats, dict) else None
                    if replay:
                        replays_path.mkdir(parents=True, exist_ok=True)
                        stamp = _utc_timestamp()
                        replay_path = replays_path / f"{policy.label}_{opponent.label}_{stamp}.txt"
                        replay_path.write_text(str(replay), encoding="utf-8")
                except Exception:
                    pass

            if per_episode_path is not None:
                per_episode_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "policy": policy.label,
                    "opponent": opponent.label,
                    "reward": float(reward[0]),
                    "result": outcome,
                    "turns": turns,
                    "repaired_action": bool(info.get("repaired_action", False)),
                    "sanitized_action": bool(info.get("sanitized_action", False)),
                }
                with per_episode_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload) + "\n")

            episode += 1
            obs = env.reset()

    env.close()

    return {
        "episodes": episode,
        "battles": battles,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "rewards": total_rewards,
        "turns": total_turns,
        "sanitized_action_count": sanitized_count,
        "repaired_action_count": repaired_count,
    }


def _ensure_server(settings):
    server_cfg = _resolve_server_configuration(settings.get("server_url"))
    if server_cfg is None:
        raise ValueError("could not resolve server configuration")
    _ensure_server_available(server_cfg)
    return server_cfg


def _resolve_eval_context(settings, team_text: str):
    server_cfg = _ensure_server(settings)
    act_size = action_space_size(settings.get("battle_format", "gen9doublesou"))
    return server_cfg, act_size, team_text


__all__ = [
    "_collect_episode",
    "_episode_outcome",
    "_episode_turns",
    "_ensure_server",
    "_resolve_eval_context",
]
