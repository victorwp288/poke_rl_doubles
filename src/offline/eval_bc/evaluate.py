import time

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from src.online.env import make_maskable_env

from .actions import _ensure_mask, _sanitize_action
from .env import _build_opponent
from .repair_step import _step_with_repair

DEFAULT_WATCHDOG_SECONDS = 45.0


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


__all__ = ["DEFAULT_WATCHDOG_SECONDS", "_evaluate"]
