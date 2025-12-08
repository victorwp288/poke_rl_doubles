#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# environment construction and config-override logic from tools/online.py
from tools.online import (
    _apply_overrides,
    _build_env,
    _ensure_server_available,
    _mode_settings,
    _resolve_server_configuration,
    _safe_close,
    _team_details,
    parse_override_pairs,
)

# Model loader: prefer project-specific PPO, fall back to SB3-Contrib Maskable
try:
    from src.online.kl_ppo import KLRegularizedMaskablePPO

    MODEL_CLASS = KLRegularizedMaskablePPO
except Exception:
    try:
        from sb3_contrib.common.maskable.policies import MaskablePPO

        MODEL_CLASS = MaskablePPO
    except Exception:
        MODEL_CLASS = None


# typed alias for environment info dicts
InfoDict = dict[str, Any]


# Create output directory and return paths for JSONL + CSV result files.
def _make_output_paths() -> tuple[Path, Path]:
    out_dir = Path("outputs") / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        out_dir / f"eval_{ts}.jsonl",
        out_dir / f"summary_{ts}.csv",
    )


# Extract final battle result ("win", "loss", "draw") from env info.
def _get_results_from_info(info: InfoDict | None) -> str:
    if not info:
        return "unknown"
    if isinstance(info, dict):
        bs = info.get("battle_stats")
        if isinstance(bs, dict):
            res = bs.get("result")
            if isinstance(res, str):
                return res
        top = info.get("result")
        if isinstance(top, str):
            return top
    return "unknown"


# Convert any reward returned by the environment into a float scalar.
def _float_reward(reward: Any) -> float:
    try:
        arr = np.asarray(reward)
        if arr.size > 0:
            return float(arr.reshape(-1)[0])
    except Exception:
        pass
    try:
        return float(reward)
    except Exception:
        return 0.0


# Evaluation loop
def run_eval(
    policy_path: str,
    episodes: int = 10,
    mode: str = "scratch",
    overrides: dict[str, Any] | None = None,
) -> None:
    # Make sure a loadable model class is available.
    if MODEL_CLASS is None:
        raise RuntimeError("Model class unavailable — install sb3-contrib or fix imports.")

    # Load configuration using same system as tools/online.py
    settings = _apply_overrides(_mode_settings(mode), overrides or {})

    # Force evaluation to use a single env instance.
    settings = dict(settings)
    settings["parallel_battles"] = 1

    # Load team and server configuration exactly like online training.
    team_path, team_text = _team_details(settings)
    server_cfg = _resolve_server_configuration(settings.get("server_url"))
    _ensure_server_available(server_cfg)

    eval_env = _build_env(settings, team_text=team_text, server_cfg=server_cfg)

    # Access the actual underlying MaskableDoublesEnv inside the VecEnv wrapper.
    try:
        base_env = eval_env.envs[0]
    except Exception:
        base_env = eval_env

    # Load PPO policy.
    print(f"[eval_compare] Loading policy from: {policy_path}")
    model = MODEL_CLASS.load(policy_path)

    # Prepare JSONL and CSV result files
    jsonl_path, csv_path = _make_output_paths()

    # Store per-episode results for summary and export.
    episode_results: list[dict[str, Any]] = []

    # Per-Episode Evaluation
    try:
        # Reset env and run battle to completion.
        for ep in range(int(episodes)):
            obs, info = base_env.reset()
            done = False
            total_reward = 0.0
            turns = 0

            while not done:
                # Get action mask from environment (mask illegal choices).
                try:
                    action_masks = base_env.action_masks()
                except Exception:
                    action_masks = None

                # Query policy for next action.
                try:
                    if action_masks is not None:
                        action, _ = model.predict(
                            obs, deterministic=True, action_masks=action_masks
                        )
                    else:
                        action, _ = model.predict(obs, deterministic=True)
                except TypeError:
                    # Older versions may not accept action_masks.
                    action, _ = model.predict(obs, deterministic=True)

                # Step environment (handles both Gym and Gymnasium signatures).
                out = base_env.step(action)
                if len(out) == 5:
                    obs, reward, terminated, truncated, info = out
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, info = out

                # Accumulate reward and turn count.
                total_reward += _float_reward(reward)
                turns += 1

            # Extract outcome and store result.
            outcome = _get_results_from_info(info)
            ep_data = {
                "episode": ep,
                "outcome": outcome,
                "reward": float(total_reward),
                "turns": int(turns),
                "info": info if isinstance(info, dict) else {},
            }
            episode_results.append(ep_data)

            # Append JSON line.
            with open(jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(ep_data) + "\n")

            print(
                f"[eval_compare] ep={ep:03d} outcome={outcome} reward={total_reward:.3f} turns={turns}"
            )

    finally:
        _safe_close(eval_env)

    wins = sum(r["outcome"] == "win" for r in episode_results)
    draws = sum(r["outcome"] == "draw" for r in episode_results)
    losses = sum(r["outcome"] == "loss" for r in episode_results)

    avg_reward = float(np.mean([r["reward"] for r in episode_results]))
    avg_turns = float(np.mean([r["turns"] for r in episode_results]))
    win_rate = wins / max(1, len(episode_results))

    # Summarize and write CSV
    with open(csv_path, "w", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["episodes", "wins", "losses", "draws", "win_rate", "avg_reward", "avg_turns"]
        )
        writer.writerow(
            [len(episode_results), wins, losses, draws, win_rate, avg_reward, avg_turns]
        )

    print("=== Eval Summary ===")
    print(f" Episodes: {len(episode_results)}")
    print(f" Wins:     {wins}")
    print(f" Losses:   {losses}")
    print(f" Draws:    {draws}")
    print(f" Win rate: {win_rate:.4f}")
    print(f" Avg rew:  {avg_reward:.4f}")
    print(f" Avg turn: {avg_turns:.2f}")
    print(f" JSONL:    {jsonl_path}")
    print(f" CSV:      {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved policy vs scripted bots (simple)."
    )

    parser.add_argument(
        "--policy",
        required=True,
        type=str,
        help="Path to saved policy .zip (PPO)",
    )

    parser.add_argument(
        "--episodes",
        required=True,
        type=int,
        help="Number of evaluation episodes to run",
    )

    parser.add_argument(
        "mode",
        nargs="?",
        type=str,
        default="scratch",
        help="Which mode from config.online.modes to use for env settings (default: scratch)",
    )

    parser.add_argument(
        "--override",
        type=str,
        action="append",
        help="Override configuration settings in key=value format. Example: --override parallel_battles=1",
    )

    args = parser.parse_args()
    overrides = parse_override_pairs(args.override)

    # Match args to run_eval parameters
    run_eval(policy_path=args.policy, episodes=args.episodes, mode=args.mode, overrides=overrides)


if __name__ == "__main__":
    main()
