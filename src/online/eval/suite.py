from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from poke_env.player.baselines import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import section
from src.online.env import make_maskable_env
from src.online.policy.load import load_maskable_policy
from src.utils.teambuilders import read_showdown_team


def _load_policy(path, device="cpu"):
    return load_maskable_policy(path, device=device)


def _make_env(opponent_cls, battle_format, team_text, server_url, opponent_team=None):
    from poke_env import AccountConfiguration, LocalhostServerConfiguration, ServerConfiguration

    url = server_url or "http://localhost:8000"
    cfg = (
        LocalhostServerConfiguration
        if url in {"http://localhost:8000", "local", "localhost"}
        else ServerConfiguration(url.replace("http", "ws"), "")
    )
    opp_acct = AccountConfiguration(f"EvalOpp{random.randint(0, 999999)}", None)
    opp_kwargs = {
        "account_configuration": opp_acct,
        "battle_format": battle_format,
        "max_concurrent_battles": 1,
        "server_configuration": cfg,
    }
    if opponent_team:
        opp_kwargs["team"] = opponent_team
    opponent = opponent_cls(**opp_kwargs)

    player_acct = AccountConfiguration(f"EvalOur{random.randint(0, 999999)}", None)
    env = make_maskable_env(
        opponent=opponent,
        battle_format=battle_format,
        rewards={},
        team=team_text,
        account_configuration1=player_acct,
        server_configuration=cfg,
        step_delay=0.0,
        console_log_mode="off",
        console_log_interval_sec=5.0,
    )
    return DummyVecEnv([lambda: env])


def _eval(model, env, episodes):
    wins = losses = draws = 0
    results = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        info = {}
        ep_reward = 0.0
        while not done:
            mask = env.envs[0].action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, done, info = env.step(action)
            done = bool(done)
            ep_reward += float(reward)
        info = info[0] if isinstance(info, list | tuple) else info
        result = info.get("battle_stats", {}).get("result")
        if result == "win":
            wins += 1
        elif result == "loss":
            losses += 1
        else:
            draws += 1
        results.append(
            {
                "episode": ep,
                "result": result,
                "reward": ep_reward,
                "turn": info.get("battle_stats", {}).get("turn"),
                "repair": info.get("repaired_action", False),
            }
        )
    return {"wins": wins, "losses": losses, "draws": draws, "results": results}


def _build_parser():
    parser = argparse.ArgumentParser(description="Run standard eval suite for a saved policy.")
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--server-url", type=str, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = section("evaluation") or {}
    battle_format = cfg.get("battle_format", "gen9doublesou")
    team_path = Path(cfg.get("our_team_path", "teams/gen9dou_fixed.txt"))
    team_text = read_showdown_team(team_path)
    server_url = args.server_url or cfg.get("server_url", "http://localhost:8000")
    episodes = max(args.episodes, 1)

    model = _load_policy(args.policy)

    opponents = {
        "simple": SimpleHeuristicsPlayer,
        "maxbp": MaxBasePowerPlayer,
        "random": RandomPlayer,
    }

    summary: dict[str, Any] = {}
    all_results = []

    for name, cls in opponents.items():
        env = _make_env(cls, battle_format, team_text, server_url)
        res = _eval(model, env, episodes)
        summary[name] = {k: v for k, v in res.items() if k != "results"}
        for r in res["results"]:
            r["opponent"] = name
            all_results.append(r)
        env.close()

    env = _make_env(SimpleHeuristicsPlayer, battle_format, team_text, server_url)
    res = _eval(model, env, episodes)
    summary["mirror"] = {k: v for k, v in res.items() if k != "results"}
    for r in res["results"]:
        r["opponent"] = "mirror"
        all_results.append(r)
    env.close()

    ts = int(time.time())
    out_dir = Path("outputs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"eval_suite_{ts}.jsonl"
    summary_path = out_dir / f"eval_suite_{ts}_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for row in all_results:
            jf.write(json.dumps(row) + "\n")
    summary["policy"] = str(args.policy)
    summary["episodes_per_opponent"] = episodes
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[eval_suite] wrote {jsonl_path} and {summary_path}")


__all__ = ["main"]
