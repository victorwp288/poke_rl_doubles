from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from poke_env import AccountConfiguration
from poke_env.player.baselines import SimpleHeuristicsPlayer
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import section
from src.online.env import make_maskable_env
from src.online.environment import _resolve_server_configuration
from src.online.kl_ppo import KLRegularizedMaskablePPO
from src.online.opponents import _unique_username
from src.online.policy.head import configure_action_head
from src.utils.teambuilders import read_showdown_team


def _build_env(team_text, battle_format, server_cfg):
    def _factory():
        opponent_account = AccountConfiguration(_unique_username("EvalOpp"), None)
        opponent = SimpleHeuristicsPlayer(
            account_configuration=opponent_account,
            battle_format=battle_format,
            max_concurrent_battles=1,
            server_configuration=server_cfg,
        )
        player_account = AccountConfiguration(_unique_username("EvalPPO"), None)
        return make_maskable_env(
            opponent=opponent,
            battle_format=battle_format,
            team=team_text,
            account_configuration1=player_account,
            server_configuration=server_cfg,
        )

    return DummyVecEnv([_factory])


def _evaluate(model, env, episodes: int):
    wins = losses = draws = 0
    rewards = []
    episode = 0
    obs = env.reset()
    ep_reward = 0.0
    while episode < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        ep_reward += float(reward[0])
        if done[0]:
            stats = info[0].get("battle_stats") if info else None
            result = stats.get("result") if isinstance(stats, dict) else None
            if result == "win":
                wins += 1
            elif result == "loss":
                losses += 1
            elif result == "draw":
                draws += 1
            rewards.append(ep_reward)
            episode += 1
            ep_reward = 0.0
            obs = env.reset()
    return {
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "rewards": rewards,
    }


def _build_parser():
    parser = argparse.ArgumentParser(description="Evaluate PPO policy against simple heuristics.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/models/maskable_ppo_warmstart_best.zip",
        help="Path to PPO checkpoint (.zip).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--battle_format",
        type=str,
        default=None,
        help="Override battle format (defaults to config.online.battle_format).",
    )
    parser.add_argument(
        "--team_path",
        type=str,
        default=None,
        help="Override team file (defaults to config.online.team_path).",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default=None,
        help="Override server URL (defaults to config.online.server_url).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    online_cfg = section("online") or {}
    battle_format = args.battle_format or online_cfg.get("battle_format", "gen9doublesou")
    team_path = args.team_path or online_cfg.get("team_path")
    if not team_path:
        raise ValueError("team_path is required (set config.online.team_path or --team_path)")
    team_text = read_showdown_team(Path(team_path))
    server_cfg = _resolve_server_configuration(args.server_url or online_cfg.get("server_url"))

    env = _build_env(team_text, battle_format, server_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hidden_dim = int(online_cfg.get("policy_hidden_dim", 1536))
    hidden_layers = int(online_cfg.get("policy_hidden_layers", 6))
    head_dim = int(online_cfg.get("policy_head_mlp_dim", 512))
    policy_kwargs = {
        "net_arch": {
            "pi": [hidden_dim] * hidden_layers,
            "vf": [hidden_dim] * hidden_layers,
        },
        "activation_fn": torch.nn.ReLU,
        "ortho_init": False,
    }

    model = KLRegularizedMaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        policy_kwargs=policy_kwargs,
        tensorboard_log=None,
        verbose=0,
        device=device,
    )
    configure_action_head(model.policy, head_dim)
    model.set_parameters(args.checkpoint, exact_match=False, device=device)

    result = _evaluate(model, env, args.episodes)
    win_rate = result["wins"] / result["episodes"] if result["episodes"] else 0.0
    print(
        f"[eval] episodes={result['episodes']} win={result['wins']} "
        f"loss={result['losses']} draw={result['draws']} "
        f"reward_mean={result['mean_reward']:.3f} win_rate={win_rate:.3f}",
        flush=True,
    )
    summary_path = Path("outputs/eval/ppo_eval_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    env.close()


__all__ = ["main"]
