import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from src.config import section
from src.online.env import make_maskable_env
from src.online.policy.head import configure_action_head
from src.online.policy.warmstart import load_behavior_clone_weights
from src.utils.teambuilders import read_showdown_team

from .config import build_arg_parser, load_defaults, merge_cli_overrides
from .env import _build_opponent, _policy_kwargs, _resolve_server_configuration
from .evaluate import _evaluate


def main(argv: list[str] | None = None):
    defaults = load_defaults()
    parser = build_arg_parser(defaults)
    args = parser.parse_args(argv)
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
    summary: dict[str, Any] = {
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


__all__ = ["main"]
