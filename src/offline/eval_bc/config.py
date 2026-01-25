import argparse
from pathlib import Path

from src.config import section


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


__all__ = ["build_arg_parser", "load_defaults", "merge_cli_overrides"]
