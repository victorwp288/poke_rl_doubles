from __future__ import annotations

import argparse
import sys

from src.online.runner import run_online_training

from . import batch as batch_cli
from . import grid as grid_cli

SUBCOMMANDS = {"train", "grid", "batch"}


def _build_train_parser():
    parser = argparse.ArgumentParser(description="Run online PPO training.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="scratch",
        help="Which training mode from config.online.modes to run (default: scratch)",
    )
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        help="Override configuration settings in key=value format. "
        "For example: --override total_timesteps=5000000 --override learning_rate=0.0001",
    )
    return parser


def _select_command(argv: list[str]) -> tuple[str, list[str]]:
    if not argv or argv[0].startswith("-"):
        return "train", argv
    if argv[0] in SUBCOMMANDS:
        return argv[0], argv[1:]
    return "train", argv


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd, sub_argv = _select_command(argv)

    if cmd == "grid":
        grid_cli.main(sub_argv)
        return
    if cmd == "batch":
        batch_cli.main(sub_argv)
        return

    parser = _build_train_parser()
    args = parser.parse_args(sub_argv)
    run_online_training(args)


__all__ = ["main"]
