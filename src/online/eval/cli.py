from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.offline.eval_bc import cli as bc_cli

from . import compare as compare_cli
from . import ppo_simple as ppo_cli
from . import suite as suite_cli
from .legacy import run_legacy_eval
from .parse import _parse_overrides_from_args, _split_policy_args
from .settings import _resolve_settings

SUBCOMMANDS = {"legacy", "compare", "suite", "ppo", "bc"}


def _build_args_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO policies against baseline opponents."
    )
    parser.add_argument(
        "--policy",
        type=str,
        action="append",
        required=False,
        default=[],
        help="Policy checkpoint in label=path form. Can be repeated.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Episodes to play per policy/opponent pair.",
    )
    parser.add_argument(
        "--env-mode",
        type=str,
        default="scratch",
        help="Config online.modes entry to use for environment settings.",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="*",
        default=[],
        help="Baseline opponents (simple|maxbp|random).",
    )
    parser.add_argument(
        "--crossplay",
        action="store_true",
        help="Also evaluate each policy against every other policy.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Also evaluate each policy against itself as opponent.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render matches in the console (slow).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep between steps (seconds).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--replays-dir",
        type=str,
        default=None,
        help="Optional directory to save replay URLs.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default=None,
        help="Optional notes to include in run_config.json.",
    )
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        default=[],
        help="Override settings, supports rewards.* keys. Format key=value.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Recompute summaries for an existing eval folder.",
    )
    parser.add_argument(
        "--summary-policy",
        type=str,
        action="append",
        default=[],
        help="Policy label=path for summary-only mode.",
    )
    parser.add_argument(
        "--summary-opponent",
        type=str,
        action="append",
        default=[],
        help="Opponent label=kind for summary-only mode.",
    )
    parser.add_argument(
        "--summary-overrides",
        type=str,
        action="append",
        default=[],
        help="Overrides for summary-only mode (key=value or rewards.k=v).",
    )
    return parser


def _parse_args(argv: list[str] | None = None):
    parser = _build_args_parser()
    return parser.parse_args(argv)


def _select_command(argv: list[str]) -> tuple[str, list[str]]:
    if not argv or argv[0].startswith("-"):
        return "legacy", argv
    if argv[0] in SUBCOMMANDS:
        return argv[0], argv[1:]
    return "legacy", argv


def _run_legacy(argv: list[str] | None, root: Path) -> None:
    args = _parse_args(argv)
    if not args.policy and not args.summary_only:
        parser = _build_args_parser()
        parser.print_help()
        print("\nExamples:\n")
        print(
            "  python tools/eval_models.py \\\n"
            "    --policy scratch=outputs/models/maskable_ppo_scratch_best.zip \\\n"
            "    --policy warmstart=outputs/models/maskable_ppo_warmstart_best.zip \\\n"
            "    --episodes 200 --opponents simple maxbp random\n"
        )
        return

    overrides = _parse_overrides_from_args(args)
    settings = _resolve_settings(args.env_mode, overrides or None)
    policies = _split_policy_args(root, args)
    run_legacy_eval(
        root=root,
        args=args,
        policies=policies,
        env_mode=args.env_mode,
        settings=settings,
        overrides=overrides,
    )


def main(argv: list[str] | None = None, *, root: Path | None = None):
    root = root or Path(__file__).resolve().parents[3]
    argv_list = list(sys.argv[1:] if argv is None else argv)
    cmd, sub_argv = _select_command(argv_list)

    if cmd == "compare":
        compare_cli.main(sub_argv)
        return
    if cmd == "suite":
        suite_cli.main(sub_argv)
        return
    if cmd == "ppo":
        ppo_cli.main(sub_argv)
        return
    if cmd == "bc":
        bc_cli.main(sub_argv)
        return

    _run_legacy(sub_argv, root)


__all__ = ["main"]
