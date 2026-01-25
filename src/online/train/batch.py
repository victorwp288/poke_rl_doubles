from __future__ import annotations

import argparse

from src.config import section
from src.online.runner import run


def _build_parser():
    parser = argparse.ArgumentParser(description="Run PPO experiments for one or more modes.")
    parser.add_argument(
        "modes",
        nargs="*",
        help="List of modes to run (default: from config ppo_runs, or ['warmstart','scratch'])",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    configured = section("ppo_runs") or []
    requested = args.modes or []

    order = requested or configured
    if not order:
        order = ["warmstart", "scratch"]

    for mode in order:
        print(f"[ppo] starting {mode}", flush=True)
        try:
            run(mode)
        except Exception as exc:  # noqa: BLE001
            print(f"[ppo] ERROR in {mode}: {exc}", flush=True)
            continue
        print(f"[ppo] finished {mode}", flush=True)


__all__ = ["main"]
