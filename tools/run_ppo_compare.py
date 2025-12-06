#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.online import run  # noqa: E402

from src.config import section  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run PPO experiments for one or more modes.")
    parser.add_argument(
        "modes",
        nargs="*",
        help="List of modes to run (default: from config ppo_runs, or ['warmstart','scratch'])",
    )
    parser.parse_args()

    configured = section("ppo_runs") or []
    requested = sys.argv[1:]

    order = requested or configured
    if not order:
        order = ["warmstart", "scratch"]

    for mode in order:
        print(f"[ppo] starting {mode}", flush=True)
        try:
            run(mode)
        except Exception as e:
            print(f"[ppo] ERROR in {mode}: {e}", flush=True)
            continue
        print(f"[ppo] finished {mode}", flush=True)


if __name__ == "__main__":
    main()
