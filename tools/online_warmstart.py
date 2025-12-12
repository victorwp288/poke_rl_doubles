#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.online import parse_override_pairs, run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run online training with warmstart")
    parser.add_argument(
        "--mode",
        type=str,
        default="warmstart",
        help="Override the default 'warmstart' mode passed to tools.online.run()",
    )
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        help="Override configuration settings in key=value format. "
        "For example: --override parallel_battles=16 --override console_log_mode='off'",
    )

    args = parser.parse_args()

    overrides = parse_override_pairs(args.override)
    run(args.mode, overrides or None)


if __name__ == "__main__":
    main()
