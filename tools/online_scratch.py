#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.online import run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run online training starting from scratch")
    parser.add_argument(
        "--mode",
        type=str,
        default="scratch",
        help="Override the default 'scratch' mode passed to tools.online.run()",
    )
    parser.parse_args()

    run("scratch")


if __name__ == "__main__":
    main()
