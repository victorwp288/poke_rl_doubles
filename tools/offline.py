#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import section  # noqa: E402
from src.offline import train_offline  # noqa: E402


def run():
    settings = section("offline")
    print("starting offline training")
    train_offline(settings)


def main():
    run()


if __name__ == "__main__":
    main()
