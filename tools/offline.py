#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.offline import OfflineConfig, train_offline  # noqa: E402


def main() -> None:
    config = OfflineConfig()
    print("starting offline training:", config)
    train_offline(config)


if __name__ == "__main__":
    main()
