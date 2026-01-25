#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.offline.train import cli as _impl  # noqa: E402


def main():
    _impl.main()


if __name__ == "__main__":
    main()
