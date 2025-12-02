#!/usr/bin/env python 3
"""
CLI for fetching Pokémon Showdown replay data.
Wraps 'tools/data_fetch.py' with argparse interface, allowing CLI configuration of parameters under
the dataclass 'Settings'.
If no flags are provided, it falls back to default or the optional 'tools/data_fetch_config.json' file.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data_fetch import (
    Settings,
    run,
    default_settings,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Pokemon Showdown replays using custom or default settings"
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Directory to save download replays (default from config).",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="Path to a file containing replay IDs (default from config)",
    )
    parser.add_argument(
        "--urls-file",
        type=Path,
        help="Path to a file containing replay URLs (default from config)",
    )
    parser.add_argument(
        "--ids",
        type=str,
        help="Replay IDs to fetch (space-seperated).",
    )
    parser.add_argument(
        "--user",
        type=str,
        help="Showdown username to fetch recent replays for.",
    )
    parser.add_argument(
        "--format",
        default=None,
        type=str,
        help="Battle format to fetch (e.g., gen9doublesou).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of replays to fetch per user (defualt 200)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=None,
        help="Requests per second (e.g., 0.5 for one request every 2s)",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default=None,
        help="Custom User-Agent string for HTTP requests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_false",
        help="Overwrite existing replay files if present.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    base = default_settings()

    #Merge command-line overrides with default settings
    settings = Settings(
        out_dir=args.out_dir or base.out_dir,
        ids_file=args.ids_file or base.ids_file,
        urls_file=args.urls_file or base.urls_file,
        ids=args.ids or base.ids,
        user=args.user or base.user,
        fmt=args.format or base.fmt,
        limit=args.limit or base.limit,
        rate=args.rate or base.rate,
        user_agent=args.user_agent or base.user_agent,
        overwrite=args.overwrite or base.overwrite,
    )     

    print("Data fetch configuration:")
    for field, value in vars (settings).items():
        print(f" {field}: {value}")

    run(settings)

if __name__ == "__main__":
    main()

"""
To run this script:

python tools/cli_data-fetch.py
python tools/cli_data-fetch.py --user someuser --limit 50
python tools/cli_data-fetch.py --out-dir data/raw/custom --rate 0.2
python tools/cli_data-fetch.py --overwrite
"""    