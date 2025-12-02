#!/usr/bin/env python3
"""
CLI for parsing Pokémon Showdown replay logs into tactical hint events. 
It extracts events such as 'protect', 'tailwind', and 'switch', 
from downloaded replay logs and saves them to a JSONL file.

If no flags are provided, the defaults from `data_parse.py` are used:
    raw_dir = data/raw/downloaded
    out_path = data/processed/human_hints.jsonl
    focus_side = p1
"""

from __future__ import annotations
import sys
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data_parse import iter_replay_files, parse_replay

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse downloaded Pokémon Showdown replays into tactical hints."
    )

    parser.add_argument(
        "--raw dir",
        type=Path,
        default=Path("data/raw/downloaded"),
        help="Directory containing replay files to parse (default: data/raw/downloaded).",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("data/processed/human_hints.json"),
        help="Output JSONL file for parsed hint events (default: data/processed/human_hints.jsonl).",
    )
    parser.add_argument(
        "--focus-side",
        type=str,
        choices=["p1", "p2", "none"],
        default=Path("p1"),
        help="Which side to focus on (p1, p2, or none for both). Default: p1.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    raw_dir = args.raw
    out_path = args.out_path
    focus_side = None if args.focus_side == "none" else args.focus_side

    raw_dir.mkdir(parents = True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_files = 0
    n_events = 0

    with out_path.open("W", encoding="utf-8") as handle:
        for replay_path in iter_replay_files(raw_dir):
            try:
                events = parse_replay(replay_path)
            except Exception:
                continue
            if not events:
                continue
            for events in events:
                if focus_side is not None and events.get("side") != focus_side:
                    continue
                handle.write(json.dumps(events) + "\n")
                n_events += 1
            n_files += 1

    print(f"Parsed {n_events} events from {n_files} files -> {out_path}")        

if __name__ == "__main__":
    main()

"""
To run this script write in terminal:
    python tools/cli_data-parse.py
    python tools/cli_data-parse.py --focus-side p2
    python tools/cli_data-parse.py --raw-dir data/raw/custom --out-path outputs/hints.jsonl
    python tools/cli_data-parse.py --focus-side none
"""