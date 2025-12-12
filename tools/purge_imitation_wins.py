#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def iter_records(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main():
    parser = argparse.ArgumentParser(description="Copy win-only records from imitation JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/imitation.jsonl"),
        help="Source imitation JSONL (not modified).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/imitation_wins.jsonl"),
        help="Destination JSONL containing only win records.",
    )
    parser.add_argument(
        "--include-draws",
        action="store_true",
        help="Also keep draws in addition to wins.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # First pass: collect battle_tags that are wins (and optionally draws) from summary lines.
    allowed_tags: set[str] = set()
    summaries = 0
    for record in iter_records(args.input):
        tag = record.get("battle_tag")
        if not tag:
            continue
        result = record.get("result") or record.get("stats", {}).get("result")
        if result in ("win",) or (args.include_draws and result == "draw"):
            allowed_tags.add(tag)
        if result is not None:
            summaries += 1

    if not allowed_tags:
        print(
            f"[warn] found {summaries} summaries but no winning/draw battle_tags; nothing written."
        )
        return

    kept = 0
    total = 0
    with args.output.open("w", encoding="utf-8") as out:
        for record in iter_records(args.input):
            tag = record.get("battle_tag")
            if tag and tag in allowed_tags:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            total += 1

    print(
        f"wrote {kept} records (from {total}) to {args.output} "
        f"using {len(allowed_tags)} winning battle_tags"
    )


if __name__ == "__main__":
    main()
