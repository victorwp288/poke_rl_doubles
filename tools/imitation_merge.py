#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import section  # noqa: E402

CHUNK = 1024 * 1024


def _resolve_source(pattern):
    path = Path(pattern)
    if path.is_file():
        return [path]
    if path.suffix:
        return sorted(path.parent.glob(path.name))
    return sorted(path.parent.glob(f"{path.name}*.jsonl"))


def merge(sources, output_path):
    files = []
    for pattern in sources:
        resolved = _resolve_source(pattern)
        if not resolved:
            print(f"[warn] no files for pattern {pattern}")
        files.extend(resolved)
    if not files:
        raise RuntimeError("no source files resolved for merge")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as dst:
        for path in files:
            print(f"[merge] {path} -> {output_path}")
            with path.open("rb") as src:
                while True:
                    chunk = src.read(CHUNK)
                    if not chunk:
                        break
                    dst.write(chunk)
    print(f"[merge] wrote {output_path} from {len(files)} files")

    _summarise_dataset(output_path)


def _summarise_dataset(dataset_path: Path):
    total_samples = 0
    total_entries = 0
    total_zeros = 0
    action_dim = None

    dataset_path = Path(dataset_path)
    stats_path = dataset_path.with_name("merge_stats.json")

    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            mask = obj.get("mask")
            if mask is None:
                continue

            flat = [x for row in mask for x in row] if isinstance(mask[0], list) else mask

            if action_dim is None:
                action_dim = len(flat)

            total_samples += 1
            total_entries += len(flat)
            total_zeros += sum(1 for x in flat if x == 0)

    zero_mask_fraction = total_zeros / total_entries if total_entries > 0 else 0.0

    summary = {
        "dataset_path": str(dataset_path),
        "total_samples": total_samples,
        "action_dim": action_dim,
        "zero_mask_fraction": zero_mask_fraction,
    }

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[merge stats] {dataset_path} | samples={total_samples} | "
        f"action_dim={action_dim} | zero_mask_fraction={zero_mask_fraction:.4f}"
    )

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Merge imitation data files according to config")

    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="List of source file patterns to merge",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output merged file",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.sources and args.output:
        merge(args.sources, args.output)
        return

    merge_cfg = section("imitation_merge") or {}
    if not merge_cfg:
        raise RuntimeError("imitation_merge block not configured")

    sources = merge_cfg.get("sources")
    if not sources:
        raise RuntimeError("imitation_merge.sources is empty")

    output_path = merge_cfg.get("output_path")
    if not output_path:
        raise RuntimeError("imitation_merge.output_path missing")

    merge(sources, output_path)


if __name__ == "__main__":
    main()
