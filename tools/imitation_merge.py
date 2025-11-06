#!/usr/bin/env python3
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


def main():
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
