from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

from src.config import section
from src.offline import train_offline


def _grid_values(values):
    if isinstance(values, list | tuple):
        return list(values)
    return [values]


def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def _safe_stem(value):
    token = str(value)
    return token.replace("/", "-").replace(" ", "_").replace(".", "_")


def _unique_paths(base_path, suffix):
    base_path = Path(base_path)
    return base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")


def _run_single(base_settings, overrides):
    settings = dict(base_settings)
    settings.update(overrides)

    suffix = "_".join(f"{key}-{_safe_stem(value)}" for key, value in overrides.items()) or "base"

    base_policy = settings.get("policy_path") or "outputs/models/bc_policy.pt"
    settings["policy_path"] = str(_unique_paths(base_policy, suffix))
    for key in ("stats_path", "best_policy_path", "best_stats_path"):
        base_path = settings.get(key)
        if base_path:
            settings[key] = str(_unique_paths(base_path, suffix))
    tb_dir = settings.get("tensorboard_dir")
    if tb_dir:
        settings["tensorboard_dir"] = str(Path(tb_dir) / suffix)

    print(f"[offline-grid] overrides={overrides}")
    metrics = train_offline(settings)
    final_metric = metrics[-1] if metrics else {}
    return {
        "overrides": overrides,
        "metrics": final_metric,
        "paths": {
            "policy_path": settings.get("policy_path"),
            "stats_path": settings.get("stats_path"),
            "tensorboard_dir": settings.get("tensorboard_dir"),
        },
    }


def _build_parser():
    parser = argparse.ArgumentParser(description="Grid search for offline BC training")
    parser.add_argument("--limit", type=int, default=None, help="maximum number of runs")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional path to save sweep results (JSON)",
    )
    parser.add_argument(
        "--offline",
        type=str,
        default=None,
        help="JSON dict to override offline base settings (e.g. '{\"lr\":1e-4}')",
    )
    parser.add_argument(
        "--sweep",
        type=str,
        default=None,
        help="JSON dict to override offline-sweeps config (e.g. '{\"lr\":[1e-4,1e-5]}')",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.offline:
        base_settings = json.loads(args.offline)
        print("[offline-grid using CLI base offline settings override]")
    else:
        base_settings = dict(section("offline"))

    if args.sweep:
        sweep_config = json.loads(args.sweep)
        print("[offline-grid using CLI sweep config override]")
    else:
        sweep_config = section("offline-sweeps")

    if not sweep_config:
        print("[offline-grid] no offline_sweeps configured")
        return

    keys = list(sweep_config.keys())
    values = [_grid_values(sweep_config[key]) for key in keys]

    results = []
    combinations = itertools.product(*values)
    for index, combo in enumerate(combinations, 1):
        overrides = dict(zip(keys, combo, strict=True))
        results.append(_run_single(base_settings, overrides))
        if args.limit and index >= args.limit:
            break

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[offline-grid] results saved to {args.output}")
    else:
        root = Path(__file__).resolve().parents[3]
        default_path = root / "outputs" / "sweeps" / "offline" / f"grid_{_timestamp()}.json"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        default_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[offline-grid] results saved to {default_path}")


__all__ = ["main"]
