from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

from src.config import section
from src.online.runner import run


def _grid_values(values):
    if isinstance(values, list | tuple):
        return list(values)
    return [values]


def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def _safe_suffix(overrides):
    parts = [
        f"{key}-{str(value).replace('/', '-').replace(' ', '_')}"
        for key, value in overrides.items()
    ]
    return "_".join(parts) if parts else "base"


def _unique_path(base_path, suffix):
    base = Path(base_path)
    return base.with_name(f"{base.stem}_{suffix}{base.suffix}")


def _prepare_overrides(mode_settings, overrides):
    suffix = _safe_suffix(overrides)
    updated = dict(overrides)
    policy_path = mode_settings.get("policy_path")
    if policy_path:
        updated["policy_path"] = _unique_path(policy_path, suffix)
    best_path = mode_settings.get("best_policy_path")
    if best_path:
        updated["best_policy_path"] = _unique_path(best_path, suffix)
    tensorboard_dir = mode_settings.get("tensorboard_dir")
    if tensorboard_dir:
        updated["tensorboard_dir"] = Path(tensorboard_dir) / suffix
    checkpoint_freq = mode_settings.get("checkpoint_freq")
    if checkpoint_freq:
        updated["checkpoint_freq"] = checkpoint_freq
    return updated


def _build_parser():
    parser = argparse.ArgumentParser(description="Grid search for PPO online training")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="maximum runs per mode",
    )
    parser.add_argument(
        "modes",
        nargs="*",
        help="Overide sweep modes (default comes from config online_sweeps or online)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    online_cfg = section("online") or {}
    sweep_cfg = section("online_sweeps") or {}

    modes = (
        args.modes
        if args.modes
        else sweep_cfg.pop("modes", list((online_cfg.get("modes") or {}).keys()) or ["scratch"])
    )

    if not sweep_cfg:
        print("[online-grid] no sweep parameters configured")
        return

    keys = list(sweep_cfg.keys())
    values = [_grid_values(sweep_cfg[key]) for key in keys]

    start_timestamp = _timestamp()
    summary = []

    for mode in modes:
        mode_base = dict((online_cfg.get("modes") or {}).get(mode, {}))
        combinations = itertools.product(*values)
        for index, combo in enumerate(combinations, 1):
            overrides = dict(zip(keys, combo, strict=True))
            prepared = _prepare_overrides(mode_base, overrides)
            print(f"[online-grid] mode={mode} overrides={overrides}")
            try:
                run(mode, overrides=prepared)
                result = {"mode": mode, "overrides": overrides, "status": "ok"}
            except Exception as exc:  # noqa: BLE001
                print(f"[online-grid] ERROR mode={mode} overrides={overrides}: {exc}")
                result = {
                    "mode": mode,
                    "overrides": overrides,
                    "status": "error",
                    "error": str(exc),
                }
            summary.append(result)
            if args.limit and index >= args.limit:
                break

    root = Path(__file__).resolve().parents[3]
    output_dir = root / "outputs" / "sweeps" / "online"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"grid_{start_timestamp}.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[online-grid] results saved to {output_path}")


__all__ = ["main"]
