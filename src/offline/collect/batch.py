import argparse
import asyncio
from pathlib import Path

from src.config import section

from .config import Settings
from .runner import play_dataset


def _merge_settings(base, overrides):
    merged = dict(base)
    for key, value in (overrides or {}).items():
        merged[key] = value
    return merged


def _build_settings(base, batch_cfg, replica_idx):
    merged = _merge_settings(base, batch_cfg.get("settings"))
    if "opponents_kinds" not in merged and "opponents" in merged:
        merged["opponents_kinds"] = list(merged["opponents"])
    if "our_team_path" in merged:
        merged["our_team_path"] = Path(merged["our_team_path"])
    if "opponent_teams_dir" in merged:
        merged["opponent_teams_dir"] = Path(merged["opponent_teams_dir"])
    if "teacher_path" in merged and merged["teacher_path"]:
        merged["teacher_path"] = Path(merged["teacher_path"])
    prefix = batch_cfg.get("out_path_prefix")
    if not prefix:
        raise ValueError(f"batch {batch_cfg} missing out_path_prefix")
    out_path = Path(f"{prefix}.{replica_idx:03d}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seed_start = int(batch_cfg.get("seed_start", 0))
    seed = int(merged.get("seed", seed_start + replica_idx))
    merged["seed"] = seed
    merged["out_path"] = out_path
    return Settings(**merged)


async def _run_batch(name, batch_cfg, base_settings):
    replicas = int(batch_cfg.get("replicas", 1))
    concurrency = max(1, int(batch_cfg.get("concurrency", replicas)))
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def _runner(replica_idx):
        async with sem:
            settings = _build_settings(base_settings, batch_cfg, replica_idx)
            print(
                f"[batch] {name} replica {replica_idx + 1}/{replicas} "
                f"-> {settings.out_path} seed={settings.seed}",
                flush=True,
            )
            await play_dataset(settings)

    for idx in range(replicas):
        results.append(asyncio.create_task(_runner(idx)))
    for task in results:
        await task


def build_arg_parser(defaults_batches):
    parser = argparse.ArgumentParser(description="Run imitation learning data-collection batches")

    parser.add_argument(
        "--batches",
        nargs="*",
        help="Names of batches to run (defaults.yaml: imitation_batches). "
        "If omitted, all batches are run.",
    )

    parser.add_argument(
        "--override-settings",
        type=str,
        default=None,
        help="JSON string of overrides applied to *every batch* (rarely needed). "
        'Example: \'{"episodes": 30, "max_turns": 50}\'',
    )

    return parser


def apply_global_overrides(base_settings, override_str):
    if not override_str:
        return base_settings

    import json

    overrides = json.loads(override_str)

    merged = base_settings.copy()
    merged.update(overrides)
    return merged


async def main(argv: list[str] | None = None):
    defaults_batches = section("imitation_batches") or {}
    base_settings = section("imitation_collect") or {}

    parser = build_arg_parser(defaults_batches)
    args = parser.parse_args(argv)

    base_settings = apply_global_overrides(base_settings, args.override_settings)

    if not defaults_batches:
        print("no imitation_batches configured in defaults.yaml")
        return

    if args.batches:
        selection = {}
        for name in args.batches:
            if name not in defaults_batches:
                raise KeyError(f"unknown batch '{name}'")
            selection[name] = defaults_batches[name]
    else:
        selection = defaults_batches

    for name, batch_cfg in selection.items():
        print(f"[batch] starting {name}", flush=True)
        await _run_batch(name, batch_cfg, base_settings)
        print(f"[batch] finished {name}", flush=True)


__all__ = ["main"]
