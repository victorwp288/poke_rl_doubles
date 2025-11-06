#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.imitation_collect import Settings, play_dataset  # noqa: E402

from src.config import section  # noqa: E402


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


async def main():
    requested = sys.argv[1:]
    batches = section("imitation_batches")
    if not batches:
        print("no imitation_batches configured in config/defaults.yaml")
        return
    base_settings = section("imitation_collect")
    selection = {}
    if requested:
        for name in requested:
            if name not in batches:
                raise KeyError(f"unknown batch '{name}'")
            selection[name] = batches[name]
    else:
        selection = batches
    for name, batch_cfg in selection.items():
        print(f"[batch] starting {name}", flush=True)
        await _run_batch(name, batch_cfg, base_settings)
        print(f"[batch] finished {name}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
