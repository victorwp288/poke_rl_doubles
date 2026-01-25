from __future__ import annotations

# CLI wrapper that maps offline config/overrides into train_offline().
import argparse

from src.config import section
from src.offline import train_offline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline BC training")
    parser.add_argument("--dataset-path", type=str, help="Override offline.dataset_path")
    parser.add_argument("--epochs", type=int, help="Override offline.epochs")
    parser.add_argument("--device", type=str, help="Override offline.device")
    parser.add_argument("--num-workers", type=int, help="Override offline.num_workers")
    parser.add_argument("--batch-size", type=int, help="Override offline.batch_size")
    parser.add_argument("--learning-rate", type=float, help="Override offline.learning_rate")
    return parser


def _apply_cli_overrides(base, args):
    settings = dict(base)
    if args.dataset_path is not None:
        settings["dataset_path"] = args.dataset_path
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    if args.device is not None:
        settings["device"] = args.device
    if args.num_workers is not None:
        settings["num_workers"] = args.num_workers
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        settings["learning_rate"] = args.learning_rate
    return settings


def run(settings=None):
    if settings is None:
        settings = section("offline")

    print("starting offline training")
    train_offline(settings)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = section("offline")
    settings = _apply_cli_overrides(settings, args)
    run(settings)


__all__ = ["main", "run"]
