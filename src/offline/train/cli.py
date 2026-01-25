from __future__ import annotations

import sys

from src.offline.eval_bc import cli as eval_bc_cli

from . import grid as grid_cli
from . import sweep as sweep_cli
from . import train as train_cli

SUBCOMMANDS = {"train", "grid", "sweep", "eval-bc"}


def _select_command(argv: list[str]) -> tuple[str, list[str]]:
    if not argv or argv[0].startswith("-"):
        return "train", argv
    if argv[0] in SUBCOMMANDS:
        return argv[0], argv[1:]
    return "train", argv


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd, sub_argv = _select_command(argv)

    if cmd == "grid":
        grid_cli.main(sub_argv)
    elif cmd == "sweep":
        sweep_cli.main(sub_argv)
    elif cmd == "eval-bc":
        eval_bc_cli.main(sub_argv)
    else:
        train_cli.main(sub_argv)


__all__ = ["main"]
