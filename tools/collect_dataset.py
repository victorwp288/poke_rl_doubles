#!/usr/bin/env python3
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import fetch as fetch_cli  # noqa: E402
from src.data import parse as parse_cli  # noqa: E402
from src.offline.collect import batch as batch_cli  # noqa: E402
from src.offline.collect import cli as collect_cli  # noqa: E402
from src.offline.collect import merge as merge_cli  # noqa: E402
from src.offline.collect import purge as purge_cli  # noqa: E402

SUBCOMMANDS = {"collect", "batch", "merge", "purge", "fetch", "parse"}


def _select_command(argv):
    if not argv:
        return "collect", []
    if argv[0] in SUBCOMMANDS:
        return argv[0], argv[1:]
    return "collect", argv


def main():
    cmd, sub_argv = _select_command(sys.argv[1:])

    if cmd == "batch":
        asyncio.run(batch_cli.main(sub_argv))
    elif cmd == "merge":
        merge_cli.main(sub_argv)
    elif cmd == "purge":
        purge_cli.main(sub_argv)
    elif cmd == "fetch":
        fetch_cli.main(sub_argv)
    elif cmd == "parse":
        parse_cli.main(sub_argv)
    else:
        collect_cli.main(sub_argv)


if __name__ == "__main__":
    main()
