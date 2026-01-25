import argparse
from pathlib import Path

from .config import load_settings
from .runner import collect_imitation


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Collect imitation tuples from heuristic self-play in Gen 9 Doubles."
    )

    parser.add_argument("--n-battles", type=int)
    parser.add_argument("--server-url", type=str)
    parser.add_argument("--battle-format", type=str)
    parser.add_argument("--our-team-path", type=str)
    parser.add_argument("--opponent-teams-dir", type=str)
    parser.add_argument("--teacher-kind", type=str)
    parser.add_argument("--teacher-path", type=str)
    parser.add_argument("--opponents", nargs="*")
    parser.add_argument("--out-path", type=str)
    parser.add_argument("--battle-timeout", type=float)
    parser.add_argument("--rotate-every", type=int)
    parser.add_argument("--max-concurrent-battles", type=int)
    parser.add_argument("--opponent-concurrency", type=int)
    parser.add_argument("--seed", type=int)

    args = parser.parse_args(argv)

    base = load_settings()

    if args.n_battles is not None:
        base.n_battles = args.n_battles
    if args.server_url is not None:
        base.server_url = args.server_url
    if args.battle_format is not None:
        base.battle_format = args.battle_format
    if args.our_team_path is not None:
        base.our_team_path = Path(args.our_team_path)
    if args.opponent_teams_dir is not None:
        base.opponent_teams_dir = Path(args.opponent_teams_dir)
    if args.teacher_kind is not None:
        base.teacher_kind = args.teacher_kind
    if args.teacher_path is not None:
        base.teacher_path = Path(args.teacher_path)
    if args.opponents is not None:
        base.opponents_kinds = args.opponents
    if args.out_path is not None:
        base.out_path = Path(args.out_path)
    if args.battle_timeout is not None:
        base.battle_timeout = args.battle_timeout
    if args.rotate_every is not None:
        base.rotate_every = args.rotate_every
    if args.max_concurrent_battles is not None:
        base.max_concurrent_battles = args.max_concurrent_battles
    if args.opponent_concurrency is not None:
        base.opponent_concurrency = args.opponent_concurrency
    if args.seed is not None:
        base.seed = args.seed

    collect_imitation(base)


__all__ = ["main"]
