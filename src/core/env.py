# Small helpers shared by scripts when talking to poke-env.

from __future__ import annotations

from poke_env.battle import AbstractBattle
from poke_env.player import Player


def action_space_size(battle_format: str) -> int:
    fmt = battle_format.lower()
    gimmick_by_gen = {"gen6": 1, "gen7": 2, "gen8": 3, "gen9": 4}
    gimmicks = next(
        (count for prefix, count in gimmick_by_gen.items() if fmt.startswith(prefix)), 0
    )
    return 1 + 6 + 4 * 5 * (gimmicks + 1)


def make_env(player_cls: type[Player], **kwargs) -> Player:
    return player_cls(**kwargs)


__all__ = ["action_space_size", "make_env", "AbstractBattle", "Player"]
