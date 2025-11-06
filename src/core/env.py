from poke_env.battle import AbstractBattle
from poke_env.player import Player


def action_space_size(battle_format):
    fmt = battle_format.lower()
    gimmicks_map = {"gen9": 4, "gen8": 3, "gen7": 2, "gen6": 1}
    gimmicks = next((count for prefix, count in gimmicks_map.items() if fmt.startswith(prefix)), 0)
    return 1 + 6 + 4 * 5 * (gimmicks + 1)


__all__ = ["action_space_size", "AbstractBattle", "Player"]
