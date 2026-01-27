"""
Core action-space utilities.

High-signal contract:
- Policies operate on a per-slot discrete vocabulary of size `act_size = action_space_size(fmt)`.
- Doubles actions are represented as a 2-vector `[slot0_action, slot1_action]`.
- Legality is enforced via action masks + sanitize/repair in the online env (see `src/online/env.py`).

Related docs/tests:
- docs/codebase_overview.md (action representation)
- docs/ARCHITECTURE.md (mask/obs contracts)
"""

from poke_env.battle import AbstractBattle
from poke_env.player import Player


def action_space_size(battle_format):
    # NOTE: This is an abstract sizing formula for the discrete action vocabulary, not a guarantee
    # that every action is legal at a given turn. Per-turn legality is enforced by masks:
    # - `src/core/action_mask.py` (mask layout: [slot0 | slot1])
    # - `src/online/env_mask_*` (sanitize -> repair -> fallback)
    fmt = battle_format.lower()
    gimmicks_map = {"gen9": 4, "gen8": 3, "gen7": 2, "gen6": 1}
    gimmicks = next((count for prefix, count in gimmicks_map.items() if fmt.startswith(prefix)), 0)
    return 1 + 6 + 4 * 5 * (gimmicks + 1)


__all__ = ["action_space_size", "AbstractBattle", "Player"]
