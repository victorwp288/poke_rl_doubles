from types import SimpleNamespace

from tests.helpers_battle import DummyMove


def build_mask_battle():
    move_1 = DummyMove("tackle", current_pp=35, max_pp=35)
    move_2 = DummyMove("protect", current_pp=10, max_pp=10)
    switch_a = SimpleNamespace(species="AllyA", base_species="AllyA", fainted=False)
    switch_b = SimpleNamespace(species="AllyB", base_species="AllyB", fainted=False)
    battle = SimpleNamespace(
        turn=3,
        force_switch=[False, False],
        trapped=[False, False],
        can_mega_evolve=[False, False],
        can_z_move=[False, False],
        can_dynamax=[False, False],
        can_tera=[False, False],
        available_moves=[[move_1], [move_2]],
        available_switches=[[switch_a, switch_b], [switch_a]],
    )
    return battle


__all__ = ["build_mask_battle"]
