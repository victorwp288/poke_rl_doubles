from types import SimpleNamespace

from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather


class DummyType:
    def __init__(self, name: str):
        self.name = name


class DummyMove:
    def __init__(
        self,
        move_id: str,
        current_pp: int,
        max_pp: int,
        priority: int = 0,
        disabled: bool = False,
    ):
        self.id = move_id
        self.current_pp = current_pp
        self.max_pp = max_pp
        self.priority = priority
        self.disabled = disabled


class DummyPokemon:
    def __init__(
        self,
        name: str,
        *,
        types: tuple[str, ...],
        status,
        status_counter: int,
        current_hp: int,
        max_hp: int,
        base_speed: int,
        boosts: dict[str, int],
        effects: set,
        protect_counter: int,
        must_recharge: bool,
        item: str | None,
        ability: str | None,
        moves: list[DummyMove],
        revealed: bool,
        active: bool,
        first_turn: bool,
        fainted: bool,
        available_z_moves: list[DummyMove] | None = None,
    ):
        self.base_species = name
        self.species = name
        self.types = [DummyType(t) for t in types]
        self.status = status
        self.status_counter = status_counter
        self.current_hp = current_hp
        self.max_hp = max_hp
        self.boosts = dict(boosts)
        self.effects = set(effects)
        self.protect_counter = protect_counter
        self.must_recharge = must_recharge
        self.item = item
        self._data = SimpleNamespace(UNKNOWN_ITEM="unknown")
        self.ability = ability
        self.moves = {move.id: move for move in moves}
        self.base_stats = {"spe": base_speed}
        self.fainted = fainted
        self.active = active
        self.revealed = revealed
        self.first_turn = first_turn
        self.available_z_moves = list(available_z_moves or [])


class DummyBattle:
    def __init__(self):
        move_a1 = DummyMove("tackle", current_pp=35, max_pp=35)
        move_a2 = DummyMove("fakeout", current_pp=10, max_pp=10, priority=3)
        move_b1 = DummyMove("protect", current_pp=10, max_pp=10, priority=4)
        move_b2 = DummyMove("thunderbolt", current_pp=15, max_pp=15)
        move_c1 = DummyMove("earthquake", current_pp=10, max_pp=10)
        move_c2 = DummyMove("recover", current_pp=8, max_pp=8)

        boosts = {
            "atk": 1,
            "def": 0,
            "spa": -1,
            "spd": 2,
            "spe": 0,
            "accuracy": 0,
            "evasion": -2,
        }

        self.mon_a = DummyPokemon(
            "Pikachu",
            types=("ELECTRIC",),
            status=Status.SLP,
            status_counter=2,
            current_hp=75,
            max_hp=100,
            base_speed=90,
            boosts=boosts,
            effects={Effect.SUBSTITUTE, Effect.TAUNT},
            protect_counter=1,
            must_recharge=False,
            item="lightball",
            ability="static",
            moves=[move_a1, move_a2],
            revealed=True,
            active=True,
            first_turn=True,
            fainted=False,
            available_z_moves=[move_a1],
        )
        self.mon_b = DummyPokemon(
            "Garchomp",
            types=("DRAGON", "GROUND"),
            status=Status.BRN,
            status_counter=0,
            current_hp=120,
            max_hp=180,
            base_speed=102,
            boosts={**boosts, "atk": 2},
            effects={Effect.FLINCH},
            protect_counter=0,
            must_recharge=True,
            item="choicescarf",
            ability="roughskin",
            moves=[move_b1, move_b2],
            revealed=True,
            active=True,
            first_turn=False,
            fainted=False,
            available_z_moves=[move_b2],
        )
        self.mon_c = DummyPokemon(
            "Amoonguss",
            types=("GRASS", "POISON"),
            status=None,
            status_counter=0,
            current_hp=150,
            max_hp=150,
            base_speed=30,
            boosts=boosts,
            effects=set(),
            protect_counter=0,
            must_recharge=False,
            item=None,
            ability="regenerator",
            moves=[move_c1, move_c2],
            revealed=False,
            active=False,
            first_turn=False,
            fainted=False,
            available_z_moves=[],
        )

        self.opp_a = DummyPokemon(
            "Landorus",
            types=("GROUND", "FLYING"),
            status=Status.PSN,
            status_counter=0,
            current_hp=60,
            max_hp=200,
            base_speed=91,
            boosts=boosts,
            effects={Effect.CONFUSION},
            protect_counter=0,
            must_recharge=False,
            item="yacheberry",
            ability="intimidate",
            moves=[move_a1, move_b1],
            revealed=True,
            active=True,
            first_turn=False,
            fainted=False,
            available_z_moves=[move_a1],
        )
        self.opp_b = DummyPokemon(
            "FlutterMane",
            types=("GHOST", "FAIRY"),
            status=None,
            status_counter=0,
            current_hp=90,
            max_hp=120,
            base_speed=135,
            boosts=boosts,
            effects=set(),
            protect_counter=0,
            must_recharge=False,
            item=None,
            ability="protosynthesis",
            moves=[move_a1, move_b2],
            revealed=True,
            active=True,
            first_turn=False,
            fainted=False,
            available_z_moves=[],
        )
        self.opp_c = DummyPokemon(
            "Amoonguss",
            types=("GRASS", "POISON"),
            status=Status.PAR,
            status_counter=0,
            current_hp=60,
            max_hp=150,
            base_speed=30,
            boosts=boosts,
            effects={Effect.ENCORE},
            protect_counter=0,
            must_recharge=False,
            item=None,
            ability="regenerator",
            moves=[move_c1],
            revealed=True,
            active=False,
            first_turn=False,
            fainted=False,
            available_z_moves=[],
        )

        self.turn = 12
        self.weather = {Weather.RAINDANCE: self.turn - 2}
        self.fields = {Field.TRICK_ROOM: self.turn - 3}
        self.side_conditions = {SideCondition.AURORA_VEIL: 1}
        self.opponent_side_conditions = {
            SideCondition.LIGHT_SCREEN: 2,
            SideCondition.REFLECT: 3,
        }
        self.force_switch = [True, False]
        self.available_moves = [[move_a1, move_a2], [move_b1]]
        self.available_switches = [[self.mon_c], [self.mon_c]]
        self.trapped = [False, True]
        self.can_mega_evolve = [False, False]
        self.can_z_move = [True, False]
        self.can_dynamax = [False, True]
        self.can_tera = [True, True]
        self.team = {"a": self.mon_a, "b": self.mon_b, "c": self.mon_c}
        self.opponent_team = {"x": self.opp_a, "y": self.opp_b, "z": self.opp_c}
        self.active_pokemon = [self.mon_a, self.mon_b]
        self.opponent_active_pokemon = [self.opp_a, self.opp_b]
        self.rules = []
        self.gen = 9


def build_dummy_battle():
    return DummyBattle()


__all__ = ["DummyMove", "build_dummy_battle"]
