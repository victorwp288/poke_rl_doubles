from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.weather import Weather


class FeatureConfig:
    def __init__(self):
        self.turn_cap = 100
        self.screen_turns = 5
        self.room_turns = 5
        self.tailwind_turns = 4
        self.legal_action_divisor = 16.0
        self.per_mon_features = 44
        self.global_features = 61
        self.observation_size = 393
        self.type_names = (
            "NORMAL",
            "FIRE",
            "WATER",
            "ELECTRIC",
            "GRASS",
            "ICE",
            "FIGHTING",
            "POISON",
            "GROUND",
            "FLYING",
            "PSYCHIC",
            "BUG",
            "ROCK",
            "GHOST",
            "DRAGON",
            "DARK",
            "STEEL",
            "FAIRY",
        )
        self.status_names = ("SLP", "PAR", "BRN", "FRZ", "PSN", "TOX")
        self.weather_order = (
            None,
            Weather.SUNNYDAY,
            Weather.RAINDANCE,
            Weather.SANDSTORM,
            Weather.HAIL,
            Weather.SNOW,
        )
        self.terrain_order = (
            None,
            Field.ELECTRIC_TERRAIN,
            Field.GRASSY_TERRAIN,
            Field.MISTY_TERRAIN,
            Field.PSYCHIC_TERRAIN,
        )
        self.field_room_order = (
            Field.GRAVITY,
            Field.TRICK_ROOM,
            Field.MAGIC_ROOM,
            Field.WONDER_ROOM,
        )
        self.screen_side_conditions = (
            SideCondition.REFLECT,
            SideCondition.LIGHT_SCREEN,
            SideCondition.AURORA_VEIL,
        )
        self.support_side_conditions = (
            SideCondition.SAFEGUARD,
            SideCondition.MIST,
            SideCondition.LUCKY_CHANT,
        )
        self.hazard_conditions = (
            SideCondition.STEALTH_ROCK,
            SideCondition.SPIKES,
            SideCondition.TOXIC_SPIKES,
            SideCondition.STICKY_WEB,
        )
        self.volatile_effects = (
            Effect.SUBSTITUTE,
            Effect.TAUNT,
            Effect.ENCORE,
            Effect.DISABLE,
            Effect.LOCKED_MOVE,
            Effect.PARTIALLY_TRAPPED,
            Effect.CONFUSION,
            Effect.FLINCH,
            Effect.TORMENT,
            Effect.HEAL_BLOCK,
            Effect.EMBARGO,
            Effect.MAGNET_RISE,
            Effect.LASER_FOCUS,
            Effect.YAWN,
            Effect.LEECH_SEED,
            Effect.PERISH3,
        )
        self.boost_order = (
            "atk",
            "def",
            "spa",
            "spd",
            "spe",
            "accuracy",
            "evasion",
        )
        self.last_action_categories = (
            "ATTACK_PHYSICAL",
            "ATTACK_SPECIAL",
            "STATUS",
            "SWITCH",
            "PROTECT",
            "STRUGGLE",
        )
        self.layered_side_conditions = (
            (SideCondition.SPIKES, 3),
            (SideCondition.TOXIC_SPIKES, 2),
        )


__all__ = ["FeatureConfig"]
