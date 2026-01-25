from __future__ import annotations

from ..constants import FeatureConfig

# Assemble the fixed-order observation vector for all policies.
from .base_features import ObservationBaseMixin
from .mon_features import ObservationMonMixin
from .type_features import ObservationTypeMixin
from .utils import ObservationUtilsMixin

ObservationVector = list[float]


class ObservationEncoder(
    ObservationBaseMixin,
    ObservationMonMixin,
    ObservationTypeMixin,
    ObservationUtilsMixin,
):
    def __init__(self, config=None):
        self.config = config or FeatureConfig()
        self.type_index = {name: idx for idx, name in enumerate(self.config.type_names)}
        self.status_index = {name: idx for idx, name in enumerate(self.config.status_names)}
        self.layered_side_conditions = dict(self.config.layered_side_conditions)
        self._gen_data_cache = {}

    @property
    def size(self):
        return self.config.observation_size

    def encode(self, battle) -> ObservationVector:
        features = []
        # Feature order is a training/eval contract; reordering breaks saved datasets/checkpoints.
        # Append only, and regenerate fixtures if this order changes.
        features.extend(self._base_slots(battle))
        features.extend(self._global_state(battle))
        player_slots = list(battle.active_pokemon)
        opponent_slots = list(battle.opponent_active_pokemon)
        for mon in player_slots:
            features.extend(self._per_mon(mon, opponent_slots))
        for mon in opponent_slots:
            features.extend(self._per_mon(mon, player_slots))
        features.extend(self._type_matchups(battle, player_slots, opponent_slots))
        features.extend(self._priority_flags(player_slots, opponent_slots))
        features.extend(self._fake_out_flags(player_slots))
        features.extend(self._type_coverage(battle.team))
        features.extend(self._type_coverage(battle.opponent_team))
        features.extend(self._legal_action_counts(battle))
        # Pad to fixed observation size expected by policies.
        return self._pad(features)


ENCODER = ObservationEncoder()
CONFIG = ENCODER.config
TYPE_NAMES = CONFIG.type_names
STATUS_NAMES = CONFIG.status_names
OBSERVATION_SIZE = CONFIG.observation_size


def encode_observation(battle) -> ObservationVector:
    return ENCODER.encode(battle)


def observation_size() -> int:
    return OBSERVATION_SIZE


__all__ = [
    "CONFIG",
    "ENCODER",
    "OBSERVATION_SIZE",
    "ObservationEncoder",
    "STATUS_NAMES",
    "TYPE_NAMES",
    "encode_observation",
    "observation_size",
]
