import numpy as np
from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.data import GenData
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.battle_order import DefaultBattleOrder, DoubleBattleOrder, SingleBattleOrder


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


class ObservationEncoder:
    def __init__(self, config=None):
        self.config = config or FeatureConfig()
        self.type_index = {name: idx for idx, name in enumerate(self.config.type_names)}
        self.status_index = {name: idx for idx, name in enumerate(self.config.status_names)}
        self.layered_side_conditions = dict(self.config.layered_side_conditions)
        self._gen_data_cache = {}

    @property
    def size(self):
        return self.config.observation_size

    def encode(self, battle):
        features = []
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
        return self._pad(features)

    def _base_slots(self, battle):
        slots = list(battle.active_pokemon) + list(battle.opponent_active_pokemon)
        features = []
        for mon in slots:
            features.append(self._hp_ratio(mon))
            features.extend(self._status_vector(mon))
            features.extend(self._type_vector(mon))
        return features

    def _global_state(self, battle):
        features = []
        features.extend(self._weather_vector(battle.weather))
        features.append(self._weather_turns_left(battle.weather, battle.turn))
        features.extend(self._terrain_vector(battle.fields))
        features.append(self._terrain_turns_left(battle.fields, battle.turn))
        features.append(self._clamp01(float(battle.turn) / self.config.turn_cap))
        features.append(1.0 if battle.turn % 2 == 0 else 0.0)
        features.append(self._max_turns(battle))
        for room_field in self.config.field_room_order:
            features.append(1.0 if room_field in battle.fields else 0.0)
            features.append(
                self._side_condition_value(
                    battle.fields, room_field, battle.turn, self.config.room_turns
                )
            )
        my_side = battle.side_conditions
        opp_side = battle.opponent_side_conditions
        features.append(
            self._side_condition_value(
                my_side,
                SideCondition.TAILWIND,
                battle.turn,
                self.config.tailwind_turns,
            )
        )
        features.append(
            self._side_condition_value(
                opp_side,
                SideCondition.TAILWIND,
                battle.turn,
                self.config.tailwind_turns,
            )
        )
        for condition in self.config.screen_side_conditions:
            features.append(
                self._side_condition_value(
                    my_side, condition, battle.turn, self.config.screen_turns
                )
            )
            features.append(
                self._side_condition_value(
                    opp_side, condition, battle.turn, self.config.screen_turns
                )
            )
        for condition in self.config.support_side_conditions:
            features.append(
                self._side_condition_value(
                    my_side, condition, battle.turn, self.config.screen_turns
                )
            )
            features.append(
                self._side_condition_value(
                    opp_side, condition, battle.turn, self.config.screen_turns
                )
            )
        for condition in (SideCondition.MATBLOCK, SideCondition.WIDE_GUARD):
            features.append(self._side_condition_value(my_side, condition, battle.turn, 1))
            features.append(self._side_condition_value(opp_side, condition, battle.turn, 1))
        features.extend(self._hazard_values(my_side, battle.turn))
        features.extend(self._hazard_values(opp_side, battle.turn))
        features.append(self._team_alive_fraction(battle.team))
        features.append(self._team_alive_fraction(battle.opponent_team))
        features.append(self._team_fainted_fraction(battle.team))
        features.append(self._team_fainted_fraction(battle.opponent_team))
        force_switch = getattr(battle, "force_switch", [False, False])
        features.append(1.0 if force_switch[0] else 0.0)
        features.append(1.0 if force_switch[1] else 0.0)
        return self._truncate(features, self.config.global_features)

    def _per_mon(self, mon, opponents):
        if mon is None:
            return [0.0] * self.config.per_mon_features
        features = []
        for stat in self.config.boost_order:
            features.append(self._normalize_boost(mon.boosts.get(stat)))
        for effect in self.config.volatile_effects:
            features.append(self._effect_active(mon, effect))
        features.append(self._protect_active(mon))
        features.append(1.0 if mon.must_recharge else 0.0)
        features.append(self._item_revealed(mon))
        features.append(self._ability_revealed(mon))
        features.append(self._moves_seen_fraction(mon))
        presence, pp = self._move_presence_and_pp(mon)
        features.extend(presence)
        features.extend(pp)
        sleep, toxic = self._status_timers(mon)
        features.append(sleep)
        features.append(toxic)
        features.append(self._relative_speed_hint(mon, opponents))
        features.extend(self._last_action_features(mon))
        features.append(1.0 if mon.fainted else 0.0)
        features.append(1.0 if getattr(mon, "active", False) else 0.0)
        features.append(1.0 if mon.revealed else 0.0)
        features.append(self._clamp01(self._speed_value(mon) / 500.0))
        return self._truncate(features, self.config.per_mon_features)

    def _type_matchups(self, battle, player_slots, opponent_slots):
        features = []
        type_chart = self._gen_data(battle.gen).type_chart
        opponent_types = [type_ for opp in opponent_slots for type_ in self._mon_types(opp)]
        player_types = [type_ for ally in player_slots for type_ in self._mon_types(ally)]
        for mon in player_slots:
            mon_types = self._mon_types(mon)
            features.append(self._type_multiplier(mon_types, opponent_types, type_chart))
            features.append(self._type_multiplier(opponent_types, mon_types, type_chart))
            features.append(self._type_multiplier(player_types, mon_types, type_chart))
        for mon in opponent_slots:
            mon_types = self._mon_types(mon)
            features.append(self._type_multiplier(mon_types, player_types, type_chart))
            features.append(self._type_multiplier(player_types, mon_types, type_chart))
            features.append(self._type_multiplier(opponent_types, mon_types, type_chart))
        return self._truncate(features, 12)

    def _priority_flags(self, player_slots, opponent_slots):
        combined = list(player_slots) + list(opponent_slots)
        flags = [self._priority_move_known(mon) for mon in combined]
        return self._truncate(flags, 4)

    def _fake_out_flags(self, player_slots):
        flags = [self._fake_out_available(mon) for mon in player_slots]
        return self._truncate(flags, 2)

    def _type_coverage(self, team):
        counts = {type_name: 0 for type_name in self.config.type_names}
        for mon in team.values():
            if mon.revealed:
                for type_name in self._mon_types(mon):
                    counts[type_name] += 1
        total = sum(counts.values())
        if total == 0:
            return [0.0] * len(self.config.type_names)
        return [counts[type_name] / float(total) for type_name in self.config.type_names]

    def _legal_action_counts(self, battle):
        counts = []
        for idx in range(2):
            move_count = (
                len(battle.available_moves[idx]) if idx < len(battle.available_moves) else 0
            )
            switch_count = (
                len(battle.available_switches[idx]) if idx < len(battle.available_switches) else 0
            )
            total = move_count + switch_count
            counts.append(self._clamp01(total / self.config.legal_action_divisor))
        return counts

    def _hp_ratio(self, mon):
        if mon is None:
            return 0.0
        current = mon.current_hp or 0
        maximum = mon.max_hp or 0
        return float(current) / float(maximum) if maximum else 0.0

    def _status_vector(self, mon):
        vector = [0] * len(self.config.status_names)
        if mon is None:
            return vector
        status = mon.status
        if status is None:
            return vector
        idx = self.status_index.get(status.name)
        if idx is not None:
            vector[idx] = 1
        return vector

    def _type_vector(self, mon):
        vector = [0] * len(self.config.type_names)
        if mon is None:
            return vector
        for entry in mon.types or []:
            idx = self.type_index.get(entry.name)
            if idx is not None:
                vector[idx] = 1
        return vector

    def _weather_vector(self, weather):
        vector = [0.0] * len(self.config.weather_order)
        for idx, condition in enumerate(self.config.weather_order):
            if condition is None and not weather:
                vector[idx] = 1.0
                break
            if condition is not None and condition in weather:
                vector[idx] = 1.0
                break
        else:
            vector[0] = 1.0
        return vector

    def _terrain_vector(self, fields):
        vector = [0.0] * len(self.config.terrain_order)
        has_active = any(field is not None and field.is_terrain for field in fields)
        for idx, terrain in enumerate(self.config.terrain_order):
            if terrain is None:
                if not has_active:
                    vector[idx] = 1.0
                continue
            if terrain in fields:
                vector[idx] = 1.0
        if not any(vector):
            vector[0] = 1.0
        return vector

    def _weather_turns_left(self, weather, turn):
        if not weather:
            return 0.0
        condition = next(iter(weather))
        start_turn = weather[condition]
        remaining = max(0, self.config.screen_turns - (turn - start_turn))
        return self._normalize_timer(remaining, self.config.screen_turns)

    def _terrain_turns_left(self, fields, turn):
        for terrain in self.config.terrain_order:
            if terrain is None:
                continue
            start = fields.get(terrain)
            if start is not None:
                remaining = max(0, self.config.screen_turns - (turn - start))
                return self._normalize_timer(remaining, self.config.screen_turns)
        return 0.0

    def _side_condition_value(self, side_conditions, condition, turn, default_duration):
        value = side_conditions.get(condition)
        if value is None:
            return 0.0
        if isinstance(condition, SideCondition):
            duration = self.layered_side_conditions.get(condition, default_duration)
            if condition in self.layered_side_conditions:
                return self._normalize_timer(value, duration)
        else:
            duration = default_duration
        remaining = max(0, duration - (turn - value))
        return self._normalize_timer(remaining, duration)

    def _hazard_values(self, side_conditions, turn):
        values = []
        for condition in self.config.hazard_conditions:
            duration = self.layered_side_conditions.get(condition, self.config.screen_turns)
            values.append(self._side_condition_value(side_conditions, condition, turn, duration))
        return values

    def _team_alive_fraction(self, team):
        total = len(team)
        if total == 0:
            return 0.0
        alive = sum(not mon.fainted for mon in team.values())
        return float(alive) / float(total)

    def _team_fainted_fraction(self, team):
        total = len(team)
        if total == 0:
            return 0.0
        fainted = sum(mon.fainted for mon in team.values())
        return float(fainted) / float(total)

    def _max_turns(self, battle):
        for rule in battle.rules:
            if "maximum" not in rule.lower():
                continue
            parts = rule.split("=")
            if len(parts) == 2 and parts[1].isdigit():
                return self._normalize_timer(int(parts[1]), self.config.turn_cap)
        return 0.0

    def _effect_active(self, mon, effect):
        return 1.0 if effect in mon.effects else 0.0

    def _protect_active(self, mon):
        return 1.0 if getattr(mon, "protect_counter", 0) > 0 else 0.0

    def _normalize_boost(self, boost):
        value = max(min(boost or 0, 6), -6)
        return (value + 6) / 12.0

    def _stage_multiplier(self, stage):
        if stage >= 0:
            return (2 + stage) / 2
        return 2 / (2 - stage)

    def _speed_value(self, mon):
        if mon is None:
            return 0.0
        base = mon.base_stats.get("spe") or 0
        boost = mon.boosts.get("spe", 0)
        return float(base) * self._stage_multiplier(boost)

    def _relative_speed_hint(self, mon, opponents):
        if mon is None:
            return 0.0
        own_speed = self._speed_value(mon)
        opp_speeds = [self._speed_value(opp) for opp in opponents if opp is not None]
        if not opp_speeds:
            return 0.0
        mean_speed = sum(opp_speeds) / float(len(opp_speeds))
        total = own_speed + mean_speed
        return own_speed / total if total else 0.0

    def _moves_seen_fraction(self, mon):
        if mon is None:
            return 0.0
        return float(len(mon.moves)) / 4.0

    def _move_presence_and_pp(self, mon):
        presence = [0.0] * 4
        pp = [0.0] * 4
        if mon is None:
            return presence, pp
        moves = sorted(mon.moves.values(), key=lambda move: move.id)
        for idx, move in enumerate(moves[:4]):
            presence[idx] = 1.0
            denominator = move.max_pp or 1
            pp[idx] = self._clamp01(float(move.current_pp) / float(denominator))
        return presence, pp

    def _status_timers(self, mon):
        if mon is None:
            return 0.0, 0.0
        sleep = self._clamp01(float(mon.status_counter) / 5.0) if mon.status == Status.SLP else 0.0
        toxic = self._clamp01(float(mon.status_counter) / 15.0) if mon.status == Status.TOX else 0.0
        return sleep, toxic

    def _item_revealed(self, mon):
        if mon is None:
            return 0.0
        item = mon.item
        return 1.0 if item is not None and item != mon._data.UNKNOWN_ITEM else 0.0

    def _ability_revealed(self, mon):
        if mon is None:
            return 0.0
        return 1.0 if mon.ability is not None else 0.0

    def _priority_move_known(self, mon):
        if mon is None:
            return 0.0
        return 1.0 if any(move.priority > 0 for move in mon.moves.values()) else 0.0

    def _fake_out_available(self, mon):
        if mon is None:
            return 0.0
        has_fake_out = any(move.id == "fakeout" for move in mon.moves.values())
        return 1.0 if has_fake_out and getattr(mon, "first_turn", False) else 0.0

    def _last_action_features(self, mon):
        size = len(self.config.last_action_categories)
        if mon is None:
            return [0.0] * size
        # TODO: populate last-action categories once poke-env exposes per-mon last action metadata.
        return [0.0] * size

    def _mon_types(self, mon):
        if mon is None or not mon.types:
            return ()
        return tuple(entry.name for entry in mon.types)

    def _type_multiplier(self, attack_types, defender_types, type_chart):
        if not attack_types or not defender_types:
            return 0.0
        best = 0.0
        for attack in attack_types:
            chart_row = type_chart.get(attack, {})
            total = 1.0
            for defense in defender_types:
                total *= chart_row.get(defense, 1.0)
            best = max(best, total)
        return best

    def _truncate(self, values, target):
        if len(values) < target:
            return values + [0.0] * (target - len(values))
        del values[target:]
        return values

    def _pad(self, values):
        if len(values) < self.config.observation_size:
            values.extend([0.0] * (self.config.observation_size - len(values)))
        else:
            del values[self.config.observation_size :]
        return values

    def _normalize_timer(self, value, max_turns):
        if max_turns <= 0:
            return 0.0
        return float(max(min(value, max_turns), 0)) / float(max_turns)

    def _clamp01(self, value):
        return float(max(min(value, 1.0), 0.0))

    def _gen_data(self, gen):
        data = self._gen_data_cache.get(gen)
        if data is None:
            data = GenData.from_gen(gen)
            self._gen_data_cache[gen] = data
        return data


ENCODER = ObservationEncoder()
CONFIG = ENCODER.config
TYPE_NAMES = CONFIG.type_names
STATUS_NAMES = CONFIG.status_names
OBSERVATION_SIZE = CONFIG.observation_size


def encode_observation(battle):
    return ENCODER.encode(battle)


def _legal_orders(battle, slot):
    orders = []
    if slot < 0 or slot >= 2:
        return orders
    active = battle.active_pokemon
    mon = active[slot] if slot < len(active) else None
    switches = battle.available_switches[slot] if slot < len(battle.available_switches) else []
    force_switch = getattr(battle, "force_switch", [False, False])
    if force_switch[slot]:
        for switch in switches:
            orders.append(SingleBattleOrder(switch))
        return orders
    if mon is None:
        return orders

    moves = battle.available_moves[slot] if slot < len(battle.available_moves) else []
    available_z_moves = set(getattr(mon, "available_z_moves", []) or [])
    for move in moves:
        try:
            targets = battle.get_possible_showdown_targets(move, mon)
        except Exception:
            targets = [0]
        if not targets:
            targets = [0]
        for target in targets:
            orders.append(SingleBattleOrder(move, move_target=target))
            if battle.can_mega_evolve[slot]:
                orders.append(SingleBattleOrder(move, move_target=target, mega=True))
            if battle.can_z_move[slot] and move in available_z_moves:
                orders.append(SingleBattleOrder(move, move_target=target, z_move=True))
            if battle.can_dynamax[slot]:
                orders.append(SingleBattleOrder(move, move_target=target, dynamax=True))
            if battle.can_tera[slot]:
                orders.append(SingleBattleOrder(move, move_target=target, terastallize=True))

    if not battle.trapped[slot]:
        for switch in switches:
            orders.append(SingleBattleOrder(switch))
    return orders


def slot_action_mask(battle, slot, act_size):
    legal_actions = set()
    if slot < 0 or slot >= 2:
        return [1] * act_size

    force_flags = getattr(battle, "force_switch", [False, False])
    force_active = bool(force_flags[slot]) if slot < len(force_flags) else False
    available_switches = getattr(battle, "available_switches", [])
    both_forced_single = (
        len(force_flags) >= 2
        and all(bool(flag) for flag in force_flags[:2])
        and all(
            len(available_switches[idx]) == 1 if idx < len(available_switches) else False
            for idx in range(2)
        )
    )

    def _is_default_single(single_order):
        if single_order is None:
            return True
        if isinstance(single_order, DefaultBattleOrder):
            return True
        if isinstance(getattr(single_order, "order", None), str):
            normalized = single_order.order.strip().lower()
            return normalized in {"/choose default", "/choose pass"}
        return False

    def _maybe_add(action_value):
        if action_value is None:
            return
        try:
            value = int(action_value)
        except Exception:
            return
        if 0 <= value < act_size:
            legal_actions.add(value)

    def _collect_joint_orders():
        try:
            orders_a = _legal_orders(battle, 0)
        except Exception:
            orders_a = []
        try:
            orders_b = _legal_orders(battle, 1)
        except Exception:
            orders_b = []

        try:
            joint_orders = DoubleBattleOrder.join_orders(orders_a, orders_b)
        except Exception:
            joint_orders = []

        if not joint_orders:
            joint_orders = [DoubleBattleOrder(first_order=None, second_order=None)]

        for joint in joint_orders:
            single = joint.first_order if slot == 0 else joint.second_order
            if force_active and _is_default_single(single):
                continue
            try:
                vector = DoublesEnv.order_to_action(joint, battle, fake=False, strict=True)
            except Exception:
                try:
                    vector = DoublesEnv.order_to_action(joint, battle, fake=False, strict=False)
                except Exception:
                    continue
            arr = np.asarray(vector, dtype=int).reshape(-1)
            if slot >= arr.size:
                continue
            _maybe_add(arr[slot])

    def _collect_individual_orders():
        try:
            single_orders = _legal_orders(battle, slot)
        except Exception:
            single_orders = []
        for order in single_orders:
            if force_active and _is_default_single(order):
                continue
            try:
                raw_action = DoublesEnv._order_to_action_individual(order, battle, True, slot)
            except Exception:
                try:
                    raw_action = DoublesEnv._order_to_action_individual(order, battle, False, slot)
                except Exception:
                    continue
            _maybe_add(raw_action)

    _collect_joint_orders()
    _collect_individual_orders()

    if both_forced_single and force_active:
        legal_actions = {0}
    elif not legal_actions:
        if force_active:
            return [0] * act_size
        legal_actions.add(0)

    mask = [0] * act_size
    for action_raw in legal_actions:
        action_index = int(action_raw)
        if 0 <= action_index < act_size:
            mask[action_index] = 1
    return mask


def combine_slot_masks(mask_a, mask_b):
    arr_a = np.asarray(list(mask_a), dtype=np.uint8)
    arr_b = np.asarray(list(mask_b), dtype=np.uint8)
    return np.concatenate((arr_a, arr_b), axis=0)


def observation_size():
    return OBSERVATION_SIZE


__all__ = [
    "CONFIG",
    "OBSERVATION_SIZE",
    "encode_observation",
    "observation_size",
    "slot_action_mask",
    "combine_slot_masks",
    "STATUS_NAMES",
    "TYPE_NAMES",
]
