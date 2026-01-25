from poke_env.battle.side_condition import SideCondition


class ObservationBaseMixin:
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
        terrain_keys = self._fields_as_keys(fields)
        terrain_set = {terrain for terrain in self.config.terrain_order if terrain is not None}
        has_active = any(key in terrain_set for key in terrain_keys)
        for idx, terrain in enumerate(self.config.terrain_order):
            if terrain is None:
                if not has_active:
                    vector[idx] = 1.0
                continue
            if terrain in terrain_keys:
                vector[idx] = 1.0
        if not any(vector):
            vector[0] = 1.0
        return vector

    def _fields_as_keys(self, fields):
        if isinstance(fields, dict):
            return tuple(fields.keys())
        if fields is None:
            return ()
        try:
            return tuple(fields)
        except TypeError:
            return ()

    def _weather_turns_left(self, weather, turn):
        if not weather:
            return 0.0
        condition = next(iter(weather))
        start_turn = weather[condition]
        remaining = max(0, self.config.screen_turns - (turn - start_turn))
        return self._normalize_timer(remaining, self.config.screen_turns)

    def _terrain_turns_left(self, fields, turn):
        field_map = fields if isinstance(fields, dict) else {}
        for terrain in self.config.terrain_order:
            if terrain is None:
                continue
            start = field_map.get(terrain)
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
