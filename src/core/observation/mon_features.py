from poke_env.battle.status import Status


class ObservationMonMixin:
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
