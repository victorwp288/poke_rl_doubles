class ObservationTypeMixin:
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
