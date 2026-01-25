from poke_env.data import GenData


class ObservationUtilsMixin:
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
