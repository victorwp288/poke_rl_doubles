import numpy as np
from poke_env.battle import DoubleBattle

from .env_helpers import _joint_action_mask


class MaskableEnvCoreMixin:
    @property
    def base_env(self):
        return self.env

    def get_action_mask(self):
        battle = self.base_env.battle1
        if not isinstance(battle, DoubleBattle) or battle.finished:
            # No active battle yet: allow all actions to keep rollout stable.
            return np.ones(self._mask_shape, dtype=np.uint8)
        return _joint_action_mask(battle, self._act_size)

    def action_masks(self):
        return self.get_action_mask().astype(bool, copy=False)

    def _sanitize_action(self, action):
        space = self._action_space()
        if space is None:
            return np.asarray(action), False

        vector = self._action_vector(action)
        slots = len(space.nvec)
        if vector.size < slots:
            vector = np.pad(vector, (0, slots - vector.size), constant_values=0)

        mask = self.get_action_mask()
        if mask is None:
            mask = np.ones(self._mask_shape, dtype=bool)
        mask = np.asarray(mask, dtype=bool).reshape(-1, self._act_size)

        cleaned = vector[:slots].copy()
        changed = False
        for idx, limit in enumerate(space.nvec):
            limit = int(limit)
            slot_mask = mask[idx] if idx < mask.shape[0] else np.ones(self._act_size, dtype=bool)
            legal = np.where(slot_mask)[0]
            if legal.size == 0:
                legal = np.arange(min(self._act_size, limit))
            choice = cleaned[idx] if idx < cleaned.size else 0
            if choice < 0 or choice >= limit or choice >= self._act_size or not slot_mask[choice]:
                cleaned[idx] = int(np.random.choice(legal))
                changed = True

        return cleaned.reshape(space.shape), changed

    def _battle_state(self):
        return getattr(self.base_env, "battle1", None)

    def _slot_masks(self):
        try:
            mask = self.get_action_mask()
        except Exception:
            return None
        if mask is None:
            return None
        arr = np.asarray(mask, dtype=bool).reshape(-1)
        if arr.size == 0:
            return None
        try:
            return arr.reshape(-1, self._act_size)
        except ValueError:
            return None
