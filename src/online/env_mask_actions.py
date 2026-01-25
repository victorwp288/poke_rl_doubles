import numpy as np
from gymnasium import spaces
from poke_env.environment.doubles_env import DoublesEnv

from src.core.action_mask import _legal_orders


class MaskableEnvActionMixin:
    def _mask_allows_action(self, vector, slot_masks):
        if slot_masks is None:
            return True
        for idx, value in enumerate(vector):
            if idx >= slot_masks.shape[0]:
                break
            if value < 0 or value >= slot_masks.shape[1]:
                return False
            if not slot_masks[idx, value]:
                return False
        return True

    def _within_action_space(self, vector):
        space = getattr(self, "action_space", None)
        if not isinstance(space, spaces.MultiDiscrete):
            return True
        arr = np.asarray(vector, dtype=int).reshape(-1)
        if arr.size != len(space.nvec):
            return False
        for value, limit in zip(arr, space.nvec, strict=False):
            if value < 0 or value >= int(limit):
                return False
        return True

    def _action_space(self):
        space = getattr(self, "action_space", None)
        return space if isinstance(space, spaces.MultiDiscrete) else None

    def _action_nvec(self):
        space = self._action_space()
        return [int(n) for n in space.nvec] if space else []

    def _action_vector(self, action):
        return np.asarray(action, dtype=int).reshape(-1)

    def _reshape_like_space(self, vector):
        arr = np.asarray(vector, dtype=int)
        space = self._action_space()
        if space and arr.shape != space.shape:
            arr = arr.reshape(space.shape)
        return arr

    def _action_to_order(self, vector, battle, strict=None):
        vector_arr = np.asarray(vector, dtype=int)
        if strict is True:
            flags = (True, False)
        elif strict is False:
            flags = (False,)
        else:
            flags = (True, False)
        for flag in flags:
            try:
                return DoublesEnv.action_to_order(vector_arr, battle, fake=False, strict=flag)
            except Exception:
                continue
        return None

    def _order_to_vector(self, battle, order):
        for flag in (True, False):
            try:
                action = DoublesEnv.order_to_action(order, battle, fake=False, strict=flag)
                return self._action_vector(action)
            except Exception:
                continue
        return None

    def _legal_orders_safe(self, battle, index):
        try:
            return _legal_orders(battle, index)
        except Exception:
            return []

    def _same_action_vector(self, first, second):
        left = np.asarray(first, dtype=int).reshape(-1)
        right = np.asarray(second, dtype=int).reshape(-1)
        return np.array_equal(left, right)

    def _describe_vector(self, vector):
        battle = self._battle_state()
        order = self._action_to_order(vector, battle, False)
        if order is None:
            return "<order_error>"
        return str(order)

    def _mask_summary(self, slot_masks):
        if slot_masks is None:
            return None
        try:
            masks = np.asarray(slot_masks, dtype=bool)
        except Exception:
            return None
        summary = []
        try:
            rows = masks.reshape(masks.shape[0], -1)
        except Exception:
            rows = masks
        for idx, row in enumerate(np.atleast_2d(rows)):
            try:
                legal = np.where(row)[0]
            except Exception:
                legal = np.array([], dtype=int)
            sample = legal[:5].tolist()
            summary.append(
                {
                    "slot": int(idx),
                    "legal_count": int(legal.size),
                    "sample": sample,
                }
            )
        return summary
