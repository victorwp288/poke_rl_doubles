import uuid
from pathlib import Path

import numpy as np
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.baselines import SimpleHeuristicsPlayer

from src.core.action_mask import slot_action_mask
from src.core.observation import encode_observation
from src.online.policy.load import load_maskable_policy


def _unique_username(prefix):
    token = (prefix or "bot")[:11]
    suffix = uuid.uuid4().hex[:6]
    return f"{token}{suffix}"


class PolicyOpponentPlayer(SimpleHeuristicsPlayer):
    """
    Opponent driven by a frozen MaskablePPO checkpoint.
    Falls back to SimpleHeuristics if prediction fails.
    """

    def __init__(self, model_path, act_size, **kwargs):
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"opponent policy checkpoint not found: {self._model_path}")
        self._model = load_maskable_policy(self._model_path, device="cpu")
        self._act_size = act_size
        super().__init__(**kwargs)

    def choose_move(self, battle):
        mask0 = slot_action_mask(battle, 0, self._act_size)
        mask1 = slot_action_mask(battle, 1, self._act_size)
        concat_mask = np.concatenate([mask0, mask1]).astype(np.int8)
        obs = np.asarray(encode_observation(battle), dtype=np.float32)
        try:
            action, _ = self._model.predict(obs, action_masks=concat_mask, deterministic=True)
            action = np.asarray(action, dtype=int)
            if action.shape[0] == 2:
                order = DoublesEnv.action_to_order(action, battle, fake=False, strict=False)
                if order is not None:
                    return order
        except Exception:
            pass
        return super().choose_move(battle)


__all__ = ["PolicyOpponentPlayer", "_unique_username"]
