from pathlib import Path

import numpy as np
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.baselines import SimpleHeuristicsPlayer

from src.core.action_mask import slot_action_mask
from src.core.observation import encode_observation
from src.online.policy.load import load_maskable_policy

from .utils import _action_to_tuple


class RecordingHeuristics(SimpleHeuristicsPlayer):
    def __init__(self, recorder, act_size, teacher_name, **kwargs):
        self._recorder = recorder
        self._act_size = act_size
        self._teacher_name = teacher_name
        super().__init__(**kwargs)

    def choose_move(self, battle):
        order = super().choose_move(battle)
        mask0 = slot_action_mask(battle, 0, self._act_size)
        mask1 = slot_action_mask(battle, 1, self._act_size)
        first, second = _action_to_tuple(order, battle)
        # Step-record schema contract (used by offline dataset loader):
        # - observation: list[float] (len 393)
        # - action: [slot0_action, slot1_action]
        # - mask: [slot0_mask, slot1_mask] (each len act_size)
        record = {
            "battle_tag": battle.battle_tag,
            "turn": battle.turn,
            "teacher": self._teacher_name,
            "format": battle.format,
            "observation": encode_observation(battle),
            "action": [first, second],
            "mask": [mask0, mask1],
        }
        self._recorder.write(record)
        return order


class PolicyTeacherPlayer(SimpleHeuristicsPlayer):
    """
    Player that uses a saved MaskablePPO policy to select actions.
    Falls back to SimpleHeuristics if the policy proposes an invalid order.
    """

    def __init__(self, recorder, act_size, model_path, **kwargs):
        self._recorder = recorder
        self._act_size = act_size
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"policy teacher checkpoint not found: {self._model_path}")
        self._model = load_maskable_policy(self._model_path, device="cpu")
        super().__init__(**kwargs)

    def choose_move(self, battle):
        mask0 = slot_action_mask(battle, 0, self._act_size)
        mask1 = slot_action_mask(battle, 1, self._act_size)
        concat_mask = np.concatenate([mask0, mask1]).astype(np.int8)
        obs = np.asarray(encode_observation(battle), dtype=np.float32)

        order = None
        action = None
        try:
            action, _ = self._model.predict(obs, action_masks=concat_mask, deterministic=True)
            action = np.asarray(action, dtype=int)
            if action.shape[0] == 2:
                order = DoublesEnv.action_to_order(action, battle, fake=False, strict=False)
        except Exception as exc:  # pragma: no cover - robustness path
            print(f"[warn] policy teacher predict failed: {exc}", flush=True)

        # Fallback if policy output is unusable
        if order is None:
            order = super().choose_move(battle)

        mask0 = slot_action_mask(battle, 0, self._act_size)
        mask1 = slot_action_mask(battle, 1, self._act_size)
        first, second = _action_to_tuple(order, battle)
        record = {
            "battle_tag": battle.battle_tag,
            "turn": battle.turn,
            "teacher": "policy",
            "format": battle.format,
            "observation": encode_observation(battle),
            "action": [first, second],
            "mask": [mask0, mask1],
        }
        self._recorder.write(record)
        return order


__all__ = ["PolicyTeacherPlayer", "RecordingHeuristics"]
