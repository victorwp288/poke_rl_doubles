import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from gymnasium import spaces

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.online import env_helpers
from src.online.env import MaskableDoublesEnv


class DummyOpponent:
    def choose_move(self, battle):
        return 0


class DummyAgent:
    def __init__(self, username: str):
        self.username = username


class DummyBaseEnv:
    def __init__(self, raise_on_default: bool = False):
        self.agent1 = DummyAgent("agent1")
        self.agent2 = DummyAgent("agent2")
        self._battle_format = "gen9doublesou"
        self.battle1 = None
        self.battle2 = object()
        self.fake = False
        self.strict = False
        self._np_random = np.random.RandomState(0)
        self._step_count = 0
        self._raised = False
        self.raise_on_default = raise_on_default

        self.observation_spaces = {
            self.agent1.username: spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
            self.agent2.username: spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        }
        self.action_spaces = {
            self.agent1.username: spaces.MultiDiscrete([3, 3]),
            self.agent2.username: spaces.MultiDiscrete([3, 3]),
        }

    def order_to_action(self, order, battle, fake=False, strict=False):
        return np.asarray([0, 0], dtype=int)

    def reset(self, seed=None, options=None):
        self._step_count = 0
        self._raised = False
        obs = np.zeros(4, dtype=np.float32)
        return {self.agent1.username: obs, self.agent2.username: obs}, {
            self.agent1.username: {},
            self.agent2.username: {},
        }

    def step(self, actions):
        action = np.asarray(actions[self.agent1.username], dtype=int).reshape(-1).tolist()
        if self.raise_on_default and action == [0, 0] and not self._raised:
            self._raised = True
            raise AssertionError("invalid action")
        self._step_count += 1
        obs = np.full(4, self._step_count, dtype=np.float32)
        rewards = {self.agent1.username: 0.0, self.agent2.username: 0.0}
        terms = {self.agent1.username: False, self.agent2.username: False}
        truncs = {self.agent1.username: False, self.agent2.username: False}
        infos = {self.agent1.username: {}, self.agent2.username: {}}
        return {self.agent1.username: obs, self.agent2.username: obs}, rewards, terms, truncs, infos

    def close(self):
        return None


def test_action_mask_concat_order_and_dtype(monkeypatch):
    battle = SimpleNamespace(finished=False)

    def fake_slot_mask(_, slot, act_size):
        return [1, 0, 1] if slot == 0 else [0, 1, 0]

    monkeypatch.setattr("src.online.env_helpers.slot_action_mask", fake_slot_mask)

    mask = env_helpers._joint_action_mask(battle, 3)
    assert mask.dtype == np.uint8
    assert mask.tolist() == [1, 0, 1, 0, 1, 0]


def test_maskable_env_constructs_and_steps():
    env = MaskableDoublesEnv(base_env=DummyBaseEnv(), opponent=DummyOpponent())
    obs, info = env.reset()
    assert obs.shape == (4,)
    assert isinstance(info, dict)
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(np.asarray([1, 1], dtype=int))
        assert obs.shape == (4,)
        assert isinstance(info, dict)
        assert terminated is False
        assert truncated is False


def test_step_sets_sanitized_and_repaired_info_flags():
    env = MaskableDoublesEnv(
        base_env=DummyBaseEnv(raise_on_default=True),
        opponent=DummyOpponent(),
    )

    with (
        patch.object(
            MaskableDoublesEnv,
            "_sanitize_action",
            return_value=(np.asarray([0, 0], dtype=int), True),
        ),
        patch.object(
            MaskableDoublesEnv,
            "_repair_action",
            return_value=np.asarray([1, 1], dtype=int),
        ),
        patch.object(MaskableDoublesEnv, "_maps_to_default", return_value=False),
        patch.object(MaskableDoublesEnv, "_is_default_action", return_value=False),
        patch.object(MaskableDoublesEnv, "_describe_vector", return_value="<order>"),
    ):
        _, _, _, _, info = env.step(np.asarray([2, 2], dtype=int))

    assert info.get("sanitized_action") is True
    assert info.get("repaired_action") is True
