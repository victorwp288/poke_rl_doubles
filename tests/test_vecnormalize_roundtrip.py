import sys
from pathlib import Path

import numpy as np
from gymnasium import Env, spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(str(Path(__file__).resolve().parents[1]))


class ConstantEnv(Env):
    def __init__(self):
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        obs = np.array([0.5, -0.5, 1.0], dtype=np.float32)
        return obs, {}

    def step(self, action):
        self._step += 1
        obs = np.array([0.5 + self._step * 0.1, -0.5, 1.0], dtype=np.float32)
        reward = float(self._step)
        terminated = self._step >= 3
        truncated = False
        return obs, reward, terminated, truncated, {}


def test_vecnormalize_roundtrip_preserves_stats(tmp_path: Path):
    venv = DummyVecEnv([ConstantEnv])
    vec = VecNormalize(venv, norm_obs=True, norm_reward=True)
    vec.reset()
    for _ in range(3):
        vec.step([0])

    vec_path = tmp_path / "vecnormalize.pkl"
    vec.save(str(vec_path))

    loaded = VecNormalize.load(str(vec_path), venv)
    loaded.training = False
    loaded.norm_reward = False

    sample = np.array([0.6, -0.5, 1.0], dtype=np.float32)
    original = vec.normalize_obs(sample.copy())
    restored = loaded.normalize_obs(sample.copy())

    np.testing.assert_allclose(original, restored, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(vec.obs_rms.mean, loaded.obs_rms.mean, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(vec.obs_rms.var, loaded.obs_rms.var, rtol=0.0, atol=0.0)
