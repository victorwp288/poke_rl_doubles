"""
Core online environment wrapper (poke-env ↔ Gymnasium ↔ SB3).

Summary:
- `Gen9DoublesEnv` defines the fixed observation space (393-dim vector) and computes per-step rewards.
- `MaskableDoublesEnv` wraps the base env to add action masks + sanitize→repair→fallback so PPO never
  has to learn from illegal actions.

High-signal contracts:
- `action_masks()` returns a boolean mask shaped `(2 * act_size,)` interpreted as `[slot0 | slot1]`.
- Observation ordering is fixed across offline/online; changing it breaks dataset/checkpoint parity.
"""
import threading
import time
from pathlib import Path

import numpy as np
from gymnasium import spaces
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from src.core.env import action_space_size
from src.core.observation import encode_observation, observation_size
from src.online.env_helpers import _battle_cache_key
from src.online.env_mask_actions import MaskableEnvActionMixin
from src.online.env_mask_core import MaskableEnvCoreMixin
from src.online.env_mask_logging import MaskableEnvLoggingMixin
from src.online.env_mask_repair import MaskableEnvRepairMixin
from src.online.env_mask_step import MaskableEnvStepMixin
from src.online.env_rewards import (
    _default_rewards,
    _reward_metrics,
    _reward_stats,
    _score_from_metrics,
)


class Gen9DoublesEnv(DoublesEnv):
    def __init__(self, battle_format="gen9doublesou", rewards=None, **kwargs):
        self._battle_format = battle_format
        self._obs_dim = observation_size()
        self._obs_space = spaces.Box(low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)
        base = _default_rewards()
        if isinstance(rewards, dict):
            for key, value in rewards.items():
                if key in base and value is not None:
                    base[key] = float(value)
        self._rewards = base
        self._score_cache = {}
        self._last_stats = {}
        super().__init__(battle_format=battle_format, **kwargs)
        self.observation_spaces = {agent: self._obs_space for agent in self.possible_agents}

    def embed_battle(self, battle):
        # Single encoding entrypoint: battle state -> fixed observation vector.
        return np.asarray(encode_observation(battle), dtype=np.float32)

    def calc_reward(self, battle):
        key = _battle_cache_key(battle)
        knocked_out_me = float(sum(mon.fainted for mon in battle.team.values()))
        knocked_out_opp = float(sum(mon.fainted for mon in battle.opponent_team.values()))
        metrics = _reward_metrics(battle, knocked_out_me, knocked_out_opp)
        total = _score_from_metrics(self._rewards, metrics)
        previous = self._score_cache.get(key)
        previous_total = previous["total"] if previous else 0.0
        delta = total - previous_total
        self._score_cache[key] = {"metrics": metrics, "total": total}
        stats = {
            "battle_tag": battle.battle_tag,
            "knocked_out_me": knocked_out_me,
            "knocked_out_opp": knocked_out_opp,
            "score": float(total),
            "delta": float(delta),
            "turn": getattr(battle, "turn", 0),
        }
        stats.update(metrics)
        stats.update(_reward_stats(self._rewards))
        if battle.finished:
            outcome = "win" if battle.won else "loss" if battle.lost else "draw"
            bonus = self._rewards[outcome]
            stats["result"] = outcome
            delta += bonus
            stats["terminal_bonus"] = bonus
        self._last_stats[key] = stats
        return float(delta)

    def calc_term_trunc(self, battle):
        return bool(battle.finished), False

    def reset(self, seed=None, options=None):
        observations, info = super().reset(seed=seed, options=options)
        self._score_cache.clear()
        self._last_stats.clear()
        return observations, info

    def latest_stats(self, battle):
        return self._last_stats.get(_battle_cache_key(battle))

    def latest_global_stats(self):
        if not self._last_stats:
            return None
        return next(reversed(self._last_stats.values()))

    def set_rewards(self, **updates):
        changed = {}
        for key, value in updates.items():
            if key in self._rewards and value is not None:
                changed[key] = float(value)
        if not changed:
            return
        self._rewards.update(changed)
        stats_map = _reward_stats(self._rewards)
        for cache in self._score_cache.values():
            metrics = cache["metrics"]
            cache["total"] = _score_from_metrics(self._rewards, metrics)
        for key, stats in self._last_stats.items():
            cached = self._score_cache.get(key)
            if cached and "total" in cached:
                stats["score"] = float(cached["total"])
            else:
                stats["score"] = stats.get("score", 0.0)
            stats.update(stats_map)


class MaskableDoublesEnv(
    MaskableEnvCoreMixin,
    MaskableEnvActionMixin,
    MaskableEnvRepairMixin,
    MaskableEnvLoggingMixin,
    MaskableEnvStepMixin,
    SingleAgentWrapper,
):
    _global_log_lock = threading.Lock()
    _global_last_log_time = 0.0
    _global_ema_reward = 0.0
    _global_ema_count = 0
    _global_last_reward = 0.0
    _global_last_sps = 0.0
    _global_last_turn = None
    _global_last_step = 0

    def __init__(
        self,
        base_env,
        opponent,
        *,
        step_delay=0.0,
        console_log_mode="off",
        console_log_interval_sec=5.0,
    ):
        self._act_size = action_space_size(base_env._battle_format)
        super().__init__(base_env, opponent)
        if isinstance(self.action_space, spaces.MultiDiscrete) and len(self.action_space.nvec) >= 1:
            self._act_size = int(self.action_space.nvec[0])
        # Action mask is flattened as [slot0 | slot1] with length 2 * act_size.
        # This mask layout is a train/eval contract (see docs/ARCHITECTURE.md and golden mask tests).
        self._mask_shape = (2 * self._act_size,)
        self._step_timeout = 5.0
        self._timeout_lock = threading.Lock()
        self._timeout_triggered = False
        now = time.monotonic()
        self._min_step_delay = max(float(step_delay), 0.0)
        self._last_step_time = now
        self._start_time = now
        self._step_counter = 0
        self._last_logged_turn = None
        # Normalize console_log_mode; accept booleans and strings.
        if console_log_mode is False:
            mode_token = "off"
        else:
            mode_token = str(console_log_mode or "summary").strip().lower()
            if mode_token not in {"summary", "debug", "off"}:
                mode_token = "summary"
        self._console_log_mode = mode_token
        self._console_log_interval_sec = max(float(console_log_interval_sec or 0.0), 0.0)
        self._last_console_log_time = now
        root = Path(__file__).resolve().parents[2]
        self._log_path = root / "outputs" / "logs" / "online_env.log"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = None

        import logging
        from logging.handlers import RotatingFileHandler

        self.logger = logging.getLogger("online_env")
        if not any(isinstance(h, RotatingFileHandler) for h in self.logger.handlers):
            self.logger.setLevel(logging.INFO)
            h = RotatingFileHandler(self._log_path, maxBytes=20 * 1024 * 1024, backupCount=3)
            import logging as _l

            fmt = _l.Formatter("%(asctime)s - %(message)s")
            h.setFormatter(fmt)
            self.logger.addHandler(h)


def make_maskable_env(
    *,
    opponent,
    battle_format="gen9doublesou",
    rewards=None,
    step_delay=0.0,
    console_log_mode="off",
    console_log_interval_sec=5.0,
    **env_kwargs,
):
    base_env = Gen9DoublesEnv(battle_format=battle_format, rewards=rewards, **env_kwargs)
    return MaskableDoublesEnv(
        base_env=base_env,
        opponent=opponent,
        step_delay=step_delay,
        console_log_mode=console_log_mode,
        console_log_interval_sec=console_log_interval_sec,
    )


__all__ = ["Gen9DoublesEnv", "MaskableDoublesEnv", "make_maskable_env"]
