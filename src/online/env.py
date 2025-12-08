import contextlib
import threading
import time
from pathlib import Path

import numpy as np
from gymnasium import spaces
from poke_env.battle import DoubleBattle
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.battle_order import DefaultBattleOrder, DoubleBattleOrder, SingleBattleOrder

from src.config import section
from src.core.env import action_space_size
from src.core.features import (
    _legal_orders,
    combine_slot_masks,
    encode_observation,
    observation_size,
    slot_action_mask,
)

_METRIC_KEYS = {
    "faint": "faint_diff",
    "team_hp": "team_hp_advantage",
    "active_hp": "active_hp_advantage",
    "status": "status_advantage",
    "side_condition": "side_condition_advantage",
}


def _default_rewards():
    rewards = {
        "win": 1.0,
        "loss": -1.0,
        "draw": -0.1,
        "faint": 0.1,
        "team_hp": 0.05,
        "active_hp": 0.1,
        "status": 0.05,
        "side_condition": 0.05,
    }
    config = section("online")
    base = config.get("base_rewards", {})
    if isinstance(base, dict):
        for key, value in base.items():
            if key in rewards and value is not None:
                rewards[key] = float(value)
    return rewards


def _reward_stats(rewards):
    return {
        "win_reward": rewards["win"],
        "loss_penalty": rewards["loss"],
        "draw_penalty": rewards["draw"],
        "faint_reward": rewards["faint"],
        "team_hp_reward": rewards["team_hp"],
        "active_hp_reward": rewards["active_hp"],
        "status_reward": rewards["status"],
        "side_condition_reward": rewards["side_condition"],
    }


def _team_hp_fraction(team):
    total = 0.0
    count = 0
    for mon in team:
        if mon is None:
            continue
        max_hp = getattr(mon, "max_hp", None)
        current_hp = getattr(mon, "current_hp", None)
        if max_hp:
            numer = current_hp if current_hp is not None else max_hp
            total += float(numer) / float(max_hp)
            count += 1
            continue
        fraction = getattr(mon, "current_hp_fraction", None)
        if fraction is not None:
            total += float(fraction)
            count += 1
            continue
        total += 1.0
        count += 1
    return total / max(1.0, float(count))


def _active_hp_fraction(slots):
    total = 0.0
    count = 0
    for mon in slots:
        if mon is None:
            continue
        max_hp = getattr(mon, "max_hp", None)
        current_hp = getattr(mon, "current_hp", None)
        if max_hp:
            numer = current_hp if current_hp is not None else max_hp
            total += float(numer) / float(max_hp)
            count += 1
            continue
        fraction = getattr(mon, "current_hp_fraction", None)
        if fraction is not None:
            total += float(fraction)
            count += 1
            continue
        total += 1.0
        count += 1
    return total / max(1.0, float(count))


def _status_count(team):
    count = 0
    for mon in team:
        if mon is None:
            continue
        if getattr(mon, "status", None):
            count += 1
    return count


def _reward_metrics(battle, knocked_out_me, knocked_out_opp):
    return {
        "faint_diff": float(knocked_out_opp - knocked_out_me),
        "team_hp_advantage": _team_hp_fraction(battle.team.values())
        - _team_hp_fraction(battle.opponent_team.values()),
        "active_hp_advantage": _active_hp_fraction(battle.active_pokemon)
        - _active_hp_fraction(battle.opponent_active_pokemon),
        "status_advantage": float(
            _status_count(battle.opponent_team.values()) - _status_count(battle.team.values())
        ),
        "side_condition_advantage": float(
            len(getattr(battle, "opponent_side_conditions", {}))
            - len(getattr(battle, "side_conditions", {}))
        ),
    }


def _score_from_metrics(rewards, metrics):
    total = 0.0
    for reward_key, metric_key in _METRIC_KEYS.items():
        total += rewards[reward_key] * metrics[metric_key]
    return total


def _battle_key(battle):
    return battle.battle_tag, getattr(battle, "player_role", "player")


def _battle_mask(battle, act_size):
    mask_a = slot_action_mask(battle, 0, act_size)
    mask_b = slot_action_mask(battle, 1, act_size)
    return combine_slot_masks(mask_a, mask_b)


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
        return np.asarray(encode_observation(battle), dtype=np.float32)

    def calc_reward(self, battle):
        key = _battle_key(battle)
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
        return self._last_stats.get(_battle_key(battle))

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


class MaskableDoublesEnv(SingleAgentWrapper):
    def __init__(self, base_env, opponent, *, step_delay=0.0):
        self._act_size = action_space_size(base_env._battle_format)
        super().__init__(base_env, opponent)
        if isinstance(self.action_space, spaces.MultiDiscrete) and len(self.action_space.nvec) >= 1:
            self._act_size = int(self.action_space.nvec[0])
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
        root = Path(__file__).resolve().parents[2]
        self._log_path = root / "outputs" / "logs" / "online_env.log"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = None

        import logging
        from logging.handlers import RotatingFileHandler
        self.logger = logging.getLogger("online_env")
        if not any(isinstance(h,RotatingFileHandler) for h in self.logger.handlers):
            self.logger.setLevel(logging.INFO)
            h=RotatingFileHandler(self._log_path, maxBytes=20*1024*1024, backupCount=3)
            import logging as _l
            fmt=_l.Formatter("%(asctime)s - %(message)s")
            h.setFormatter(fmt)
            self.logger.addHandler(h)


    @property
    def base_env(self):
        return self.env

    def get_action_mask(self):
        battle = self.base_env.battle1
        if not isinstance(battle, DoubleBattle) or battle.finished:
            return np.ones(self._mask_shape, dtype=np.uint8)
        return _battle_mask(battle, self._act_size)

    def action_masks(self):
        return self.get_action_mask().astype(bool, copy=False)

    def _sanitize_action(self, action):
        space = self._action_space()
        if space is None:
            return np.asarray(action), False

        vector = self._vector(action)
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

    def _mask_allows(self, vector, slot_masks):
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

    def _within_space(self, vector):
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

    def _nvec(self):
        space = self._action_space()
        return [int(n) for n in space.nvec] if space else []

    def _vector(self, action):
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
                return self._vector(action)
            except Exception:
                continue
        return None

    def _legal_orders_safe(self, battle, index):
        try:
            return _legal_orders(battle, index)
        except Exception:
            return []

    def _same_action(self, first, second):
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

    def _log_invalid_action_context(self, retry_choice, slot_masks, candidate_snapshot):
        try:
            choice_vec = np.asarray(retry_choice, dtype=int).reshape(-1).tolist()
            strict_default = bool(self._maps_to_default(retry_choice, strict=True))
            loose_default = bool(self._maps_to_default(retry_choice, strict=False))
            order_desc = self._describe_vector(retry_choice)
            mask_summary = self._mask_summary(slot_masks)
            context = {
                "choice": choice_vec,
                "order": order_desc,
                "maps_to_default_strict": strict_default,
                "maps_to_default_loose": loose_default,
                "mask_summary": mask_summary,
                "candidate_queue": candidate_snapshot,
            }
            message = f"[online env] invalid action context {context}"
            print(message, flush=True)
            self._write_step_log(message, {})
        except Exception:
            pass

    def _log_repair_failure(self, vector, slot_masks, joint_candidates):
        try:
            mask_summary = self._mask_summary(slot_masks)
            preview = []
            for candidate in joint_candidates[:3]:
                arr = np.asarray(candidate, dtype=int)
                preview.append(
                    {
                        "vector": arr.reshape(-1).tolist(),
                        "order": self._describe_vector(arr),
                        "maps_to_default_strict": bool(self._maps_to_default(arr, strict=True)),
                    }
                )
            context = {
                "repair_failed_vector": np.asarray(vector, dtype=int).reshape(-1).tolist(),
                "mask_summary": mask_summary,
                "joint_candidate_count": int(len(joint_candidates)),
                "candidate_preview": preview,
            }
            message = f"[online env] repair failure {context}"
            print(message, flush=True)
            self._write_step_log(message, {})
        except Exception:
            pass

    def _order_is_default(self, order):
        if isinstance(order, DefaultBattleOrder):
            return True
        if isinstance(order, str):
            return order.strip() == "/choose default"
        command = getattr(order, "order", None)
        if isinstance(command, str):
            return command.strip() == "/choose default"
        return False

    def _maps_to_default(self, vector, strict=False):
        battle = self._battle_state()
        if battle is None:
            return False
        order = self._action_to_order(vector, battle, strict)
        return self._order_is_default(order) if order is not None else False

    def _is_default_action(self, vector):
        battle = self._battle_state()
        if battle is None:
            return False
        order = self._action_to_order(vector, battle, True)
        return self._order_is_default(order) if order is not None else False

    def _joint_candidates(self, slot_masks):
        battle = self._battle_state()
        if battle is None:
            return []
        orders_a = self._legal_orders_safe(battle, 0)
        orders_b = self._legal_orders_safe(battle, 1)
        try:
            joined = DoubleBattleOrder.join_orders(orders_a, orders_b)
        except Exception:
            joined = []
        masked_candidates = []
        unmasked_candidates = []
        for order in joined:
            vector = self._order_to_vector(battle, order)
            if vector is None or not self._within_space(vector):
                continue
            if self._mask_allows(vector, slot_masks):
                masked_candidates.append(vector)
            else:
                unmasked_candidates.append(vector)
        if masked_candidates:
            return masked_candidates
        if unmasked_candidates:
            return unmasked_candidates
        forced_vectors = self._forced_switch_vectors(battle, slot_masks=slot_masks)
        if forced_vectors:
            return forced_vectors
        nvec = self._nvec()
        if nvec:
            zeros = np.zeros(len(nvec), dtype=int)
            if self._mask_allows(zeros, slot_masks):
                return [zeros]
        return []

    def _forced_switch_vectors(self, battle, slot_masks=None):
        force_flags = getattr(battle, "force_switch", None)
        if not isinstance(force_flags, list | tuple):
            return []
        if not any(force_flags):
            return []
        slot_orders = []
        for idx in range(2):
            forced = bool(force_flags[idx]) if idx < len(force_flags) else False
            if forced:
                switches = (
                    battle.available_switches[idx]
                    if idx < len(getattr(battle, "available_switches", []))
                    else []
                )
                orders = [SingleBattleOrder(switch) for switch in switches or []]
                if not orders:
                    return []
            else:
                orders = self._legal_orders_safe(battle, idx)
                if not orders:
                    return []
            slot_orders.append(orders)
        while len(slot_orders) < 2:
            slot_orders.append(self._legal_orders_safe(battle, len(slot_orders)))
        if len(slot_orders) < 2 or not slot_orders[0] or not slot_orders[1]:
            return []
        try:
            joined = DoubleBattleOrder.join_orders(slot_orders[0], slot_orders[1])
        except Exception:
            joined = []
        vectors = []
        for order in joined:
            vector = self._order_to_vector(battle, order)
            if vector is None or not self._within_space(vector):
                continue
            if slot_masks is not None and not self._mask_allows(vector, slot_masks):
                continue
            if self._maps_to_default(vector, strict=True):
                continue
            vectors.append(vector)
        return vectors

    def _repair_action(self, action, slot_masks):
        battle = self._battle_state()
        space = self._action_space()
        if battle is None or space is None:
            return None
        nvec = self._nvec()
        vector = self._vector(action)
        if vector.size < len(nvec):
            vector = np.pad(vector, (0, len(nvec) - vector.size), constant_values=0)
        vector = vector[: len(nvec)]
        for idx, limit in enumerate(nvec):
            if limit <= 0:
                vector[idx] = 0
            else:
                vector[idx] = max(0, min(vector[idx], limit - 1))
        sanitized_default = False
        if self._mask_allows(vector, slot_masks):
            order = self._action_to_order(vector, battle, True)
            if order is not None:
                sanitized_default = self._order_is_default(order)
                if not sanitized_default:
                    return self._reshape_like_space(vector)
                for forced in self._forced_switch_vectors(battle, slot_masks=slot_masks):
                    if self._maps_to_default(forced, strict=True):
                        continue
                    return self._reshape_like_space(forced)
        joint_candidates = list(self._joint_candidates(slot_masks))
        for candidate in joint_candidates:
            order = self._action_to_order(candidate, battle, True)
            if order is None:
                continue
            if not self._order_is_default(order):
                return self._reshape_like_space(candidate)
        for forced in self._forced_switch_vectors(battle, slot_masks=slot_masks):
            if self._maps_to_default(forced, strict=True):
                continue
            return self._reshape_like_space(forced)
        self._log_repair_failure(vector, slot_masks, joint_candidates)
        if sanitized_default and self._mask_allows(vector, slot_masks):
            return self._reshape_like_space(vector)
        return None

    def _default_fallback(self, slot_masks):
        space = self._action_space()
        if space is None:
            return None
        nvec = self._nvec()
        if not nvec:
            return None
        zeros = np.zeros(len(nvec), dtype=int)
        if self._mask_allows(zeros, slot_masks):
            return self._reshape_like_space(zeros)
        return None

    def _safe_action_fallback(self, slot_masks, candidate_vectors=None):
        space = self._action_space()
        if space is None:
            return None
        vectors = candidate_vectors
        if vectors is None:
            vectors = list(self._joint_candidates(slot_masks))
        for candidate in vectors:
            if self._maps_to_default(candidate, strict=True):
                continue
            return self._reshape_like_space(candidate)
        fallback = self._default_fallback(slot_masks)
        if fallback is not None and not self._maps_to_default(fallback, strict=True):
            return fallback
        return None

    def _candidate_queue(self, action):
        sanitized, sanitized_changed = self._sanitize_action(action)
        slot_masks = self._slot_masks()
        queue = []
        sanitized_is_default = self._is_default_action(sanitized)
        default_candidate = None
        if sanitized_is_default:
            default_candidate = (sanitized, True, False)
        else:
            queue.append((sanitized, bool(sanitized_changed), False))
        repaired = self._repair_action(sanitized, slot_masks)
        if (
            repaired is not None
            and not self._same_action(repaired, sanitized)
            and not self._maps_to_default(repaired, strict=True)
        ):
            queue.append((repaired, True, True))
        fallback = self._default_fallback(slot_masks)
        if fallback is not None and all(
            not self._same_action(fallback, existing[0]) for existing in queue
        ):
            if not self._maps_to_default(fallback, strict=True):
                queue.append((fallback, True, True))
            elif default_candidate is None:
                default_candidate = (fallback, True, True)
        if default_candidate is not None:
            queue.append(default_candidate)
        return queue

    def _run_candidate(self, choice, changed, repaired, step_start, snapshot, candidate_vectors):
        wait_needed = self._min_step_delay - (time.monotonic() - self._last_step_time)
        if self._min_step_delay > 0.0 and wait_needed > 0.0:
            time.sleep(wait_needed)
        retry_choice = self._vector(choice)
        retry_changed = changed
        retry_repaired = repaired
        while True:
            timer = threading.Timer(self._step_timeout, self._force_progress)
            timer.daemon = True
            timer.start()
            call_start = time.monotonic()
            try:
                if self._maps_to_default(retry_choice, strict=True):
                    fallback_choice = self._first_non_default_candidate()
                    if fallback_choice is not None and not self._same_action(
                        fallback_choice, retry_choice
                    ):
                        retry_choice = self._vector(fallback_choice)
                        retry_changed = True
                        retry_repaired = True
                env_choice = self._reshape_like_space(retry_choice)
                obs, reward, terminated, truncated, info = super().step(env_choice)
                call_end = time.monotonic()
                timer.cancel()
                self._last_step_time = call_end
                info = self._augment_info(info, retry_changed, retry_repaired)
                self._step_counter += 1

                reward_value = self._reward_value(reward)
                stats = info.get("battle_stats") if isinstance(info, dict) else None
                turn = stats.get("turn") if isinstance(stats, dict) else None
                result = stats.get("result") if isinstance(stats, dict) else None
                need_order = (
                    retry_changed
                    or retry_repaired
                    or (turn is not None and turn != self._last_logged_turn)
                )
                order_desc = self._describe_vector(retry_choice) if need_order else None
                if turn is not None:
                    self._last_logged_turn = turn
                total_ms = (call_end - step_start) * 1000.0
                wait_ms = (call_end - call_start) * 1000.0
                elapsed = call_end - self._start_time
                sps = self._step_counter / elapsed if elapsed > 0.0 else 0.0
                candidate_vec = self._vector(retry_choice).tolist()
                message = self._step_message(
                    turn=turn,
                    candidate_vec=candidate_vec,
                    changed=retry_changed,
                    repaired=retry_repaired,
                    reward_value=reward_value,
                    total_ms=total_ms,
                    wait_ms=wait_ms,
                    sps=sps,
                    result=result,
                    order_desc=order_desc,
                )
                print(message, flush=True)
                self._write_step_log(message, info)
                return (obs, reward, terminated, truncated, info), None
            except AssertionError as exc:
                timer.cancel()
                if "invalid action" not in str(exc):
                    raise
                slot_masks = self._slot_masks()
                self._log_invalid_action_context(retry_choice, slot_masks, snapshot)
                replacement = self._repair_action(retry_choice, slot_masks)
                if replacement is not None and not self._same_action(replacement, retry_choice):
                    retry_choice = self._vector(replacement)
                    retry_changed = True
                    retry_repaired = True
                    continue
                fallback_choice = self._first_non_default_candidate()
                if fallback_choice is not None and not self._same_action(
                    fallback_choice, retry_choice
                ):
                    retry_choice = self._vector(fallback_choice)
                    retry_changed = True
                    retry_repaired = True
                    continue
                safe_choice = self._safe_action_fallback(slot_masks, candidate_vectors)
                if safe_choice is not None and not self._same_action(safe_choice, retry_choice):
                    retry_choice = self._vector(safe_choice)
                    retry_changed = True
                    retry_repaired = True
                    continue
                return None, exc
            except Exception:
                timer.cancel()
                raise

    def _snapshot_candidates(self, candidates):
        snapshot = []
        vectors = []
        for vector, changed, repaired in candidates:
            arr = np.asarray(vector, dtype=int).reshape(-1)
            vectors.append(arr.copy())
            snapshot.append(
                {
                    "vector": arr.tolist(),
                    "changed": bool(changed),
                    "repaired": bool(repaired),
                    "maps_to_default_strict": bool(self._maps_to_default(arr, strict=True)),
                    "maps_to_default_loose": bool(self._maps_to_default(arr, strict=False)),
                    "order": self._describe_vector(arr),
                }
            )
        return snapshot, vectors

    def _reward_value(self, reward):
        try:
            reward_array = np.asarray(reward)
            if getattr(reward_array, "size", 0):
                return float(reward_array.reshape(-1)[0])
            return float(reward)
        except (TypeError, ValueError):
            try:
                return float(reward)
            except (TypeError, ValueError):
                return 0.0

    def _step_message(
        self,
        turn,
        candidate_vec,
        changed,
        repaired,
        reward_value,
        total_ms,
        wait_ms,
        sps,
        result,
        order_desc,
    ):
        parts = [
            f"[online env] step={self._step_counter}",
            f"turn={int(turn) if turn is not None else '-'}",
            f"candidate={candidate_vec}",
            f"changed={bool(changed)}",
            f"repaired={bool(repaired)}",
            f"reward={reward_value:.3f}",
            f"total_ms={total_ms:.1f}",
            f"wait_ms={wait_ms:.1f}",
            f"sps={sps:.2f}",
        ]
        if result:
            parts.append(f"result={result}")
        if order_desc is not None:
            parts.append(f"order={order_desc}")
        return " ".join(parts)

    def _first_non_default_candidate(self):
        if self._action_space() is None:
            return None
        slot_masks = self._slot_masks()
        for vector in self._joint_candidates(slot_masks):
            if not self._maps_to_default(vector, strict=True):
                return self._reshape_like_space(vector)
        return None

    def _augment_info(self, info, changed, repaired):
        info = dict(info)
        try:
            info["action_mask"] = self.get_action_mask().astype(bool, copy=False)
        except Exception:
            info["action_mask"] = None
        battle = self._battle_state()
        if isinstance(battle, DoubleBattle):
            info["battle_stats"] = self.base_env.latest_stats(battle)
        if changed:
            info["sanitized_action"] = True
        if repaired:
            info["repaired_action"] = True
        if self._timeout_triggered:
            info["step_timeout"] = True
            self._timeout_triggered = False
        return info

    def _write_step_log(self, message, info):
        if not hasattr(self, "_log_path"):
            return
        try:
            stats = info.get("battle_stats") if isinstance(info, dict) else None
            stats_text = ""
            if isinstance(stats, dict):
                stats_text = f" | stats={stats}"
            handle = self._ensure_log_handle()
            handle.write(f"{time.time():.3f} {message}{stats_text}\n")
            handle.flush()
        except Exception:
            pass

    def _ensure_log_handle(self):
        if self._log_handle is None:
            self._log_handle = self._log_path.open("a", encoding="utf-8")
        return self._log_handle

    def _close_log_handle(self):
        if self._log_handle is not None:
            with contextlib.suppress(Exception):
                self._log_handle.close()
            self._log_handle = None

    def close(self):
        self._close_log_handle()
        close_fn = getattr(super(), "close", None)
        if callable(close_fn):
            close_fn()

    def _force_progress(self):
        with self._timeout_lock:
            base_env = self.base_env
            agent1 = getattr(base_env, "agent1", None)
            agent2 = getattr(base_env, "agent2", None)
            for agent in (agent1, agent2):
                if agent is None:
                    continue
                for attr in ("_waiting", "_trying_again"):
                    event = getattr(agent, attr, None)
                    if event is not None:
                        try:
                            event.set()
                        except Exception:
                            continue
            self._timeout_triggered = True

    def reset(self, *, seed=None, options=None):
        attempts = 0
        while True:
            try:
                obs, info = super().reset(seed=seed, options=options)
                break
            except RuntimeError as exc:
                message = str(exc)
                if "Agent is not challenging" not in message or attempts >= 5:
                    raise
                wait = 0.5 * (attempts + 1)
                print(
                    f"[online env] retrying reset (attempt={attempts + 1} wait={wait:.1f}s cause={message})",
                    flush=True,
                )
                time.sleep(wait)
                attempts += 1
        info = dict(info)
        info["action_mask"] = self.get_action_mask().astype(bool, copy=False)
        battle = self.base_env.battle1
        if isinstance(battle, DoubleBattle):
            info["battle_stats"] = self.base_env.latest_stats(battle)
        return obs, info

    def step(self, action):
        step_start = time.monotonic()
        candidates = self._candidate_queue(action)
        candidate_snapshot, candidate_vectors = self._snapshot_candidates(candidates)
        last_error = None
        for choice, changed, repaired in candidates:
            result, error = self._run_candidate(
                choice, changed, repaired, step_start, candidate_snapshot, candidate_vectors
            )
            if result is not None:
                return result
            if error is not None:
                last_error = error
        raise AssertionError("invalid action (repair failed)") from last_error


def make_maskable_env(
    *, opponent, battle_format="gen9doublesou", rewards=None, step_delay=0.0, **env_kwargs
):
    base_env = Gen9DoublesEnv(battle_format=battle_format, rewards=rewards, **env_kwargs)
    return MaskableDoublesEnv(base_env=base_env, opponent=opponent, step_delay=step_delay)


__all__ = ["Gen9DoublesEnv", "MaskableDoublesEnv", "make_maskable_env"]
