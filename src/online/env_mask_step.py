import threading
import time

import numpy as np
from poke_env.battle import DoubleBattle


class MaskableEnvStepMixin:
    def _run_candidate(self, choice, changed, repaired, step_start, snapshot, candidate_vectors):
        wait_needed = self._min_step_delay - (time.monotonic() - self._last_step_time)
        if self._min_step_delay > 0.0 and wait_needed > 0.0:
            time.sleep(wait_needed)
        retry_choice = self._action_vector(choice)
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
                    if fallback_choice is not None and not self._same_action_vector(
                        fallback_choice, retry_choice
                    ):
                        retry_choice = self._action_vector(fallback_choice)
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
                candidate_vec = self._action_vector(retry_choice).tolist()
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
                if self._should_print_step_log(
                    reward_value=reward_value,
                    result=result,
                    changed=retry_changed,
                    repaired=retry_repaired,
                ):
                    print(message, flush=True)
                self._maybe_global_summary_log(
                    reward_value=reward_value,
                    sps=sps,
                    turn=turn,
                    result=result,
                )
                self._write_step_log(message, info)
                return (obs, reward, terminated, truncated, info), None
            except AssertionError as exc:
                timer.cancel()
                if "invalid action" not in str(exc):
                    raise
                slot_masks = self._slot_masks()
                self._log_invalid_action_context(retry_choice, slot_masks, snapshot)
                replacement = self._repair_action(retry_choice, slot_masks)
                if replacement is not None and not self._same_action_vector(
                    replacement, retry_choice
                ):
                    retry_choice = self._action_vector(replacement)
                    retry_changed = True
                    retry_repaired = True
                    continue
                fallback_choice = self._first_non_default_candidate()
                if fallback_choice is not None and not self._same_action_vector(
                    fallback_choice, retry_choice
                ):
                    retry_choice = self._action_vector(fallback_choice)
                    retry_changed = True
                    retry_repaired = True
                    continue
                safe_choice = self._safe_action_fallback(slot_masks, candidate_vectors)
                if safe_choice is not None and not self._same_action_vector(
                    safe_choice, retry_choice
                ):
                    retry_choice = self._action_vector(safe_choice)
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
        # Action handling is sanitize -> repair -> fallback; info flags record which happened.
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
