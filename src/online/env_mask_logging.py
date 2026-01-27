import contextlib
import time

import numpy as np
from poke_env.battle import DoubleBattle


class MaskableEnvLoggingMixin:
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

    def _augment_info(self, info, changed, repaired):
        info = dict(info)
        try:
            info["action_mask"] = self.get_action_mask().astype(bool, copy=False)
        except Exception:
            info["action_mask"] = None
        battle = self._battle_state()
        if isinstance(battle, DoubleBattle):
            info["battle_stats"] = self.base_env.latest_stats(battle)
        # Flags separate input sanitization from post-sanitize repair substitutions.
        # Parity/semantics are checked in `tests/test_env_smoke.py`.
        if changed:
            info["sanitized_action"] = True  # input action was modified
        if repaired:
            info["repaired_action"] = True  # post-sanitize repair was applied
        if self._timeout_triggered:
            info["step_timeout"] = True
            self._timeout_triggered = False
        return info

    def _should_print_step_log(self, reward_value, result, changed, repaired):
        mode = self._console_log_mode
        if mode == "off":
            return False
        # Summary and other modes: suppress per-step logging; training callback handles summaries.
        return mode == "debug"

    def _maybe_global_summary_log(self, reward_value, sps, turn, result):
        # No-op by default; summary printing is handled by the training callback.
        if self._console_log_mode != "debug":
            return
        now = time.monotonic()
        cls = self.__class__
        with cls._global_log_lock:
            alpha = 0.1
            if cls._global_ema_count == 0:
                cls._global_ema_reward = reward_value
            else:
                cls._global_ema_reward += alpha * (reward_value - cls._global_ema_reward)
            cls._global_ema_count += 1
            cls._global_last_reward = reward_value
            cls._global_last_sps = sps
            cls._global_last_turn = turn
            cls._global_last_step = max(cls._global_last_step, self._step_counter)
            if now - cls._global_last_log_time < self._console_log_interval_sec:
                return
            cls._global_last_log_time = now
            message = (
                f"[online env] debug_summary step={cls._global_last_step} "
                f"sps={cls._global_last_sps:.1f} "
                f"reward_last={cls._global_last_reward:.3f} "
                f"reward_ema={cls._global_ema_reward:.3f}"
            )
            if cls._global_last_turn is not None:
                message += f" turn={cls._global_last_turn}"
            print(message, flush=True)

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

    def latest_global_stats(self):
        base = getattr(self, "base_env", None)
        if base is None:
            return None
        getter = getattr(base, "latest_global_stats", None)
        return getter() if callable(getter) else None

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
