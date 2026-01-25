import numpy as np

from .actions import _mask_allows, _mask_view
from .debug import _action_to_list, _log_invalid_action
from .repair import _repair_action
from .repair_candidates import _is_action_valid


def _step_with_repair(env, action, mask, max_attempts=5):
    candidate = np.asarray(action, dtype=int)
    attempt = 0
    current_mask = mask
    last_error = None
    while attempt < max_attempts:
        mask_arr, act_size, _ = _mask_view(env, None)
        if mask_arr is None or act_size is None:
            mask_arr, act_size, _ = _mask_view(env, current_mask)
        if not _mask_allows(candidate, env, mask_arr, act_size):
            _log_invalid_action(
                "mask_reject",
                env,
                candidate,
                note={"attempt": attempt, "candidate": _action_to_list(candidate)},
            )
            repaired = _repair_action(env, candidate, mask_arr)
            if repaired is None:
                _log_invalid_action(
                    "repair_unavailable",
                    env,
                    candidate,
                    note={"attempt": attempt, "candidate": _action_to_list(candidate)},
                )
                break
            candidate = np.asarray(repaired, dtype=int)
            current_mask = mask_arr
            attempt += 1
            continue
        if not _is_action_valid(candidate, env):
            _log_invalid_action(
                "preflight_invalid",
                env,
                candidate,
                note={"attempt": attempt, "candidate": _action_to_list(candidate)},
            )
            repaired = _repair_action(env, candidate, mask_arr)
            if repaired is None:
                _log_invalid_action(
                    "repair_unavailable",
                    env,
                    candidate,
                    note={"attempt": attempt, "candidate": _action_to_list(candidate)},
                )
                break
            candidate = np.asarray(repaired, dtype=int)
            current_mask = mask_arr
            attempt += 1
            continue
        try:
            return env.step(candidate)
        except AssertionError as exc:
            if "invalid action" not in str(exc):
                raise
            stage = "env_step_failure" if attempt == 0 else "repair_failure"
            _log_invalid_action(
                stage,
                env,
                candidate,
                note={"attempt": attempt, "candidate": _action_to_list(candidate)},
            )
            last_error = exc
            current_mask = None
            repaired = _repair_action(env, candidate, None)
            if repaired is None:
                _log_invalid_action(
                    "repair_unavailable",
                    env,
                    candidate,
                    note={"attempt": attempt, "candidate": _action_to_list(candidate)},
                )
                break
            _log_invalid_action(
                "repair_attempt",
                env,
                repaired,
                note={"attempt": attempt + 1, "candidate": _action_to_list(repaired)},
            )
            candidate = np.asarray(repaired, dtype=int)
            current_mask = None
            attempt += 1
    raise AssertionError("invalid action (repair failed)") from last_error


__all__ = ["_step_with_repair"]
