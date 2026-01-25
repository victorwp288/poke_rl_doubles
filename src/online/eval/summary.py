import math
from typing import Any

import numpy as np


def _safe_rate(value: float, total: float) -> float:
    if total <= 0:
        return 0.0
    return float(value) / float(total)


def _summarize_array(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}


def _wilson95_ci(wins: int, total: int) -> dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0}
    z = 1.959963984540054
    n = float(total)
    phat = float(wins) / n
    denom = 1.0 + (z**2) / n
    center = phat + (z**2) / (2.0 * n)
    half = z * math.sqrt((phat * (1.0 - phat) + (z**2) / (4.0 * n)) / n)
    return {
        "low": max(0.0, (center - half) / denom),
        "high": min(1.0, (center + half) / denom),
    }


def _summarize_stats(stats: dict[str, Any]):
    rewards = stats.get("rewards", [])
    turns = stats.get("turns", [])
    summary = {
        "episodes": stats.get("episodes", 0),
        "valid_episodes": stats.get("battles", 0),
        "wins": stats.get("wins", 0),
        "losses": stats.get("losses", 0),
        "draws": stats.get("draws", 0),
        "win_rate": _safe_rate(stats.get("wins", 0), stats.get("battles", 0)),
        "reward": _summarize_array(rewards),
        "turns": _summarize_array(turns),
        "sanitized_action_count": stats.get("sanitized_action_count", 0),
        "repaired_action_count": stats.get("repaired_action_count", 0),
    }
    summary["win_rate_wilson95"] = _wilson95_ci(summary["wins"], summary["battles"])
    return summary


def _policy_summary_base():
    return {"per_opponent": {}, "overall": {}}


def _merge_policy_summary(policy_summary, opponent_label: str, summary):
    policy_summary["per_opponent"][opponent_label] = summary


def _merge_policy_overall(policy_summary: dict[str, Any]):
    overall = policy_summary["overall"]
    for opponent_stats in policy_summary["per_opponent"].values():
        for key, value in opponent_stats.items():
            if key in {"reward", "turns", "win_rate_wilson95"}:
                continue
            if isinstance(value, dict):
                continue
            overall.setdefault(key, 0)
            overall[key] += value


def _finalize_policy_summary(policy_summary: dict[str, Any]):
    _merge_policy_overall(policy_summary)
    overall = policy_summary["overall"]
    overall["win_rate"] = _safe_rate(overall.get("wins", 0), overall.get("valid_episodes", 0))
    overall["reward"] = _summarize_array(
        [
            reward
            for opponent_stats in policy_summary["per_opponent"].values()
            for reward in opponent_stats.get("reward", {}).get("values", [])
        ]
    )
    overall["turns"] = _summarize_array(
        [
            turns
            for opponent_stats in policy_summary["per_opponent"].values()
            for turns in opponent_stats.get("turns", {}).get("values", [])
        ]
    )


def _ensure_values(summary: dict[str, Any]):
    for policy_summary in summary.values():
        for opponent_stats in policy_summary.get("per_opponent", {}).values():
            reward = opponent_stats.get("reward", {})
            if "values" not in reward:
                reward["values"] = []
            turns = opponent_stats.get("turns", {})
            if "values" not in turns:
                turns["values"] = []


__all__ = [
    "_ensure_values",
    "_finalize_policy_summary",
    "_merge_policy_summary",
    "_policy_summary_base",
    "_safe_rate",
    "_summarize_array",
    "_summarize_stats",
    "_wilson95_ci",
]
