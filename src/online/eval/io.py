import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .paths import _git_commit, _utc_timestamp
from .specs import OpponentSpec, PolicySpec
from .summary import _ensure_values, _policy_summary_base, _summarize_stats


def _write_jsonl(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _write_run_config(
    *,
    root: Path,
    out_dir: Path,
    env_mode: str,
    policies: list[PolicySpec],
    opponents: list[OpponentSpec],
    overrides: dict[str, Any],
    settings: dict[str, Any],
    notes: str | None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": _utc_timestamp(),
        "env_mode": env_mode,
        "policies": [{"label": p.label, "path": str(p.path)} for p in policies],
        "opponents": [
            {
                "label": o.label,
                "kind": o.kind,
                "policy_path": str(o.policy_path) if o.policy_path else None,
            }
            for o in opponents
        ],
        "overrides": overrides or {},
        "settings": settings,
        "notes": notes,
        "git_commit": _git_commit(root),
    }
    (out_dir / "run_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_eval_dir(root: Path, base_dir: Path | None):
    base_dir = base_dir or (root / "outputs" / "eval")
    stamp = _utc_timestamp()
    out_dir = base_dir / f"eval_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _load_existing_summary(out_dir: Path):
    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _recompute_summaries(out_dir: Path, *, output_dir: Path | None):
    jsonl_path = out_dir / "episodes.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"missing per-episode file: {jsonl_path}")

    output_dir = output_dir or out_dir
    print(f"[eval_models] recompute summary for {jsonl_path}", flush=True)
    per_episode = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            per_episode.append(json.loads(line))

    summary: defaultdict[str, dict[str, Any]] = defaultdict(_policy_summary_base)
    for episode in per_episode:
        policy_label = episode.get("policy")
        opponent_label = episode.get("opponent")
        if not policy_label or not opponent_label:
            continue
        stats = {
            "episodes": len(episode.get("rewards", [])),
            "battles": len(episode.get("rewards", [])),
            "wins": episode.get("wins", 0),
            "losses": episode.get("losses", 0),
            "draws": episode.get("draws", 0),
            "rewards": episode.get("rewards", []),
            "turns": episode.get("turns", []),
            "sanitized_action_count": episode.get("sanitized_action_count", 0),
            "repaired_action_count": episode.get("repaired_action_count", 0),
        }
        summary[policy_label]["per_opponent"][opponent_label] = _summarize_stats(stats)

    _ensure_values(summary)
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(
            [
                "policy",
                "opponent",
                "episodes",
                "valid_episodes",
                "wins",
                "losses",
                "draws",
                "win_rate",
                "win_rate_ci_low",
                "win_rate_ci_high",
                "reward_mean",
                "reward_std",
                "turns_mean",
                "turns_std",
            ]
        )
        for policy_label, policy_data in summary.items():
            for opp_label, opp_data in policy_data["per_opponent"].items():
                counts = opp_data.get("outcome_counts", {})
                reward_summary = opp_data.get("reward", {})
                turns_summary = opp_data.get("turns", {})
                ci = opp_data.get("win_rate_wilson95", {})
                w.writerow(
                    [
                        policy_label,
                        opp_label,
                        opp_data.get("episodes", 0),
                        opp_data.get("valid_episodes", 0),
                        counts.get("win", 0),
                        counts.get("loss", 0),
                        counts.get("draw", 0),
                        opp_data.get("win_rate", 0.0),
                        ci.get("low", 0.0),
                        ci.get("high", 0.0),
                        reward_summary.get("mean", 0.0),
                        reward_summary.get("std", 0.0),
                        turns_summary.get("mean", 0.0),
                        turns_summary.get("std", 0.0),
                    ]
                )

    print(f"[eval_models] recomputed summary JSON to {summary_path}", flush=True)
    print(f"[eval_models] recomputed summary CSV  to {csv_path}", flush=True)


__all__ = [
    "_build_eval_dir",
    "_load_existing_summary",
    "_recompute_summaries",
    "_write_jsonl",
    "_write_run_config",
]
