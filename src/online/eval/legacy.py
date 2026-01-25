from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.core.env import action_space_size
from src.online.environment import _ensure_server_available, _resolve_server_configuration

from .episodes import _collect_episode
from .io import _build_eval_dir, _recompute_summaries, _write_run_config
from .parse import _ensure_override_pairs, _parse_opponents
from .settings import _parse_notes, _read_team_text
from .specs import OpponentSpec, PolicySpec
from .summary import _ensure_values, _policy_summary_base, _summarize_stats


def _check_summary_inputs(
    *,
    out_dir: Path,
    policies: list[PolicySpec],
    opponents: list[OpponentSpec],
    overrides: dict[str, Any] | None,
    settings: dict[str, Any],
):
    if not policies:
        raise ValueError("summary mode requires at least one --summary-policy")
    if not opponents:
        raise ValueError("summary mode requires at least one --summary-opponent")
    if out_dir is None:
        raise ValueError("summary mode requires --output-dir")

    for policy in policies:
        if not policy.path.exists():
            raise FileNotFoundError(f"summary policy path not found: {policy.path}")

    for opponent in opponents:
        if opponent.kind == "policy" and opponent.policy_path is None:
            raise ValueError("summary opponent policy missing policy_path")

    for key in ("battle_format", "team_path", "server_url"):
        if not settings.get(key):
            raise ValueError(f"summary requires setting: {key}")


def _run_single_eval(
    *,
    policy: PolicySpec,
    opponent: OpponentSpec,
    settings: dict[str, Any],
    overrides: dict[str, Any] | None,
    server_cfg,
    act_size: int,
    max_episodes: int,
    team_text: str,
    per_episode_path: Path | None,
    replays_path: Path | None,
    render: bool,
    step_sleep: float,
):
    stats = _collect_episode(
        policy=policy,
        policy_model=None,
        opponent=opponent,
        settings=settings,
        overrides=overrides,
        server_cfg=server_cfg,
        act_size=act_size,
        max_steps=max_episodes,
        team_text=team_text,
        per_episode_path=per_episode_path,
        replays_path=replays_path,
        render=render,
        step_sleep=step_sleep,
    )

    summary = _summarize_stats(stats)
    summary["outcome_counts"] = {
        "win": summary.get("wins", 0),
        "loss": summary.get("losses", 0),
        "draw": summary.get("draws", 0),
    }
    summary["reward"]["values"] = stats.get("rewards", [])
    summary["turns"]["values"] = stats.get("turns", [])
    return summary


def _run_eval_mode(
    *,
    root: Path,
    args,
    policies: list[PolicySpec],
    opponents: list[OpponentSpec],
    env_mode: str,
    settings: dict[str, Any],
    overrides: dict[str, Any],
):
    _ensure_override_pairs(args.override, args.summary_overrides)
    server_cfg = _resolve_server_configuration(settings.get("server_url"))
    if server_cfg is None:
        raise ValueError("could not resolve server configuration")
    _ensure_server_available(server_cfg)
    act_size = action_space_size(settings.get("battle_format", "gen9doublesou"))
    team_path = settings.get("team_path")
    if not team_path:
        raise ValueError("summary requires setting: team_path")
    team_text = _read_team_text(root, team_path)

    if args.summary_only:
        _check_summary_inputs(
            out_dir=Path(args.output_dir),
            policies=policies,
            opponents=opponents,
            overrides=overrides,
            settings=settings,
        )
        _recompute_summaries(Path(args.output_dir), output_dir=Path(args.output_dir))
        return

    out_dir = _build_eval_dir(root, Path(args.output_dir) if args.output_dir else None)
    jsonl_path = out_dir / "episodes.jsonl"
    csv_path = out_dir / "summary.csv"
    summary_path = out_dir / "summary.json"

    _write_run_config(
        root=root,
        out_dir=out_dir,
        env_mode=env_mode,
        policies=policies,
        opponents=opponents,
        overrides=overrides,
        settings=settings,
        notes=_parse_notes(args.notes),
    )

    print(
        f"[eval_models] starting eval env_mode={args.env_mode} "
        f"episodes={args.episodes} policies={len(policies)} opponents={len(opponents)} "
        f"output_dir={out_dir}",
        flush=True,
    )
    print(f"[eval_models] writing per-episode JSONL to {jsonl_path}", flush=True)

    summary: defaultdict[str, dict[str, Any]] = defaultdict(_policy_summary_base)

    for policy in policies:
        for opponent in opponents:
            try:
                summary_stats = _run_single_eval(
                    policy=policy,
                    opponent=opponent,
                    settings=settings,
                    overrides=overrides,
                    server_cfg=server_cfg,
                    act_size=act_size,
                    max_episodes=args.episodes,
                    team_text=team_text,
                    per_episode_path=jsonl_path,
                    replays_path=Path(args.replays_dir) if args.replays_dir else None,
                    render=args.render,
                    step_sleep=float(args.sleep or 0.0),
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(
                    f"[eval_models] policy={policy.label} opp={opponent.label} error={exc}",
                    flush=True,
                )
                summary_stats = {
                    "episodes": 0,
                    "valid_episodes": 0,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                    "win_rate": 0.0,
                    "reward": {"mean": 0.0, "std": 0.0, "values": []},
                    "turns": {"mean": 0.0, "std": 0.0, "values": []},
                    "win_rate_wilson95": {"low": 0.0, "high": 0.0},
                    "sanitized_action_count": 0,
                    "repaired_action_count": 0,
                }

            summary[policy.label]["per_opponent"][opponent.label] = summary_stats

    _ensure_values(summary)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        import csv

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

    print(f"[eval_models] wrote summary JSON to {summary_path}", flush=True)
    print(f"[eval_models] wrote summary CSV  to {csv_path}", flush=True)


def run_legacy_eval(*, root: Path, args, policies, env_mode, settings, overrides):
    opponents = _parse_opponents(args, policies)
    _run_eval_mode(
        root=root,
        args=args,
        policies=policies,
        opponents=opponents,
        env_mode=env_mode,
        settings=settings,
        overrides=overrides,
    )


__all__ = ["run_legacy_eval", "_check_summary_inputs"]
