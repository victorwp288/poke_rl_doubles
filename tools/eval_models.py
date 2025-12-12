#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_env import AccountConfiguration  # noqa: E402
from poke_env.player.baselines import (  # noqa: E402
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from src.config import section  # noqa: E402
from src.core.env import action_space_size  # noqa: E402
from src.online.env import make_maskable_env  # noqa: E402
from src.online.init import load_maskable_policy  # noqa: E402
from tools.online import (  # noqa: E402
    PolicyOpponentPlayer,
    _ensure_server_available,
    _resolve_server_configuration,
    _unique_username,
)


@dataclass(frozen=True)
class PolicySpec:
    label: str
    path: Path


@dataclass(frozen=True)
class OpponentSpec:
    label: str
    kind: str
    policy_path: Path | None = None


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _parse_policy_arg(raw: str) -> PolicySpec:
    if "=" not in raw:
        raise ValueError("policy must be in label=path form")
    label, path_str = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("policy label cannot be empty")
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"policy not found: {path}")
    return PolicySpec(label=label, path=path)


def _resolve_settings(env_mode: str, overrides: dict[str, Any] | None) -> dict[str, Any]:
    online_cfg = section("online") or {}
    modes = online_cfg.get("modes", {})
    if env_mode not in modes:
        available = ", ".join(sorted(modes)) or "none"
        raise ValueError(f"unknown env_mode '{env_mode}' (available: {available})")
    settings = dict(modes[env_mode])
    # Apply base rewards + mode overrides exactly like training.
    base_rewards = dict(online_cfg.get("base_rewards", {}) or {})
    mode_overrides = dict(settings.get("rewards") or {})
    base_rewards.update(mode_overrides)
    settings["rewards"] = {k: float(v) for k, v in base_rewards.items()}

    # Fill common online defaults.
    settings["battle_format"] = online_cfg.get("battle_format", "gen9doublesou")
    settings["team_path"] = (
        settings.get("team_path")
        or online_cfg.get("team_path")
        or (section("imitation_collect") or {}).get("our_team_path")
    )
    settings["server_url"] = settings.get("server_url") or online_cfg.get(
        "server_url", "http://localhost:8000"
    )
    settings["rate_limit_per_second"] = settings.get(
        "rate_limit_per_second", online_cfg.get("rate_limit_per_second")
    )
    settings["console_log_interval_sec"] = float(
        settings.get("console_log_interval_sec", online_cfg.get("console_log_interval_sec", 5.0))
    )

    if overrides:
        merged = dict(settings)
        for key, value in overrides.items():
            if key == "rewards" and isinstance(value, dict):
                rewards = dict(merged.get("rewards", {}))
                rewards.update(value)
                merged["rewards"] = rewards
            else:
                merged[key] = value
        settings = merged
    return settings


def _read_team_text(team_path: str | Path) -> str:
    team_path = Path(team_path)
    if not team_path.is_absolute():
        team_path = ROOT / team_path
    if not team_path.exists():
        raise FileNotFoundError(f"team file not found: {team_path}")
    from src.utils.teambuilders import read_showdown_team

    text = read_showdown_team(team_path)
    if not text.strip():
        raise ValueError(f"team file {team_path} is empty")
    return text


def _step_delay(settings: dict[str, Any]) -> float:
    limit = settings.get("rate_limit_per_second")
    if limit is None:
        return 0.0
    limit = float(limit)
    if limit <= 0:
        return 0.0
    return 1.0 / max(limit, 1e-6)


def _make_opponent(
    spec: OpponentSpec, *, server_cfg, battle_format: str, act_size: int
):
    opp_account = AccountConfiguration(_unique_username("EvalOpp"), None)
    common = dict(
        account_configuration=opp_account,
        battle_format=battle_format,
        max_concurrent_battles=1,
        server_configuration=server_cfg,
    )
    if spec.kind == "simple":
        return SimpleHeuristicsPlayer(**common)
    if spec.kind == "maxbp":
        return MaxBasePowerPlayer(**common)
    if spec.kind == "random":
        return RandomPlayer(**common)
    if spec.kind == "policy":
        if spec.policy_path is None:
            raise ValueError("policy opponent requires policy_path")
        return PolicyOpponentPlayer(model_path=spec.policy_path, act_size=act_size, **common)
    raise ValueError(f"unknown opponent kind: {spec.kind}")


def _make_eval_env(opponent, settings: dict[str, Any], team_text: str, server_cfg):
    player_account = AccountConfiguration(_unique_username("EvalOur"), None)
    env = make_maskable_env(
        opponent=opponent,
        battle_format=settings["battle_format"],
        rewards=settings["rewards"],
        team=team_text,
        account_configuration1=player_account,
        server_configuration=server_cfg,
        step_delay=_step_delay(settings),
        console_log_mode="off",
        console_log_interval_sec=settings.get("console_log_interval_sec", 5.0),
    )
    return DummyVecEnv([lambda: env])


def _maybe_wrap_vecnormalize(venv, policy_path: Path):
    vec_path = policy_path.with_name(f"{policy_path.stem}_vecnorm.pkl")
    if vec_path.exists():
        wrapped = VecNormalize.load(str(vec_path), venv)
        wrapped.training = False
        wrapped.norm_reward = False
        return wrapped, vec_path
    return venv, None


def _find_mask_env(venv):
    current = venv
    for _ in range(8):
        if hasattr(current, "envs"):
            try:
                return current.envs[0]
            except Exception:
                pass
        if hasattr(current, "venv"):
            current = current.venv
            continue
        break
    return venv


def _float_reward(reward: Any) -> float:
    try:
        arr = np.asarray(reward)
        if arr.size:
            return float(arr.reshape(-1)[0])
    except Exception:
        pass
    try:
        return float(reward)
    except Exception:
        return 0.0


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }
    arr = np.asarray(values, dtype=float)
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return {
        "mean": float(arr.mean()),
        "std": std,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def _summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = [r.get("outcome", "unknown") for r in rows]
    counts = defaultdict(int)
    for o in outcomes:
        counts[str(o)] += 1
    valid_n = counts["win"] + counts["loss"] + counts["draw"]
    win_rate = counts["win"] / valid_n if valid_n else 0.0
    win_ci = _wilson_ci(counts["win"], valid_n) if valid_n else (0.0, 0.0)

    rewards = [float(r.get("total_reward", 0.0)) for r in rows]
    turns = [float(r.get("turns", 0.0)) for r in rows]

    numeric_fields: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        stats = r.get("final_stats") or {}
        if not isinstance(stats, dict):
            continue
        for k, v in stats.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                numeric_fields[k].append(float(v))

    stats_summary = {k: _numeric_summary(vs) for k, vs in sorted(numeric_fields.items())}

    return {
        "episodes": len(rows),
        "valid_episodes": valid_n,
        "outcome_counts": dict(counts),
        "win_rate": win_rate,
        "win_rate_wilson95": {"low": win_ci[0], "high": win_ci[1]},
        "reward": _numeric_summary(rewards),
        "turns": _numeric_summary(turns),
        "final_stats": stats_summary,
        "repaired_step_rate": (
            sum(int(r.get("repaired_steps", 0)) for r in rows)
            / max(1, sum(int(r.get("steps", 0)) for r in rows))
        ),
        "sanitized_step_rate": (
            sum(int(r.get("sanitized_steps", 0)) for r in rows)
            / max(1, sum(int(r.get("steps", 0)) for r in rows))
        ),
        "step_timeout_rate": (
            sum(int(r.get("step_timeout_steps", 0)) for r in rows)
            / max(1, sum(int(r.get("steps", 0)) for r in rows))
        ),
    }


def evaluate_policy(
    *,
    policy: PolicySpec,
    model,
    env,
    mask_env,
    opponent: OpponentSpec,
    episodes: int,
    battle_timeout_sec: float,
    max_steps: int,
    deterministic: bool,
    jsonl_handle,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ep in range(int(episodes)):
        seed_value = int(seed_offset + ep)
        random.seed(seed_value)
        np.random.seed(seed_value)
        try:
            env.seed(seed_value)
        except Exception:
            pass
        episode_start = time.monotonic()
        episode_timeout = False
        episode_error: str | None = None
        total_reward = 0.0
        steps = 0
        repaired_steps = 0
        sanitized_steps = 0
        step_timeout_steps = 0
        last_info: dict[str, Any] = {}
        last_stats: dict[str, Any] = {}

        try:
            # Stable-baselines VecEnv reset returns obs only.
            obs = env.reset()
        except Exception as exc:
            episode_error = f"reset_failed:{exc}"
            obs = None

        while True:
            if obs is None:
                break
            if battle_timeout_sec > 0 and (time.monotonic() - episode_start) >= battle_timeout_sec:
                episode_timeout = True
                break
            if max_steps > 0 and steps >= max_steps:
                episode_timeout = True
                break

            try:
                mask = mask_env.action_masks()
            except Exception:
                mask = None
            try:
                if mask is not None:
                    action, _ = model.predict(
                        obs, deterministic=deterministic, action_masks=mask
                    )
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)
            except TypeError:
                action, _ = model.predict(obs, deterministic=deterministic)

            try:
                obs, reward, done, info = env.step(action)
            except Exception as exc:
                episode_error = f"step_failed:{exc}"
                break

            total_reward += _float_reward(reward)
            done_flag = bool(np.asarray(done).reshape(-1)[0]) if np.asarray(done).size else bool(done)

            info0 = info[0] if isinstance(info, (list, tuple)) and info else info
            if isinstance(info0, dict):
                last_info = info0
                if info0.get("repaired_action"):
                    repaired_steps += 1
                if info0.get("sanitized_action"):
                    sanitized_steps += 1
                if info0.get("step_timeout"):
                    step_timeout_steps += 1
                stats = info0.get("battle_stats")
                if isinstance(stats, dict):
                    last_stats = stats
                    if stats.get("result") in {"win", "loss", "draw"}:
                        done_flag = True

            steps += 1
            if done_flag:
                break

        outcome = last_stats.get("result", "unknown") if last_stats else "unknown"
        if episode_timeout:
            outcome = "timeout"
        if episode_error is not None and not episode_timeout:
            outcome = "error"

        turns = int(last_stats.get("turn", steps)) if last_stats else int(steps)

        row = {
            "policy_label": policy.label,
            "policy_path": str(policy.path),
            "opponent_label": opponent.label,
            "opponent_kind": opponent.kind,
            "opponent_policy_path": str(opponent.policy_path) if opponent.policy_path else None,
            "episode": ep,
            "seed": seed_value,
            "outcome": outcome,
            "turns": turns,
            "steps": steps,
            "total_reward": float(total_reward),
            "repaired_steps": repaired_steps,
            "sanitized_steps": sanitized_steps,
            "step_timeout_steps": step_timeout_steps,
            "final_stats": last_stats,
            "error": episode_error,
            "timeout": episode_timeout,
        }
        rows.append(row)
        jsonl_handle.write(json.dumps(row) + "\n")
        jsonl_handle.flush()

        if (ep + 1) % max(1, episodes // 10) == 0 or outcome in {"timeout", "error"}:
            print(
                f"[eval_models] policy={policy.label} opp={opponent.label} "
                f"ep={ep + 1}/{episodes} outcome={outcome} reward={total_reward:.3f} "
                f"turns={turns}",
                flush=True,
            )
        # Best-effort reset after errors/timeouts to avoid stuck battles.
        if outcome in {"timeout", "error"}:
            try:
                env.reset()
            except Exception:
                pass

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Thoroughly evaluate multiple PPO policies against heuristic and policy opponents.\n"
            "Example:\n"
            "  python tools/eval_models.py \\\n"
            "    --policy scratch=outputs/models/maskable_ppo_scratch_best.zip \\\n"
            "    --policy warmstart=outputs/models/maskable_ppo_warmstart_best.zip \\\n"
            "    --episodes 500 --crossplay\n"
        )
    )
    parser.add_argument(
        "--policy",
        action="append",
        required=False,
        default=[],
        help="Policy spec in label=path form. Repeat to add multiple policies.",
    )
    parser.add_argument(
        "--summarize-only",
        type=Path,
        default=None,
        help="Path to an existing models_eval_*.jsonl to summarize without running battles.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Episodes per opponent per policy (default: 200).",
    )
    parser.add_argument(
        "--env-mode",
        type=str,
        default="scratch",
        help="Which online.modes entry to use for eval env settings (default: scratch).",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["simple", "maxbp", "random"],
        help="Opponent kinds to include from {simple,maxbp,random} (default: all three).",
    )
    parser.add_argument(
        "--no-heuristics",
        action="store_true",
        help="Skip heuristic opponents; only run policy opponents such as --crossplay/--mirror/--policy-opponent.",
    )
    parser.add_argument(
        "--policy-opponent",
        action="append",
        default=[],
        help="Add a fixed policy opponent in label=path form. Repeatable.",
    )
    parser.add_argument(
        "--crossplay",
        action="store_true",
        help="Also evaluate each policy vs every other policy as a policy opponent.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Also evaluate each policy vs itself as a policy opponent.",
    )
    parser.add_argument(
        "--battle-timeout",
        type=float,
        default=180.0,
        help="Per-battle wall-clock timeout in seconds (default: 180). 0 disables.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Max environment steps per battle (default: 0 = no cap).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic for evaluation.",
    )
    parser.add_argument(
        "--override",
        type=str,
        action="append",
        help="Override env settings in key=value format (e.g., --override rate_limit_per_second=2).",
    )
    args = parser.parse_args()
    if args.summarize_only is None and not args.policy:
        parser.error("--policy is required unless --summarize-only is set.")

    # Summarize-only path: skip battles and recompute summaries from JSONL.
    if args.summarize_only is not None:
        source_path = Path(args.summarize_only).expanduser()
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        if not source_path.exists():
            raise FileNotFoundError(f"jsonl not found: {source_path}")
        rows: list[dict[str, Any]] = []
        with source_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue

        per_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        policy_paths: dict[str, str] = {}
        opponent_map: dict[str, OpponentSpec] = {}
        for r in rows:
            pl = str(r.get("policy_label", "unknown"))
            opp_label = str(r.get("opponent_label", "unknown"))
            per_bucket[(pl, opp_label)].append(r)
            if pl not in policy_paths and r.get("policy_path"):
                policy_paths[pl] = str(r["policy_path"])
            if opp_label not in opponent_map:
                kind = str(r.get("opponent_kind", "unknown"))
                opp_policy = r.get("opponent_policy_path")
                opp_path = Path(opp_policy) if opp_policy else None
                opponent_map[opp_label] = OpponentSpec(
                    label=opp_label, kind=kind, policy_path=opp_path
                )

        policies = [
            PolicySpec(label=label, path=Path(path_str))
            for label, path_str in policy_paths.items()
        ]
        opponent_specs = list(opponent_map.values())

        policies_summary: dict[str, Any] = {}
        for policy in policies:
            per_opp_summary = {}
            overall_rows = []
            for opp in opponent_specs:
                bucket_rows = per_bucket.get((policy.label, opp.label), [])
                per_opp_summary[opp.label] = _summarize_bucket(bucket_rows)
                overall_rows.extend(bucket_rows)
            policies_summary[policy.label] = {
                "path": str(policy.path),
                "vecnormalize_path": None,
                "per_opponent": per_opp_summary,
                "overall": _summarize_bucket(overall_rows),
            }

        out_dir = Path("outputs") / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = _utc_timestamp()
        summary_path = out_dir / f"models_eval_{ts}_summary.json"
        csv_path = out_dir / f"models_eval_{ts}_summary.csv"

        bucket_sizes = [len(v) for v in per_bucket.values()]
        episodes_per_opponent = int(np.median(bucket_sizes)) if bucket_sizes else 0
        deterministic = not args.stochastic

        summary = {
            "timestamp_utc": ts,
            "git_commit": _git_commit(),
            "source_jsonl": str(source_path),
            "recomputed_from_jsonl": True,
            "env_mode": args.env_mode,
            "episodes_per_opponent": episodes_per_opponent,
            "deterministic": deterministic,
            "opponents": [
                {
                    "label": spec.label,
                    "kind": spec.kind,
                    "policy_path": str(spec.policy_path) if spec.policy_path else None,
                }
                for spec in opponent_specs
            ],
            "policies": policies_summary,
            "jsonl_path": str(source_path),
            "csv_summary_path": str(csv_path),
        }

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
            for policy_label, policy_data in policies_summary.items():
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
        return

    policies = [_parse_policy_arg(p) for p in args.policy]
    policy_opponents = [_parse_policy_arg(p) for p in args.policy_opponent]

    overrides: dict[str, Any] = {}
    if args.override:
        for pair in args.override:
            if "=" not in pair:
                raise ValueError(f"invalid override '{pair}' (expected key=value)")
            k, v = pair.split("=", 1)
            try:
                v_eval = eval(v)
            except Exception:
                v_eval = v
            overrides[k] = v_eval

    settings = _resolve_settings(args.env_mode, overrides or None)
    team_text = _read_team_text(settings["team_path"])
    server_cfg = _resolve_server_configuration(settings.get("server_url"))
    _ensure_server_available(server_cfg)

    act_size = action_space_size(settings["battle_format"])

    opponent_specs: list[OpponentSpec] = []
    if not args.no_heuristics:
        for kind in args.opponents:
            token = str(kind).strip().lower()
            if token not in {"simple", "maxbp", "random"}:
                raise ValueError(f"unknown heuristic opponent kind: {token}")
            opponent_specs.append(OpponentSpec(label=token, kind=token))

    for p in policy_opponents:
        opponent_specs.append(
            OpponentSpec(label=f"policy_{p.label}", kind="policy", policy_path=p.path)
        )

    if args.crossplay and len(policies) > 1:
        for p in policies:
            opponent_specs.append(
                OpponentSpec(label=f"policy_{p.label}", kind="policy", policy_path=p.path)
            )

    if args.mirror:
        for p in policies:
            opponent_specs.append(
                OpponentSpec(label=f"mirror_{p.label}", kind="policy", policy_path=p.path)
            )

    # Deduplicate opponents by label.
    seen = set()
    unique_opponents: list[OpponentSpec] = []
    for spec in opponent_specs:
        if spec.label in seen:
            continue
        seen.add(spec.label)
        unique_opponents.append(spec)
    opponent_specs = unique_opponents

    out_dir = Path("outputs") / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _utc_timestamp()
    jsonl_path = out_dir / f"models_eval_{ts}.jsonl"
    summary_path = out_dir / f"models_eval_{ts}_summary.json"
    csv_path = out_dir / f"models_eval_{ts}_summary.csv"

    deterministic = not args.stochastic
    print(
        f"[eval_models] starting eval env_mode={args.env_mode} "
        f"episodes_per_opponent={args.episodes} opponents={[o.label for o in opponent_specs]} "
        f"policies={[p.label for p in policies]}",
        flush=True,
    )
    print(f"[eval_models] writing per-episode JSONL to {jsonl_path}", flush=True)

    all_rows: list[dict[str, Any]] = []
    per_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    vecnorm_map: dict[str, str | None] = {}

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for opp in opponent_specs:
            opponent_obj = _make_opponent(
                opp,
                server_cfg=server_cfg,
                battle_format=settings["battle_format"],
                act_size=act_size,
            )
            env_raw = _make_eval_env(opponent_obj, settings, team_text, server_cfg)
            try:
                for policy in policies:
                    # If crossplay added the policy as an opponent, skip self-play unless --mirror.
                    if (
                        opp.kind == "policy"
                        and opp.policy_path is not None
                        and opp.label.startswith("policy_")
                        and opp.policy_path.resolve() == policy.path.resolve()
                        and not args.mirror
                    ):
                        continue
                    env, vec_path = _maybe_wrap_vecnormalize(env_raw, policy.path)
                    vecnorm_map[policy.label] = str(vec_path) if vec_path else None
                    mask_env = _find_mask_env(env)
                    model = load_maskable_policy(policy.path, device="cpu")
                    rows = evaluate_policy(
                        policy=policy,
                        model=model,
                        env=env,
                        mask_env=mask_env,
                        opponent=opp,
                        episodes=args.episodes,
                        battle_timeout_sec=float(args.battle_timeout),
                        max_steps=int(args.max_steps),
                        deterministic=deterministic,
                        jsonl_handle=jf,
                        seed_offset=0,
                    )
                    all_rows.extend(rows)
                    per_bucket[(policy.label, opp.label)].extend(rows)
            finally:
                try:
                    env_raw.close()
                except Exception:
                    pass

    # Summaries
    policies_summary: dict[str, Any] = {}
    for policy in policies:
        per_opp_summary = {}
        overall_rows = []
        for opp in opponent_specs:
            rows = per_bucket.get((policy.label, opp.label), [])
            per_opp_summary[opp.label] = _summarize_bucket(rows)
            overall_rows.extend(rows)
        policies_summary[policy.label] = {
            "path": str(policy.path),
            "vecnormalize_path": vecnorm_map.get(policy.label),
            "per_opponent": per_opp_summary,
            "overall": _summarize_bucket(overall_rows),
        }

    summary = {
        "timestamp_utc": ts,
        "git_commit": _git_commit(),
        "env_mode": args.env_mode,
        "env_settings": {
            "battle_format": settings["battle_format"],
            "team_path": str(settings["team_path"]),
            "server_url": str(settings.get("server_url")),
            "rate_limit_per_second": settings.get("rate_limit_per_second"),
            "rewards": settings.get("rewards"),
        },
        "episodes_per_opponent": args.episodes,
        "battle_timeout_sec": float(args.battle_timeout),
        "max_steps": int(args.max_steps),
        "deterministic": deterministic,
        "opponents": [
            {
                "label": spec.label,
                "kind": spec.kind,
                "policy_path": str(spec.policy_path) if spec.policy_path else None,
            }
            for spec in opponent_specs
        ],
        "policies": policies_summary,
        "jsonl_path": str(jsonl_path),
        "csv_summary_path": str(csv_path),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Lightweight CSV for top-line comparison.
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
        for policy_label, policy_data in policies_summary.items():
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


if __name__ == "__main__":
    main()
