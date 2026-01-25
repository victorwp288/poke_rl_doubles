from pathlib import Path
from typing import Any

from .settings import _parse_override_pairs, _resolve_settings
from .specs import OpponentSpec, PolicySpec


def _parse_policy_arg(root: Path, raw: str) -> PolicySpec:
    if "=" not in raw:
        raise ValueError("policy must be in label=path form")
    label, path_str = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("policy label cannot be empty")
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"policy not found: {path}")
    return PolicySpec(label=label, path=path)


def _split_policy_args(root: Path, args):
    policies = []
    for raw in args.policy:
        policies.append(_parse_policy_arg(root, raw))
    return policies


def _parse_summary_policy_arg(root: Path, arg: str) -> PolicySpec:
    if "=" not in arg:
        raise ValueError("summary policy arg must be label=path")
    label, path_str = arg.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("summary policy arg must have a label")
    path = Path(path_str)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(f"summary policy path not found: {path}")
    return PolicySpec(label=label, path=path)


def _parse_summary_opponent_arg(arg: str) -> OpponentSpec:
    if "=" not in arg:
        raise ValueError("summary opponent arg must be label=kind")
    label, kind = arg.split("=", 1)
    label = label.strip()
    kind = kind.strip()
    if not label:
        raise ValueError("summary opponent arg must have a label")
    if kind not in {"simple", "maxbp", "random", "policy"}:
        raise ValueError("summary opponent kind must be simple|maxbp|random|policy")
    return OpponentSpec(label=label, kind=kind)


def _resolve_summary_policy(root: Path, policy_map: dict[str, PolicySpec], args_policy: list[str]):
    policies = []
    for entry in args_policy or []:
        spec = _parse_summary_policy_arg(root, entry)
        policies.append(spec)
        policy_map[spec.label] = spec
    return policies


def _resolve_summary_opponents(opponent_map: dict[str, OpponentSpec], args_opponents: list[str]):
    opponents = []
    for entry in args_opponents or []:
        spec = _parse_summary_opponent_arg(entry)
        opponents.append(spec)
        opponent_map[spec.label] = spec
    return opponents


def _resolve_summary_args(root: Path, args):
    policies = _resolve_summary_policy(root, {}, args.summary_policy)
    opponents = _resolve_summary_opponents({}, args.summary_opponent)
    env_mode = args.env_mode
    if env_mode is None:
        raise ValueError("--env-mode is required for summary mode")
    settings = _resolve_settings(env_mode, overrides=None)
    overrides = None
    if args.summary_overrides:
        overrides = _parse_override_pairs(args.summary_overrides)
    return policies, opponents, env_mode, settings, overrides


def _parse_opponents(args, policies: list[PolicySpec]):
    opponent_specs: list[OpponentSpec] = []
    if args.opponents:
        for token in args.opponents:
            token = token.strip().lower()
            if token in {"simple", "maxbp", "random"}:
                opponent_specs.append(OpponentSpec(label=token, kind=token))
            else:
                raise ValueError(f"unknown opponent: {token}")

    if args.crossplay:
        for p in policies:
            opponent_specs.append(
                OpponentSpec(label=f"policy_{p.label}", kind="policy", policy_path=p.path)
            )

    if args.mirror:
        for p in policies:
            opponent_specs.append(
                OpponentSpec(label=f"policy_{p.label}", kind="policy", policy_path=p.path)
            )
            opponent_specs.append(
                OpponentSpec(label=f"mirror_{p.label}", kind="policy", policy_path=p.path)
            )

    unique_opponents: list[OpponentSpec] = []
    seen = set()
    for spec in opponent_specs:
        if spec.label in seen:
            continue
        seen.add(spec.label)
        unique_opponents.append(spec)
    return unique_opponents


def _ensure_override_pairs(overrides: list[str] | None, summary_overrides: list[str] | None):
    for entry in overrides or []:
        if "=" not in entry:
            raise ValueError(f"override must be key=value (got {entry})")
    for entry in summary_overrides or []:
        if "=" not in entry:
            raise ValueError(f"summary override must be key=value (got {entry})")


def _parse_overrides_from_args(args) -> dict[str, Any]:
    overrides = _parse_override_pairs(args.override)
    if args.summary_overrides:
        overrides.update(_parse_override_pairs(args.summary_overrides))
    return overrides


__all__ = [
    "_ensure_override_pairs",
    "_parse_opponents",
    "_parse_overrides_from_args",
    "_resolve_summary_args",
    "_split_policy_args",
]
