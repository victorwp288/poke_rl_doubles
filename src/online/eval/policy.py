from pathlib import Path

from src.online.policy.load import load_maskable_policy

from .specs import PolicySpec


def _load_policy(policy_path: Path, venv=None, act_size: int | None = None):
    _ = venv, act_size
    return load_maskable_policy(policy_path)


def _load_policies(policies: list[PolicySpec], venv, act_size: int):
    loaded = []
    for spec in policies:
        loaded.append((spec, _load_policy(spec.path, venv, act_size)))
    return loaded


def _policy_action(policy, obs):
    action, _ = policy.predict(obs, deterministic=True)
    return action


__all__ = ["_load_policies", "_load_policy", "_policy_action"]
