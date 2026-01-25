from pathlib import Path
from typing import Any

from src.config import section


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


def _read_team_text(root: Path, team_path: str | Path) -> str:
    team_path = Path(team_path)
    if not team_path.is_absolute():
        team_path = root / team_path
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


def _parse_override_pairs(pairs: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if not pairs:
        return overrides
    for entry in pairs:
        if "=" not in entry:
            raise ValueError(f"override must be key=value (got {entry})")
        key, value = entry.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"override must be key=value (got {entry})")
        if "." in key:
            root, child = key.split(".", 1)
            if root == "rewards":
                rewards = overrides.get("rewards")
                if not isinstance(rewards, dict):
                    rewards = {}
                    overrides["rewards"] = rewards
                rewards[child] = float(value)
                continue
        try:
            if "." in value:
                overrides[key] = float(value)
            else:
                overrides[key] = int(value)
            continue
        except Exception:
            pass
        if value.lower() in {"true", "false"}:
            overrides[key] = value.lower() == "true"
            continue
        overrides[key] = value
    return overrides


def _parse_notes(arg: str | None):
    if arg is None:
        return None
    cleaned = arg.strip()
    return cleaned if cleaned else None


__all__ = [
    "_parse_notes",
    "_parse_override_pairs",
    "_read_team_text",
    "_resolve_settings",
    "_step_delay",
]
