from __future__ import annotations

from contextlib import suppress
from pathlib import Path

from src.config import section
from src.utils.teambuilders import read_showdown_team

ROOT = Path(__file__).resolve().parents[2]


def parse_override_pairs(pairs):
    overrides = {}
    if not pairs:
        return overrides
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"invalid override '{pair}'. Expected key=value format")
        key, value = pair.split("=", 1)
        with suppress(Exception):
            value = eval(value)
        overrides[key] = value
    return overrides


def _team_details(settings):
    team_path = settings.get("team_path")
    if not team_path:
        raise ValueError(
            "online.team_path must be configured (or set imitation_collect.our_team_path)"
        )
    path = Path(team_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"team file not found: {path}")
    team_text = read_showdown_team(path)
    if not team_text or not team_text.strip():
        raise ValueError(f"team file {path} is empty")
    return path, team_text


def _mode_settings(name):
    config = section("online")
    offline = section("offline")
    modes = config.get("modes", {})
    if name not in modes:
        available = ", ".join(sorted(modes)) or "none"
        raise ValueError(f"unknown mode '{name}' (available: {available})")
    settings = dict(modes[name])
    rewards = dict(config.get("base_rewards", {}))
    overrides = settings.get("rewards") or {}
    for key, value in overrides.items():
        rewards[key] = value
    settings["rewards"] = {key: float(value) for key, value in rewards.items()}
    settings["battle_format"] = config.get("battle_format", "gen9doublesou")
    settings["total_timesteps"] = int(settings.get("total_timesteps", 0))
    settings["learning_rate"] = float(settings.get("learning_rate", 3e-4))
    settings["n_steps"] = int(settings.get("n_steps", 1024))
    settings["batch_size"] = int(settings.get("batch_size", 256))
    settings["parallel_battles"] = max(1, int(settings.get("parallel_battles", 1)))
    settings["eval_freq"] = int(settings.get("eval_freq", 0))
    settings["eval_episodes"] = int(settings.get("eval_episodes", 0))
    settings["checkpoint_freq"] = int(settings.get("checkpoint_freq", 0))
    settings["entropy_coef"] = float(
        settings.get("entropy_coef", config.get("entropy_coef", 0.0) or 0.0)
    )
    settings["kl_coef_start"] = float(
        settings.get("kl_coef_start", config.get("kl_coef_start", 0.0) or 0.0)
    )
    settings["kl_coef_final"] = float(
        settings.get("kl_coef_final", config.get("kl_coef_final", 0.0) or 0.0)
    )
    settings["kl_anneal_steps"] = int(
        settings.get("kl_anneal_steps", config.get("kl_anneal_steps", 0))
    )
    target_kl_value = settings.get("target_kl", config.get("target_kl"))
    settings["target_kl"] = float(target_kl_value) if target_kl_value is not None else None
    settings["log_interval"] = int(settings.get("log_interval", 2048))
    settings["console_log_mode"] = settings.get(
        "console_log_mode", config.get("console_log_mode", "off")
    )
    settings["console_log_interval_sec"] = float(
        settings.get(
            "console_log_interval_sec",
            config.get("console_log_interval_sec", 2.0),
        )
    )
    settings["load_bc"] = bool(settings.get("load_bc", False))
    settings["curriculum_note"] = settings.get("curriculum_note")
    settings["tensorboard_dir"] = Path(
        settings.get("tensorboard_dir", "outputs/tensorboard/online")
    )
    settings["policy_path"] = Path(settings.get("policy_path", "outputs/models/maskable_ppo.zip"))
    best_path = settings.get("best_policy_path")
    settings["best_policy_path"] = (
        Path(best_path)
        if best_path
        else settings["policy_path"].with_name(
            f"{settings['policy_path'].stem}_best{settings['policy_path'].suffix}"
        )
    )
    if settings["load_bc"]:
        settings["bc_checkpoint"] = Path(
            settings.get("bc_checkpoint", "outputs/models/bc_policy.pt")
        )
    else:
        settings["bc_checkpoint"] = None
    hidden_dim = (
        settings.get("policy_hidden_dim")
        or config.get("policy_hidden_dim")
        or offline.get("hidden_dim")
        or offline.get("model_hidden_dim")
        or 512
    )
    hidden_layers = (
        settings.get("policy_hidden_layers")
        or config.get("policy_hidden_layers")
        or offline.get("hidden_layers")
        or offline.get("model_hidden_layers")
        or 2
    )
    settings["policy_hidden_dim"] = int(hidden_dim)
    settings["policy_hidden_layers"] = int(hidden_layers)
    head_mlp_dim = (
        settings.get("policy_head_mlp_dim")
        or config.get("policy_head_mlp_dim")
        or offline.get("head_mlp_dim")
        or 512
    )
    settings["policy_head_mlp_dim"] = int(head_mlp_dim)
    keep_last = settings.get("checkpoint_keep_last")
    settings["checkpoint_keep_last"] = int(keep_last) if keep_last is not None else 1
    collect_cfg = section("imitation_collect")
    team_path = (
        settings.get("team_path")
        or config.get("team_path")
        or (collect_cfg.get("our_team_path") if isinstance(collect_cfg, dict) else None)
    )
    settings["team_path"] = team_path
    server_url = (
        settings.get("server_url")
        or config.get("server_url")
        or (collect_cfg.get("server_url") if isinstance(collect_cfg, dict) else None)
    )
    settings["server_url"] = server_url
    rate_limit = settings.get("rate_limit_per_second", config.get("rate_limit_per_second"))
    if rate_limit is None:
        settings["rate_limit_per_second"] = None
    else:
        settings["rate_limit_per_second"] = max(float(rate_limit), 0.0)
    settings["player_name_prefix"] = (
        settings.get("player_name_prefix") or config.get("player_name_prefix") or "RLAgent"
    )
    settings["opponent_name_prefix"] = (
        settings.get("opponent_name_prefix")
        or config.get("opponent_name_prefix")
        or "SimpleHeuristics"
    )
    opp_kinds = settings.get("opponent_kinds") or config.get("opponent_kinds")
    if opp_kinds:
        settings["opponent_kinds"] = opp_kinds
    schedule = settings.get("opponent_schedule") or config.get("opponent_schedule")
    if schedule:
        settings["opponent_schedule"] = schedule
    return settings


def _apply_overrides(settings, overrides):
    if not overrides:
        return dict(settings)
    merged = dict(settings)
    for key, value in overrides.items():
        if key == "rewards" and isinstance(value, dict):
            rewards = dict(merged.get("rewards", {}))
            rewards.update(value)
            merged["rewards"] = rewards
        else:
            merged[key] = value
    for path_key in ("policy_path", "best_policy_path", "bc_checkpoint", "tensorboard_dir"):
        if (
            path_key in merged
            and merged[path_key] is not None
            and not isinstance(merged[path_key], Path)
        ):
            merged[path_key] = Path(merged[path_key])
    return merged


__all__ = ["_apply_overrides", "_mode_settings", "_team_details", "parse_override_pairs"]
