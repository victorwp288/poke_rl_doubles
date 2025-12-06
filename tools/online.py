#!/usr/bin/env python3
import argparse
import contextlib
import shutil
import socket
import sys
import uuid
from collections import deque
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from poke_env import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ServerConfiguration,
    ShowdownServerConfiguration,
)
from poke_env.player.baselines import SimpleHeuristicsPlayer
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import section  # noqa: E402
from src.online import load_behavior_clone_weights, make_maskable_env  # noqa: E402
from src.online.kl_ppo import KLRegularizedMaskablePPO  # noqa: E402
from src.utils.teambuilders import read_showdown_team  # noqa: E402


class RollingCheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_freq,
        save_path,
        name_prefix,
        keep_last=1,
        fixed_filename=None,
    ):
        super().__init__()
        self.save_freq = max(int(save_freq), 0)
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.keep_last = max(int(keep_last), 1)
        self.fixed_filename = fixed_filename
        self._next_save = 0
        self._saved_paths: deque[Path] = deque()

    def _on_training_start(self) -> None:
        if self.save_freq == 0:
            return
        self.save_path.mkdir(parents=True, exist_ok=True)
        self._next_save = self.num_timesteps + self.save_freq

    def _remove_excess(self) -> None:
        while len(self._saved_paths) > self.keep_last:
            old_path = self._saved_paths.popleft()
            with contextlib.suppress(FileNotFoundError):
                old_path.unlink()

    def _save_checkpoint(self) -> None:
        filename = f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
        target = self.save_path / filename
        self.model.save(str(target))
        self._saved_paths.append(target)
        self._remove_excess()
        if self.fixed_filename:
            latest_path = self.save_path / self.fixed_filename
            shutil.copy(str(target), str(latest_path))

    def _on_step(self) -> bool:
        if self.save_freq == 0:
            return True
        if self.num_timesteps >= self._next_save:
            self._save_checkpoint()
            self._next_save += self.save_freq
        return True


class CopyBestModelCallback(BaseCallback):
    def __init__(self, source_dir: Path, destination: Path):
        super().__init__()
        self.source = Path(source_dir) / "best_model.zip"
        self.destination = Path(destination)
        self._last_copied_stamp: int | None = None

    def _init_callback(self) -> None:
        self.destination.parent.mkdir(parents=True, exist_ok=True)

    def _copy_if_updated(self) -> None:
        if not self.source.exists():
            return
        stamp = self.source.stat().st_mtime_ns
        if stamp == self._last_copied_stamp:
            return
        shutil.copy(self.source, self.destination)
        self._last_copied_stamp = stamp

    def _on_step(self) -> bool:
        self._copy_if_updated()
        return True


def _unique_username(prefix):
    token = (prefix or "bot")[:11]
    suffix = uuid.uuid4().hex[:6]
    return f"{token}{suffix}"


def _resolve_device(preferred=None):
    candidates: list[str] = []
    if isinstance(preferred, str) and preferred.strip():
        candidates.append(preferred.strip().lower())
    candidates.extend(["cuda", "mps", "cpu"])

    for name in candidates:
        if name.startswith("cuda"):
            if not torch.cuda.is_available():
                continue
            device = torch.device("cuda")
        elif name.startswith("mps"):
            if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
                continue
            if hasattr(torch.backends.mps, "is_built") and not torch.backends.mps.is_built():
                continue
            device = torch.device("mps")
        elif name.startswith("cpu"):
            device = torch.device("cpu")
        else:
            try:
                device = torch.device(name)
            except Exception:
                continue
        print(f"Using {device.type} device", flush=True)
        return device

    fallback = torch.device("cpu")
    print(f"Using {fallback.type} device", flush=True)
    return fallback


def _resolve_server_configuration(url):
    token = (url or "").strip()
    if not token:
        return LocalhostServerConfiguration
    lowered = token.lower()
    if lowered in {"showdown", "https://play.pokemonshowdown.com"}:
        return ShowdownServerConfiguration
    if lowered in {"local", "localhost", "http://localhost:8000"}:
        return LocalhostServerConfiguration
    websocket_url = token
    if websocket_url.startswith("http://"):
        websocket_url = "ws://" + websocket_url[len("http://") :]
    elif websocket_url.startswith("https://"):
        websocket_url = "wss://" + websocket_url[len("https://") :]
    elif not websocket_url.startswith(("ws://", "wss://")):
        websocket_url = f"ws://{websocket_url}"
    if not websocket_url.endswith("/websocket"):
        websocket_url = websocket_url.rstrip("/") + "/websocket"
    return ServerConfiguration(websocket_url, ShowdownServerConfiguration.authentication_url)


def _ensure_server_available(server_cfg):
    ws_url = getattr(server_cfg, "websocket_url", "")
    parsed = urlparse(ws_url)
    host = parsed.hostname
    if not host:
        return
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    try:
        with socket.create_connection((host, port), timeout=3):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"Could not reach Showdown server at {ws_url} "
            "(start the server or update online.server_url)."
        ) from exc


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


def _safe_close(env):
    if env is None:
        return
    with contextlib.suppress(AssertionError, RuntimeError):
        env.close()


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
    return settings


def _build_env(settings, *, team_text, server_cfg):
    player_prefix = settings.get("player_name_prefix", "RLAgent")
    opponent_prefix = settings.get("opponent_name_prefix", "SimpleHeuristics")

    def _make_env_fn():
        def _thunk():
            opponent_account = AccountConfiguration(_unique_username(opponent_prefix), None)
            opponent = SimpleHeuristicsPlayer(
                account_configuration=opponent_account,
                battle_format=settings["battle_format"],
                max_concurrent_battles=1,
                server_configuration=server_cfg,
            )
            player_account = AccountConfiguration(_unique_username(player_prefix), None)
            limit = settings["rate_limit_per_second"]
            step_delay = 0.0 if not limit else 1.0 / max(limit, 1e-6)

            return make_maskable_env(
                opponent=opponent,
                battle_format=settings["battle_format"],
                rewards=settings["rewards"],
                team=team_text,
                account_configuration1=player_account,
                server_configuration=server_cfg,
                step_delay=step_delay,
            )

        return _thunk

    env_fns = [_make_env_fn() for _ in range(settings["parallel_battles"])]
    if settings["parallel_battles"] > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def _load_bc_if_requested(model, settings):
    checkpoint = settings.get("bc_checkpoint")
    if not checkpoint:
        return
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"BC checkpoint not found: {checkpoint}")

    offline_config = section("offline")
    stats_path = offline_config.get("stats_path")
    stats_path = Path(stats_path) if stats_path else None
    try:
        stats = load_behavior_clone_weights(
            policy=model.policy,
            checkpoint_path=checkpoint,
            stats_path=stats_path,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load BC weights from {checkpoint}: {e}") from e

    if stats is not None:
        print(
            f"loaded behavior clone weights (count={stats.count} mean_dim={stats.mean.numel()})",
            flush=True,
        )


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
    return merged


def run(mode="scratch", overrides=None):
    settings = _apply_overrides(_mode_settings(mode), overrides)
    if settings["total_timesteps"] <= 0:
        raise ValueError("total_timesteps must be positive")

    seed = int(settings.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _resolve_device(settings.get("device"))
    settings["device"] = device

    settings["tensorboard_dir"].mkdir(parents=True, exist_ok=True)
    settings["policy_path"].parent.mkdir(parents=True, exist_ok=True)

    team_path, team_text = _team_details(settings)
    settings["team_path"] = team_path
    server_cfg = _resolve_server_configuration(settings.get("server_url"))
    _ensure_server_available(server_cfg)

    env = _build_env(settings, team_text=team_text, server_cfg=server_cfg)
    env.seed(seed)
    callbacks = []
    checkpoint_freq = settings.get("checkpoint_freq", 0)
    if checkpoint_freq > 0:
        checkpoint_dir = settings["policy_path"].parent / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_name = f"{settings['policy_path'].stem}_latest.zip"
        callbacks.append(
            RollingCheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=checkpoint_dir,
                name_prefix=settings["policy_path"].stem,
                keep_last=settings.get("checkpoint_keep_last", 1),
                fixed_filename=latest_name,
            )
        )

    eval_env = None
    best_dir = None
    eval_freq = settings.get("eval_freq", 0)
    if eval_freq > 0:
        eval_env = _build_env(
            {**settings, "parallel_battles": 1}, team_text=team_text, server_cfg=server_cfg
        )
        best_dir = settings["best_policy_path"].parent / f"{settings['policy_path'].stem}_best_tmp"
        best_dir.mkdir(parents=True, exist_ok=True)
        copy_best = CopyBestModelCallback(best_dir, settings["best_policy_path"])
        callbacks.append(
            EvalCallback(
                eval_env,
                callback_on_new_best=copy_best,
                best_model_save_path=str(best_dir),
                log_path=str(best_dir),
                eval_freq=eval_freq,
                n_eval_episodes=max(settings.get("eval_episodes", 1), 1),
                deterministic=True,
            )
        )

    entropy_coef = float(settings.get("entropy_coef", 0.0))
    kl_args = {
        "kl_coef_start": float(settings.get("kl_coef_start", 0.0)),
        "kl_coef_final": float(settings.get("kl_coef_final", 0.0)),
        "kl_anneal_steps": int(settings.get("kl_anneal_steps", 0)),
    }

    model = KLRegularizedMaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=settings["learning_rate"],
        n_steps=settings["n_steps"],
        batch_size=settings["batch_size"],
        policy_kwargs={
            "net_arch": {
                "pi": [settings["policy_hidden_dim"]] * settings["policy_hidden_layers"],
                "vf": [settings["policy_hidden_dim"]] * settings["policy_hidden_layers"],
            },
            "activation_fn": torch.nn.ReLU,
            "ortho_init": False,
        },
        tensorboard_log=str(settings["tensorboard_dir"]),
        verbose=1,
        device=device,
        ent_coef=entropy_coef,
        target_kl=settings.get("target_kl"),
        **kl_args,
    )

    _load_bc_if_requested(model, settings)

    if (
        kl_args["kl_coef_start"] > 0.0 or kl_args["kl_coef_final"] > 0.0
    ) and model.kl_reference_policy is None:
        model.set_reference_from_current()

    note = settings.get("curriculum_note")
    if note:
        print(f"curriculum note: {note}", flush=True)
    print(
        f"starting online training -> mode={mode} format={settings['battle_format']} "
        f"timesteps={settings['total_timesteps']} lr={settings['learning_rate']}",
        flush=True,
    )

    try:
        model.learn(
            total_timesteps=settings["total_timesteps"],
            log_interval=settings["log_interval"],
            tb_log_name=f"online_{mode}",
            callback=callbacks if callbacks else None,
            progress_bar=True,
        )
        model.save(str(settings["policy_path"]))
        print(f"saved policy to {settings['policy_path']}")
        if best_dir is not None:
            best_candidate = best_dir / "best_model.zip"
            if best_candidate.exists():
                shutil.copy(best_candidate, settings["best_policy_path"])
                print(f"saved best policy to {settings['best_policy_path']}")
    finally:
        _safe_close(env)
        if eval_env is not None:
            _safe_close(eval_env)


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


def main():
    parser = argparse.ArgumentParser(description="Run online PPO training.")

    parser.add_argument(
        "mode",
        type=str,
        default="scratch",
        help="Which training mode from config.online.modes to run (default: scratch)",
    )

    parser.add_argument(
        "--override",
        type=str,
        action="append",
        help="Override configuration settings in key=value format. "
        "For example: --override total_timesteps=5000000 --override learning_rate=0.0001",
    )

    args = parser.parse_args()
    overrides = parse_override_pairs(args.override)
    run(args.mode, overrides or None)


if __name__ == "__main__":
    main()
