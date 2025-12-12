#!/usr/bin/env python3
import argparse
import contextlib
import csv
import json
import pathlib
import shutil
import socket
import sys
import time
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
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player.baselines import SimpleHeuristicsPlayer
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3.common.vec_env import VecNormalize

from src.config import section  # noqa: E402
from src.core.env import action_space_size  # noqa: E402
from src.core.features import encode_observation, slot_action_mask  # noqa: E402
from src.online.env import make_maskable_env  # noqa: E402
from src.online.init import (  # noqa: E402
    configure_action_head,
    load_behavior_clone_weights,
    load_maskable_policy,
)
from src.online.kl_ppo import KLRegularizedMaskablePPO  # noqa: E402
from src.utils.teambuilders import read_showdown_team  # noqa: E402


class FreezeSharedCallback(BaseCallback):
    """
    Freeze policy/value trunks for an initial step window, then unfreeze.
    Useful for warmstarts to keep the policy close to BC early in training.
    """

    def __init__(self, freeze_steps: int):
        super().__init__()
        self.freeze_steps = int(freeze_steps)
        self._frozen = False
        self._unfrozen = False

    def _set_requires_grad(self, flag: bool):
        policy = self.model.policy
        modules = [
            policy.features_extractor,
            policy.mlp_extractor.policy_net,
            policy.mlp_extractor.value_net,
        ]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = flag

    def _freeze(self):
        if self._frozen or self.freeze_steps <= 0:
            return
        self._set_requires_grad(False)
        self._frozen = True
        print(f"[freeze] frozen trunks for first {self.freeze_steps} steps", flush=True)

    def _unfreeze(self):
        if self._unfrozen:
            return
        self._set_requires_grad(True)
        self._unfrozen = True
        print("[freeze] trunks unfrozen", flush=True)

    def _on_training_start(self) -> None:
        self._freeze()

    def _on_step(self) -> bool:
        if self.freeze_steps <= 0 or self._unfrozen:
            return True
        if self.num_timesteps >= self.freeze_steps:
            self._unfreeze()
        return True


def _export_eval_metrics(best_dir, settings):
    import numpy as _np

    eval_file = pathlib.Path(best_dir) / "evaluations.npz"
    if not eval_file.exists():
        print("[eval export] evaluations.npz not found", flush=True)
        return
    data = _np.load(eval_file, allow_pickle=True)
    out = pathlib.Path("outputs/eval")
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "ppo_eval.csv"
    jsonl = out / "ppo_eval.jsonl"
    exists = csv_path.exists()
    with open(csv_path, "a", newline="") as cf, open(jsonl, "a") as jf:
        w = csv.writer(cf)
        if not exists:
            w.writerow(
                [
                    "settings_id",
                    "eval_index",
                    "timesteps",
                    "mean_reward",
                    "reward_std",
                    "mean_ep_length",
                ]
            )
        sid = str(settings.get("policy_path", "unknown"))
        for i, ts in enumerate(data["timesteps"]):
            rewards = data["results"][i]
            ep_lengths = data["ep_lengths"][i]
            mr = float(rewards.mean())
            sr = float(rewards.std())
            ml = float(ep_lengths.mean())
            w.writerow([sid, i, int(ts), mr, sr, ml])
            jf.write(
                json.dumps(
                    {
                        "settings_id": sid,
                        "eval_index": i,
                        "timesteps": int(ts),
                        "mean_reward": mr,
                        "reward_std": sr,
                        "mean_ep_length": ml,
                    }
                )
                + "\n"
            )


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
        print(f"[online eval] new best model copied to {self.destination}", flush=True)

    def _on_step(self) -> bool:
        self._copy_if_updated()
        return True


class RewardMetricsCallback(BaseCallback):
    """Record reward-related stats to the logger and emit concise console summaries.

    This inspects `infos` from the rollout buffer, which is robust for both DummyVecEnv
    and SubprocVecEnv.
    """

    def __init__(self, summary_interval_sec: float = 5.0, summary_window: int = 50):
        super().__init__()
        self.summary_interval_sec = float(summary_interval_sec)
        self.summary_window = max(int(summary_window), 1)
        self._last_summary_time: float | None = None
        self._recent: deque[dict] = deque(maxlen=self.summary_window)
        self._best_mean_score: float | None = None

    def _record_from_infos(self, infos):
        if not infos:
            return
        repaired_count = 0
        for info in infos:
            if not isinstance(info, dict):
                continue
            if info.get("repaired_action"):
                repaired_count += 1
            stats = info.get("battle_stats")
            if not isinstance(stats, dict):
                continue
            # Always log numeric stats to TensorBoard logger
            for key, value in stats.items():
                try:
                    v = float(value)
                except Exception:
                    continue
                self.logger.record(f"reward_components/{key}", v)

            # Only track completed battles with a clear result for summaries
            result = stats.get("result")
            if result not in {"win", "loss", "draw"}:
                continue
            entry = {
                "score": float(stats.get("score", 0.0)),
                "delta": float(stats.get("delta", 0.0)),
                "result": result,
                "turn": int(stats.get("turn", 0)),
            }
            self._recent.append(entry)
        if infos:
            rate = repaired_count / max(len(infos), 1)
            self.logger.record("safety/repaired_rate", rate)
            if self.logger:
                # Keep a rolling scalar for inspection
                self.logger.record("safety/repaired_count", repaired_count)

    def _summary_from_recent(self):
        if not self._recent:
            return {}
        n = len(self._recent)
        mean_score = sum(e["score"] for e in self._recent) / n
        mean_delta = sum(e["delta"] for e in self._recent) / n
        wins = sum(1 for e in self._recent if e.get("result") == "win")
        losses = sum(1 for e in self._recent if e.get("result") == "loss")
        draws = sum(1 for e in self._recent if e.get("result") == "draw")
        total_results = wins + losses + draws
        win_rate = wins / total_results if total_results else 0.0
        mean_turn = sum(e["turn"] for e in self._recent) / n
        last = self._recent[-1]
        return {
            "n": n,
            "mean_score": mean_score,
            "mean_delta": mean_delta,
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "mean_turn": mean_turn,
            "last_score": last.get("score", 0.0),
            "last_delta": last.get("delta", 0.0),
            "last_result": last.get("result", "-") or "-",
            "last_turn": last.get("turn", 0),
        }

    def _maybe_print_summary(self):
        if self.summary_interval_sec <= 0.0:
            return
        now = time.time()
        if (
            self._last_summary_time is not None
            and now - self._last_summary_time < self.summary_interval_sec
        ):
            return
        self._last_summary_time = now

        summary = self._summary_from_recent()
        if not summary:
            return

        mean_score = summary["mean_score"]
        if self._best_mean_score is None or mean_score > self._best_mean_score:
            self._best_mean_score = mean_score
            best_flag = " *best*"
        else:
            best_flag = ""

        t = int(self.num_timesteps)
        print(
            f"[online train]{best_flag} t={t} n={summary['n']} "
            f"mean_score={mean_score:.3f} win_rate={summary['win_rate']:.3f} "
            f"mean_delta={summary['mean_delta']:.3f} mean_turn={summary['mean_turn']:.1f} "
            f"last_score={summary['last_score']:.3f} last_result={summary['last_result']} "
            f"last_turn={summary['last_turn']}",
            flush=True,
        )

    def _on_step(self):
        # SB3 passes rollout data via `locals`; `infos` is a list (one per env).
        infos = self.locals.get("infos") or []
        self._record_from_infos(infos)
        self._maybe_print_summary()
        return True


def _unique_username(prefix):
    token = (prefix or "bot")[:11]
    suffix = uuid.uuid4().hex[:6]
    return f"{token}{suffix}"


class PolicyOpponentPlayer(SimpleHeuristicsPlayer):
    """
    Opponent driven by a frozen MaskablePPO checkpoint.
    Falls back to SimpleHeuristics if prediction fails.
    """

    def __init__(self, model_path, act_size, **kwargs):
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"opponent policy checkpoint not found: {self._model_path}")
        self._model = load_maskable_policy(self._model_path, device="cpu")
        self._act_size = act_size
        super().__init__(**kwargs)

    def choose_move(self, battle):
        mask0 = slot_action_mask(battle, 0, self._act_size)
        mask1 = slot_action_mask(battle, 1, self._act_size)
        concat_mask = np.concatenate([mask0, mask1]).astype(np.int8)
        obs = np.asarray(encode_observation(battle), dtype=np.float32)
        try:
            action, _ = self._model.predict(obs, action_masks=concat_mask, deterministic=True)
            action = np.asarray(action, dtype=int)
            if action.shape[0] == 2:
                order = DoublesEnv.action_to_order(action, battle, fake=False, strict=False)
                if order is not None:
                    return order
        except Exception:
            pass
        return super().choose_move(battle)


def _entropy_schedule_fn(settings):
    start = float(settings.get("entropy_coef_start", settings.get("entropy_coef", 0.0)))
    final = float(settings.get("entropy_coef_final", start))
    anneal_steps = int(settings.get("entropy_anneal_steps", 0))
    total = int(settings.get("total_timesteps", 1))
    if anneal_steps <= 0 or start == final:
        return start

    def schedule(progress_remaining: float) -> float:
        # progress_remaining goes 1 -> 0 over training
        current_step = (1.0 - progress_remaining) * total
        frac = min(max(current_step / max(anneal_steps, 1), 0.0), 1.0)
        return start + (final - start) * frac

    return schedule


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


def _build_env(settings, *, team_text, server_cfg):
    player_prefix = settings.get("player_name_prefix", "RLAgent")
    opponent_prefix = settings.get("opponent_name_prefix", "SimpleHeuristics")
    opponent_kinds = settings.get("opponent_kinds") or ["simple"]
    if not isinstance(opponent_kinds, (list, tuple)) or not opponent_kinds:
        opponent_kinds = ["simple"]
    opponent_kinds = [str(k).lower() for k in opponent_kinds]
    selfplay_path = settings.get("selfplay_opponent_path") or settings.get("policy_path")
    act_size = action_space_size(settings.get("battle_format", "gen9doublesou"))

    def _make_env_fn(env_idx: int):
        def _thunk():
            # Choose opponent based on env index to diversify across workers deterministically
            kind = opponent_kinds[env_idx % len(opponent_kinds)]
            opponent_account = AccountConfiguration(_unique_username(opponent_prefix), None)
            if kind == "maxbp":
                from poke_env.player.baselines import MaxBasePowerPlayer

                opponent = MaxBasePowerPlayer(
                    account_configuration=opponent_account,
                    battle_format=settings["battle_format"],
                    max_concurrent_battles=1,
                    server_configuration=server_cfg,
                )
            elif kind == "selfplay":
                opponent = PolicyOpponentPlayer(
                    model_path=selfplay_path,
                    act_size=act_size,
                    account_configuration=opponent_account,
                    battle_format=settings["battle_format"],
                    max_concurrent_battles=1,
                    server_configuration=server_cfg,
                )
            else:
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
                console_log_mode=settings.get("console_log_mode", "summary"),
                console_log_interval_sec=settings.get("console_log_interval_sec", 2.0),
            )

        return _thunk

    env_fns = [_make_env_fn(i) for i in range(settings["parallel_battles"])]
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
    for path_key in ("policy_path", "best_policy_path", "bc_checkpoint", "tensorboard_dir"):
        if (
            path_key in merged
            and merged[path_key] is not None
            and not isinstance(merged[path_key], Path)
        ):
            merged[path_key] = Path(merged[path_key])
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

    def _init_model_with_env(env):
        vecnorm = None
        if settings.get("use_vecnormalize", False) and not isinstance(env, VecNormalize):
            vecnorm = VecNormalize(
                env,
                training=True,
                norm_obs=True,
                norm_reward=False,
                clip_obs=10.0,
            )
            env = vecnorm
        ent_coef_schedule = _entropy_schedule_fn(settings)
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
            clip_range_vf=settings.get("clip_range_vf"),
            gamma=settings.get("gamma", 0.99),
            gae_lambda=settings.get("gae_lambda", 0.95),
            vf_coef=settings.get("vf_coef", 0.5),
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
            ent_coef=ent_coef_schedule,
            target_kl=settings.get("target_kl"),
            **kl_args,
        )
        configure_action_head(model.policy, settings.get("policy_head_mlp_dim", 512))
        _load_bc_if_requested(model, settings)
        if vecnorm is not None:
            model.vecnormalize = vecnorm
        # Optional: load initial PPO weights from a checkpoint (non-strict to allow head/layout differences)
        init_path = settings.get("policy_init_path")
        if init_path:
            init_path = Path(init_path)
            if not init_path.exists():
                raise FileNotFoundError(f"policy_init_path not found: {init_path}")
            loaded = load_maskable_policy(init_path, device=device)
            loaded.set_env(model.get_env())
            model.set_parameters(loaded.get_parameters(), exact_match=False, device=device)
            if hasattr(loaded, "vecnormalize") and loaded.vecnormalize is not None:
                model.vecnormalize = loaded.vecnormalize

        if (
            kl_args["kl_coef_start"] > 0.0 or kl_args["kl_coef_final"] > 0.0
        ) and model.kl_reference_policy is None:
            model.set_reference_from_current()
        return model

    def _wrap_eval_env(eval_env, vecnorm):
        if eval_env is None:
            return None
        if vecnorm is None:
            return eval_env
        wrapped = VecNormalize(
            eval_env,
            training=False,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
        )
        # copy running stats
        wrapped.obs_rms.mean = vecnorm.obs_rms.mean.copy()
        wrapped.obs_rms.var = vecnorm.obs_rms.var.copy()
        wrapped.obs_rms.count = vecnorm.obs_rms.count
        return wrapped

    def _build_callbacks(settings, best_dir=None, eval_env=None, vecnorm=None):
        cbs = [RewardMetricsCallback()]
        checkpoint_freq = settings.get("checkpoint_freq", 0)
        if checkpoint_freq > 0:
            checkpoint_dir = settings["policy_path"].parent / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            latest_name = f"{settings['policy_path'].stem}_latest.zip"
            cbs.append(
                RollingCheckpointCallback(
                    save_freq=checkpoint_freq,
                    save_path=checkpoint_dir,
                    name_prefix=settings["policy_path"].stem,
                    keep_last=settings.get("checkpoint_keep_last", 1),
                    fixed_filename=latest_name,
                )
            )
        eval_env_wrapped = _wrap_eval_env(eval_env, vecnorm)
        if eval_env_wrapped is not None and best_dir is not None:
            copy_best = CopyBestModelCallback(best_dir, settings["best_policy_path"])
            cbs.append(
                EvalCallback(
                    eval_env_wrapped,
                    callback_on_new_best=copy_best,
                    best_model_save_path=str(best_dir),
                    log_path=str(best_dir),
                    eval_freq=settings.get("eval_freq", 0),
                    n_eval_episodes=max(settings.get("eval_episodes", 1), 1),
                    deterministic=True,
                )
            )
        freeze_steps = int(settings.get("freeze_shared_steps", 0) or 0)
        if freeze_steps > 0:
            cbs.append(FreezeSharedCallback(freeze_steps))
        return cbs

    schedule = settings.get("opponent_schedule")
    if schedule:
        remaining = settings["total_timesteps"]
        model = None
        vecnorm = None
        total_done = 0
        for idx, phase in enumerate(schedule):
            phase_steps = phase.get("timesteps")
            if phase_steps is None or phase_steps <= 0:
                phase_steps = remaining
            phase_steps = min(phase_steps, remaining)
            remaining -= phase_steps
            phase_settings = dict(settings)
            phase_settings["opponent_kinds"] = phase.get("kinds") or settings.get("opponent_kinds")
            eval_env = None
            best_dir = None
            if idx == len(schedule) - 1 and settings.get("eval_freq", 0) > 0:
                eval_env = _build_env(
                    {**phase_settings, "parallel_battles": 1},
                    team_text=team_text,
                    server_cfg=server_cfg,
                )
                best_dir = (
                    settings["best_policy_path"].parent / f"{settings['policy_path'].stem}_best_tmp"
                )
                best_dir.mkdir(parents=True, exist_ok=True)
            env_raw = _build_env(phase_settings, team_text=team_text, server_cfg=server_cfg)
            if settings.get("use_vecnormalize", False):
                if vecnorm is None:
                    vecnorm = VecNormalize(
                        env_raw,
                        training=True,
                        norm_obs=True,
                        norm_reward=False,
                        clip_obs=10.0,
                    )
                else:
                    vecnorm.set_venv(env_raw)
                env = vecnorm
            else:
                env = env_raw
            env.seed(seed)
            if model is None:
                model = _init_model_with_env(env)
            else:
                model.set_env(env)
            callbacks = _build_callbacks(
                phase_settings,
                best_dir,
                eval_env,
                vecnorm if settings.get("use_vecnormalize", False) else None,
            )
            print(
                f"[curriculum] phase {idx + 1}/{len(schedule)} kinds={phase_settings['opponent_kinds']} "
                f"steps={phase_steps} done={total_done}",
                flush=True,
            )
            model.learn(
                total_timesteps=phase_steps,
                log_interval=settings["log_interval"],
                tb_log_name=f"online_{mode}",
                callback=callbacks if callbacks else None,
                progress_bar=True,
                reset_num_timesteps=False,
            )
            total_done += phase_steps
            if not (settings.get("use_vecnormalize", False) and vecnorm is not None):
                _safe_close(env)
            if eval_env is not None:
                _export_eval_metrics(best_dir, settings)
                _safe_close(eval_env)
            if remaining <= 0:
                break
        model.save(str(settings["policy_path"]))
        best_dir = settings["best_policy_path"].parent / f"{settings['policy_path'].stem}_best_tmp"
        if best_dir.exists():
            best_candidate = best_dir / "best_model.zip"
            if best_candidate.exists():
                shutil.copy(best_candidate, settings["best_policy_path"])
                print(f"saved best policy to {settings['best_policy_path']}")
        if vecnorm is not None:
            vec_path = settings["policy_path"].with_name(
                f"{settings['policy_path'].stem}_vecnorm.pkl"
            )
            vecnorm.save(str(vec_path))
            print(f"saved vecnormalize stats to {vec_path}")
            _safe_close(vecnorm)
        print(f"saved policy to {settings['policy_path']}")
        return

    # No schedule: single-phase path
    env = _build_env(settings, team_text=team_text, server_cfg=server_cfg)
    env.seed(seed)
    eval_env = None
    best_dir = None
    if settings.get("eval_freq", 0) > 0:
        eval_env = _build_env(
            {**settings, "parallel_battles": 1}, team_text=team_text, server_cfg=server_cfg
        )
        best_dir = settings["best_policy_path"].parent / f"{settings['policy_path'].stem}_best_tmp"
        best_dir.mkdir(parents=True, exist_ok=True)
    model = _init_model_with_env(env)
    vecnorm_single = getattr(model, "vecnormalize", None)
    callbacks = _build_callbacks(settings, best_dir, eval_env, vecnorm_single)
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
        if best_dir is not None:
            _export_eval_metrics(best_dir, settings)

        print(f"saved policy to {settings['policy_path']}")
        if best_dir is not None:
            best_candidate = best_dir / "best_model.zip"
            if best_candidate.exists():
                shutil.copy(best_candidate, settings["best_policy_path"])
                print(f"saved best policy to {settings['best_policy_path']}")
        if vecnorm_single is not None:
            vec_path = settings["policy_path"].with_name(
                f"{settings['policy_path'].stem}_vecnorm.pkl"
            )
            vecnorm_single.save(str(vec_path))
            print(f"saved vecnormalize stats to {vec_path}")
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
