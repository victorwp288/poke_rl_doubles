from __future__ import annotations

import argparse
import shutil

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecNormalize

from src.online.config import _apply_overrides, _mode_settings, _team_details, parse_override_pairs
from src.online.environment import (
    _build_env,
    _ensure_server_available,
    _resolve_server_configuration,
)
from src.online.train.callbacks import _build_callbacks
from src.online.train.device import _resolve_device
from src.online.train.eval_export import _export_eval_metrics
from src.online.train.model import init_model_with_env
from src.online.train.utils import _safe_close


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

    schedule = settings.get("opponent_schedule")
    if schedule:
        # Schedule phases consume timesteps in order; the final phase gets any remainder.
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
                model = init_model_with_env(env, settings, device)
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
            # VecNormalize stats are saved alongside policies as <stem>_vecnorm.pkl.
            # Eval must reload the matching vecnorm file for apples-to-apples comparisons.
            vecnorm.save(str(vec_path))
            print(f"saved vecnormalize stats to {vec_path}")
            _safe_close(vecnorm)
        print(f"saved policy to {settings['policy_path']}")
        return

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
    model = init_model_with_env(env, settings, device)
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
            # VecNormalize stats are saved alongside policies as <stem>_vecnorm.pkl.
            # Eval must reload the matching vecnorm file for apples-to-apples comparisons.
            vecnorm_single.save(str(vec_path))
            print(f"saved vecnormalize stats to {vec_path}")
    finally:
        _safe_close(env)
        if eval_env is not None:
            _safe_close(eval_env)


def run_online_training(args: argparse.Namespace) -> None:
    overrides = parse_override_pairs(getattr(args, "override", None))
    run(args.mode, overrides or None)


__all__ = ["run", "run_online_training"]
