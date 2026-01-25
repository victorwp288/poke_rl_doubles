from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize

from src.online.callbacks.checkpoints import CopyBestModelCallback, RollingCheckpointCallback
from src.online.callbacks.freeze import FreezeSharedCallback
from src.online.callbacks.metrics import RewardMetricsCallback


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


__all__ = ["_build_callbacks"]
