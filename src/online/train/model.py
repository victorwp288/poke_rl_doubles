from pathlib import Path

import torch
from stable_baselines3.common.vec_env import VecNormalize

from src.online.kl_ppo import KLRegularizedMaskablePPO
from src.online.policy.head import configure_action_head
from src.online.policy.load import load_maskable_policy

from .entropy import _entropy_schedule_fn
from .warmstart import _load_bc_if_requested


def init_model_with_env(env, settings, device):
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
    # Init order: configure PPO heads, then optionally warmstart from BC weights.
    configure_action_head(model.policy, settings.get("policy_head_mlp_dim", 512))
    _load_bc_if_requested(model, settings)
    if vecnorm is not None:
        model.vecnormalize = vecnorm
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


__all__ = ["init_model_with_env"]
