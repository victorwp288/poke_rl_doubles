from pathlib import Path

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.save_util import load_from_zip_file

from .head import (
    _DEFAULT_HEAD_HIDDEN,
    _infer_action_head_hidden_dim_from_params,
    configure_action_head,
)


def _infer_action_head_hidden_dim(checkpoint_path):
    try:
        _, params, _ = load_from_zip_file(
            str(checkpoint_path), device="cpu", custom_objects={"kl_reference_policy": None}
        )
        return _infer_action_head_hidden_dim_from_params(params)
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"[online-init] warning: failed to infer action head dim: {exc}")
    return None


class MaskablePolicyWithHead(MaskableActorCriticPolicy):
    def __init__(self, *args, action_head_hidden_dim=None, **kwargs):
        self._action_head_hidden_dim = action_head_hidden_dim
        super().__init__(*args, **kwargs)
        hidden_dim = self._action_head_hidden_dim or _DEFAULT_HEAD_HIDDEN
        configure_action_head(self, hidden_dim)


def load_maskable_policy(checkpoint_path, *, device="cpu", head_hidden_dim=None):
    checkpoint_path = Path(checkpoint_path)
    data, params, _ = load_from_zip_file(
        str(checkpoint_path), device=device, custom_objects={"kl_reference_policy": None}
    )
    policy_kwargs = dict(data.get("policy_kwargs", {}) or {})
    inferred_dim = head_hidden_dim or _infer_action_head_hidden_dim_from_params(params)
    policy_kwargs["action_head_hidden_dim"] = inferred_dim or _DEFAULT_HEAD_HIDDEN

    custom_objects = {
        "kl_reference_policy": None,
        "policy_class": MaskablePolicyWithHead,
        "policy_kwargs": policy_kwargs,
    }
    return MaskablePPO.load(str(checkpoint_path), device=device, custom_objects=custom_objects)


__all__ = ["MaskablePolicyWithHead", "load_maskable_policy"]
