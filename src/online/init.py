import json
from pathlib import Path, PosixPath

import numpy as np
import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.save_util import load_from_zip_file
from torch import nn
from torch.serialization import add_safe_globals

# Allow safe loading of checkpoints that include numpy objects. Some numpy builds may
# lack certain dtypes; keep this registration best-effort.
try:
    add_safe_globals(
        [
            Path,
            PosixPath,
            np.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            np.float32,
            np.float64,
            getattr(np.dtypes, "Float32DType", np.float32),
            getattr(np.dtypes, "Float64DType", np.float64),
        ]
    )
except Exception as exc:  # pragma: no cover
    print(f"[online-init] warning: add_safe_globals skipped ({exc})")

HEAD_KEYS = ("head_a", "head_b")
HEAD_MLP_KEY = "head_mlp"
_DEFAULT_HEAD_HIDDEN = 512


class NormalizationStats:
    def __init__(self, mean, var, count=None):
        self.mean = mean
        self.var = var
        self.count = count


def _policy_device(policy):
    return next(policy.parameters()).device


def _linear_layers(block):
    if isinstance(block, nn.Sequential):
        return [module for module in block if isinstance(module, nn.Linear)]
    return [block] if isinstance(block, nn.Linear) else []


def _load_checkpoint(checkpoint_path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        state_dict = payload.get("state_dict", payload)
        metadata = payload
    else:
        state_dict = payload
        metadata = {}
    return state_dict, metadata


def _ensure_matching_obs_dim(metadata, features_dim):
    obs_dim = metadata.get("obs_dim")
    if obs_dim is None or obs_dim == features_dim:
        return
    print(f"[online-init] ignoring obs_dim mismatch: bc={obs_dim} policy={features_dim}")


def _load_normalization_stats(metadata, stats_path, policy, device):
    normalization = metadata.get("normalization")
    if normalization is None and stats_path is not None and stats_path.exists():
        normalization = json.loads(stats_path.read_text(encoding="utf-8"))
    if not normalization:
        return None

    mean = torch.tensor(normalization.get("mean", []), dtype=torch.float32, device=device)
    std = torch.tensor(normalization.get("std", []), dtype=torch.float32, device=device)
    if mean.numel() != policy.features_dim or std.numel() != policy.features_dim:
        return None

    var = std.pow(2)
    count_raw = normalization.get("count")
    count_value = float(count_raw) if isinstance(count_raw, int | float) and count_raw > 0 else None
    stats = NormalizationStats(mean, var, count_value)

    obs_rms = getattr(policy, "obs_rms", None)
    if obs_rms is not None:
        with torch.no_grad():
            obs_rms.mean.copy_(mean)
            obs_rms.var.copy_(var)
    return stats


def _shared_layer_pairs(state_dict, device, checkpoint_path):
    keys = [key for key in state_dict if key.startswith("shared.") and key.endswith(".weight")]
    keys.sort(key=lambda key: int(key.split(".")[1]))
    pairs = []
    for weight_key in keys:
        index_part = weight_key.split(".")[1]
        bias_key = f"shared.{index_part}.bias"
        weight = state_dict.get(weight_key)
        bias = state_dict.get(bias_key)
        if weight is None or bias is None:
            print(f"[online-init] missing shared layer {index_part} in {checkpoint_path}")
            continue
        pairs.append((weight.to(device), bias.to(device)))
    return pairs


def _apply_shared_layers(policy, pairs):
    policy_linears = _linear_layers(policy.mlp_extractor.policy_net)
    value_linears = _linear_layers(policy.mlp_extractor.value_net)
    for layer_index, (weight, bias) in enumerate(pairs):
        for block in (policy_linears, value_linears):
            if layer_index >= len(block):
                continue
            target_layer = block[layer_index]
            if target_layer.weight.shape != weight.shape:
                print(
                    f"[online-init] skipping shared layer {layer_index}:"
                    f" target={tuple(target_layer.weight.shape)} bc={tuple(weight.shape)}"
                )
                continue
            with torch.no_grad():
                target_layer.weight.copy_(weight)
                target_layer.bias.copy_(bias)


def _head_tensors(state_dict, device, checkpoint_path):
    heads = []
    for key in HEAD_KEYS:
        weight_key = f"{key}.weight"
        bias_key = f"{key}.bias"
        weight = state_dict.get(weight_key)
        bias = state_dict.get(bias_key)
        if weight is None or bias is None:
            print(f"[online-init] missing head {key} in {checkpoint_path}")
            continue
        heads.append((weight.to(device), bias.to(device)))
    return heads


def _action_linear_head(policy):
    action_net = policy.action_net
    if isinstance(action_net, nn.Linear):
        return action_net
    if isinstance(action_net, nn.Sequential) and action_net:
        # When action_net is a multi-layer head (e.g., configured by configure_action_head),
        # we want the final Linear that produces action logits, not the first hidden layer.
        for module in reversed(action_net):
            if isinstance(module, nn.Linear):
                return module
    return action_net


def _apply_action_heads(policy, heads):
    dist = policy.action_dist
    action_dims = getattr(dist, "action_dims", [])
    target = _action_linear_head(policy)
    offset = 0
    for dim_size, (weight, bias) in zip(action_dims, heads, strict=False):
        rows = slice(offset, offset + dim_size)
        with torch.no_grad():
            target.weight[rows].copy_(weight)
            target.bias[rows].copy_(bias)
        offset += dim_size


def load_behavior_clone_weights(policy, checkpoint_path, stats_path=None):
    checkpoint_path = Path(checkpoint_path)
    stats_path = Path(stats_path) if stats_path else None
    device = _policy_device(policy)
    state_dict, metadata = _load_checkpoint(checkpoint_path)
    _ensure_matching_obs_dim(metadata, policy.features_dim)

    stats = _load_normalization_stats(
        metadata=metadata,
        stats_path=stats_path,
        policy=policy,
        device=device,
    )

    shared_pairs = _shared_layer_pairs(
        state_dict=state_dict,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    _apply_shared_layers(policy, shared_pairs)

    head_pairs = _head_tensors(
        state_dict=state_dict,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    applied = False
    # If action_net is a two-layer MLP (Linear, activation, Linear), try to map BC head_mlp + heads.
    if isinstance(policy.action_net, nn.Sequential) and len(policy.action_net) >= 3:
        first = next((m for m in policy.action_net if isinstance(m, nn.Linear)), None)
        last = next((m for m in reversed(policy.action_net) if isinstance(m, nn.Linear)), None)
        if first is not None and last is not None:
            head_mlp_weight = state_dict.get(f"{HEAD_MLP_KEY}.0.weight")
            head_mlp_bias = state_dict.get(f"{HEAD_MLP_KEY}.0.bias")
            if head_mlp_weight is not None and head_mlp_bias is not None:
                head_mlp_weight = head_mlp_weight.to(device)
                head_mlp_bias = head_mlp_bias.to(device)
                head_hidden = head_mlp_weight.shape[0]
                total_hidden = head_hidden * 2
                if (
                    first.weight.shape[0] == total_hidden
                    and first.weight.shape[1] == head_mlp_weight.shape[1]
                ):
                    with torch.no_grad():
                        first.weight[:head_hidden].copy_(head_mlp_weight)
                        first.weight[head_hidden:].copy_(head_mlp_weight)
                        first.bias[:head_hidden].copy_(head_mlp_bias)
                        first.bias[head_hidden:].copy_(head_mlp_bias)
                    total_actions = sum(getattr(policy.action_dist, "action_dims", []))
                    if (
                        last.weight.shape[0] == total_actions
                        and last.weight.shape[1] == total_hidden
                    ):
                        offset = 0
                        with torch.no_grad():
                            for dim_size, (w, b) in zip(
                                getattr(policy.action_dist, "action_dims", []),
                                head_pairs,
                                strict=False,
                            ):
                                # determine slot index: 0 or 1
                                slot_index = 0 if offset == 0 else 1
                                col_start = slot_index * head_hidden
                                col_end = col_start + head_hidden
                                rows = slice(offset, offset + dim_size)
                                last.weight[rows, :] = 0.0
                                last.weight[rows, col_start:col_end].copy_(w)
                                last.bias[rows].copy_(b)
                                offset += dim_size
                        applied = True
    if not applied:
        _apply_action_heads(policy=policy, heads=head_pairs)

    return stats


def configure_action_head(policy, head_hidden_dim):
    """Replace the SB3 action_net with a two-layer MLP head while keeping output dims the same."""
    action_dims = getattr(policy.action_dist, "action_dims", [])
    if not action_dims:
        return
    total_actions = sum(action_dims)
    current = _action_linear_head(policy)
    if not isinstance(current, nn.Linear):
        return
    in_features = current.in_features
    head_hidden_dim = int(head_hidden_dim)
    if head_hidden_dim <= 0:
        return
    device = _policy_device(policy)
    new_head = nn.Sequential(
        nn.Linear(in_features, head_hidden_dim * 2),
        nn.GELU(),
        nn.Linear(head_hidden_dim * 2, total_actions),
    ).to(device)
    policy.action_net = new_head


def _infer_action_head_hidden_dim(checkpoint_path):
    try:
        _, params, _ = load_from_zip_file(
            str(checkpoint_path), device="cpu", custom_objects={"kl_reference_policy": None}
        )
        return _infer_action_head_hidden_dim_from_params(params)
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"[online-init] warning: failed to infer action head dim: {exc}")
    return None


def _infer_action_head_hidden_dim_from_params(params):
    policy_params = params.get("policy", {}) if isinstance(params, dict) else {}
    if isinstance(policy_params, dict):
        weight = policy_params.get("action_net.0.weight")
        if weight is not None and weight.ndim == 2 and weight.shape[0] % 2 == 0:
            return int(weight.shape[0] // 2)
    return None


class MaskablePolicyWithHead(MaskableActorCriticPolicy):
    def __init__(self, *args, action_head_hidden_dim=None, **kwargs):
        self._action_head_hidden_dim = action_head_hidden_dim
        super().__init__(*args, **kwargs)
        hidden_dim = self._action_head_hidden_dim or _DEFAULT_HEAD_HIDDEN
        configure_action_head(self, hidden_dim)


def load_maskable_policy(checkpoint_path, *, device="cpu", head_hidden_dim=None):
    checkpoint_path = Path(checkpoint_path)
    # Load once to read saved policy kwargs and params without instantiating a model yet.
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


__all__ = [
    "load_behavior_clone_weights",
    "NormalizationStats",
    "configure_action_head",
    "load_maskable_policy",
    "MaskablePolicyWithHead",
]
