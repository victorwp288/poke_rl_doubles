import json
from pathlib import Path, PosixPath
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.serialization import add_safe_globals

from .head import _action_linear_head, _policy_device

# Allow safe loading of checkpoints that include numpy objects. Some numpy builds may
# lack certain dtypes; keep this registration best-effort.
try:
    safe_globals: list[Any] = [
        Path,
        PosixPath,
        np.ndarray,
        np.dtype,
        np.float32,
        np.float64,
        getattr(np.dtypes, "Float32DType", np.float32),
        getattr(np.dtypes, "Float64DType", np.float64),
    ]
    multiarray = getattr(np.core, "multiarray", None)
    reconstruct = getattr(multiarray, "_reconstruct", None)
    if reconstruct is not None:
        safe_globals.append(reconstruct)
    add_safe_globals(safe_globals)
except Exception as exc:  # pragma: no cover
    print(f"[online-init] warning: add_safe_globals skipped ({exc})")

HEAD_KEYS = ("head_a", "head_b")
HEAD_MLP_KEY = "head_mlp"


class NormalizationStats:
    def __init__(self, mean, var, count=None):
        self.mean = mean
        self.var = var
        self.count = count


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
    # Warmstart transfers shared trunk + action heads only; BC-only modules/optimizer state stay untouched.
    _apply_shared_layers(policy, shared_pairs)

    head_pairs = _head_tensors(
        state_dict=state_dict,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    applied = False
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


__all__ = ["NormalizationStats", "load_behavior_clone_weights"]
