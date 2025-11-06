import json
from pathlib import Path, PosixPath

import torch
from torch import nn
from torch.serialization import add_safe_globals

add_safe_globals([Path, PosixPath])

HEAD_KEYS = ("head_a", "head_b")


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
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
        for module in action_net:
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
    _apply_action_heads(policy=policy, heads=head_pairs)

    return stats


__all__ = ["load_behavior_clone_weights", "NormalizationStats"]
