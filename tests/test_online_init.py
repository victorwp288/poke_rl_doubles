#!/usr/bin/env python3

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.online.init import configure_action_head, load_behavior_clone_weights


class DummyExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Minimal layers to satisfy _linear_layers lookups
        self.policy_net = nn.Sequential(nn.Linear(1, 1))
        self.value_net = nn.Sequential(nn.Linear(1, 1))


class DummyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.features_dim = 3
        self.mlp_extractor = DummyExtractor()
        self.action_dist = SimpleNamespace(action_dims=[2, 3])
        # Start with a single linear head; configure_action_head will replace it.
        self.action_net = nn.Linear(self.features_dim, sum(self.action_dist.action_dims))


def test_bc_heads_apply_to_output_layer_with_custom_action_head(tmp_path):
    torch.manual_seed(0)
    policy = DummyPolicy()
    # Convert action head into two-layer MLP (Linear, GELU, Linear)
    configure_action_head(policy, head_hidden_dim=4)

    # Prepare BC checkpoint containing only head_a/head_b without head_mlp to force fallback path.
    w_a = torch.full((2, 8), 0.5)
    b_a = torch.full((2,), 0.1)
    w_b = torch.full((3, 8), -0.25)
    b_b = torch.full((3,), -0.2)
    checkpoint = {
        "head_a.weight": w_a,
        "head_a.bias": b_a,
        "head_b.weight": w_b,
        "head_b.bias": b_b,
    }
    ckpt_path = tmp_path / "bc_head.pt"
    torch.save(checkpoint, ckpt_path)

    # Capture pre-load weights to ensure final layer is the target for copies.
    out_layer = policy.action_net[-1]
    original = out_layer.weight.detach().clone()

    load_behavior_clone_weights(policy, ckpt_path)

    # BC weights should land on the output layer, not the hidden layer.
    torch.testing.assert_close(out_layer.weight[:2], w_a)
    torch.testing.assert_close(out_layer.weight[2:], w_b)
    torch.testing.assert_close(out_layer.bias[:2], b_a)
    torch.testing.assert_close(out_layer.bias[2:], b_b)
    # Hidden layer should remain unchanged.
    assert not torch.equal(original, out_layer.weight), "output layer should be updated"

