# Simple two-head MLP policy

from __future__ import annotations

import torch
from torch import nn


class BehaviorCloningPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head_a = nn.Linear(hidden_dim, action_dim)
        self.head_b = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        if obs.dim() != 2 or obs.size(-1) != self.obs_dim:
            raise ValueError(f"expected (batch, {self.obs_dim}) but got {tuple(obs.shape)}")
        hidden = self.shared(obs)
        logits_a = self.head_a(hidden)
        logits_b = self.head_b(hidden)
        return {"logits": torch.stack((logits_a, logits_b), dim=1)}


__all__ = ["BehaviorCloningPolicy"]
