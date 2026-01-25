# Behavior cloning policy (shared trunk + per-slot heads).
import torch
from torch import nn


def _shared_layers(input_dim, hidden_dim, layer_count, dropout):
    layers = []
    current = input_dim
    for _ in range(layer_count):
        layers.append(nn.Linear(current, hidden_dim))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(dropout))
        current = hidden_dim
    return nn.Sequential(*layers)


class BehaviorCloningPolicy(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hidden_dim=1536,
        hidden_layers=6,
        dropout=0.2,
        attn_heads=8,
        slot_mlp_layers=2,
        head_mlp_dim=512,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(obs_dim)
        self.shared = _shared_layers(obs_dim, hidden_dim, hidden_layers, dropout)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=max(int(attn_heads), 1),
            dropout=dropout,
            batch_first=True,
        )
        self.slot_queries = nn.Parameter(torch.empty(2, hidden_dim))
        nn.init.normal_(self.slot_queries, std=0.02)
        self.slot_norm = nn.LayerNorm(hidden_dim)
        if slot_mlp_layers and slot_mlp_layers > 0:
            self.slot_mlp = _shared_layers(hidden_dim, hidden_dim, slot_mlp_layers, dropout)
        else:
            self.slot_mlp = nn.Identity()
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head_mlp = nn.Sequential(
            nn.Linear(hidden_dim, head_mlp_dim),
            nn.GELU(),
        )
        self.head_a = nn.Linear(head_mlp_dim, action_dim)
        self.head_b = nn.Linear(head_mlp_dim, action_dim)

    def forward(self, obs):
        shared = self.shared(self.input_norm(obs))
        context = self.context_norm(shared).unsqueeze(1)
        queries = self.slot_queries.unsqueeze(0).expand(obs.size(0), -1, -1)
        attn_out, _ = self.attn(queries, context, context)
        slots = self.slot_mlp(self.slot_norm(attn_out))
        head_in = self.head_norm(slots)
        head_hidden = self.head_mlp(head_in)
        slot_a, slot_b = slots.unbind(dim=1)
        hidden_a, hidden_b = head_hidden.unbind(dim=1)
        return torch.stack((self.head_a(hidden_a), self.head_b(hidden_b)), dim=1)
