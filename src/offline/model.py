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
        hidden_dim=1024,
        hidden_layers=4,
        dropout=0.2,
        attn_heads=4,
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
        self.head_a = nn.Linear(hidden_dim, action_dim)
        self.head_b = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        shared = self.shared(self.input_norm(obs))
        context = self.context_norm(shared).unsqueeze(1)
        queries = self.slot_queries.unsqueeze(0).expand(obs.size(0), -1, -1)
        attn_out, _ = self.attn(queries, context, context)
        slot_a, slot_b = self.slot_norm(attn_out).unbind(dim=1)
        return torch.stack((self.head_a(slot_a), self.head_b(slot_b)), dim=1)
