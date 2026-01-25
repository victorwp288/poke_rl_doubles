import uuid

import torch
from poke_env import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ServerConfiguration,
    ShowdownServerConfiguration,
)
from poke_env.player.baselines import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from src.config import section

BASELINE_PLAYERS = {
    "simple": SimpleHeuristicsPlayer,
    "maxbp": MaxBasePowerPlayer,
    "random": RandomPlayer,
}


def _unique_username(base):
    prefix = (base or "bot")[:11]
    suffix = uuid.uuid4().hex[:6]
    return f"{prefix}{suffix}"


def _resolve_server_configuration(url):
    token = (url or "").strip()
    if not token:
        return LocalhostServerConfiguration
    lowered = token.lower()
    if lowered in {"showdown", "https://play.pokemonshowdown.com"}:
        return ShowdownServerConfiguration
    if lowered in {"local", "localhost", "http://localhost:8000"}:
        return LocalhostServerConfiguration
    websocket_url = token
    if websocket_url.startswith("http://"):
        websocket_url = "ws://" + websocket_url[len("http://") :]
    elif websocket_url.startswith("https://"):
        websocket_url = "wss://" + websocket_url[len("https://") :]
    elif not websocket_url.startswith(("ws://", "wss://")):
        websocket_url = f"ws://{websocket_url}"
    if not websocket_url.endswith("/websocket"):
        websocket_url = websocket_url.rstrip("/") + "/websocket"
    return ServerConfiguration(websocket_url, ShowdownServerConfiguration.authentication_url)


def _policy_kwargs():
    online_cfg = section("online")
    hidden_dim = int(online_cfg.get("policy_hidden_dim", 512))
    hidden_layers = int(online_cfg.get("policy_hidden_layers", 2))
    return {
        "net_arch": {
            "pi": [hidden_dim] * hidden_layers,
            "vf": [hidden_dim] * hidden_layers,
        },
        "activation_fn": torch.nn.ReLU,
        "ortho_init": False,
    }


def _build_opponent(kind, battle_format, server_cfg):
    factory = BASELINE_PLAYERS.get(kind)
    if factory is None:
        raise ValueError(f"unknown opponent '{kind}'")
    base_label = (kind or factory.__name__).replace(" ", "")
    username = _unique_username(base_label)
    account = AccountConfiguration(username, None)
    return factory(
        account_configuration=account,
        battle_format=battle_format,
        max_concurrent_battles=1,
        server_configuration=server_cfg,
    )


__all__ = [
    "BASELINE_PLAYERS",
    "_build_opponent",
    "_policy_kwargs",
    "_resolve_server_configuration",
    "_unique_username",
]
