import contextlib
import uuid

from poke_env import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ServerConfiguration,
    ShowdownServerConfiguration,
)
from poke_env.environment.doubles_env import DoublesEnv

from src.core.env import action_space_size


def _resolve_server_configuration(url):
    token = url.strip()
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

    auth_url = ShowdownServerConfiguration.authentication_url
    return ServerConfiguration(websocket_url, auth_url)


def _probe_action_space_size(battle_format, server_cfg):
    env = None
    try:
        account = AccountConfiguration(f"Recorder{uuid.uuid4().hex[:6]}", None)
        env = DoublesEnv(
            account_configuration1=account,
            battle_format=battle_format,
            server_configuration=server_cfg,
            start_listening=False,
            fake=True,
        )
        agents = getattr(env, "possible_agents", None) or []
        agent = agents[0] if agents else getattr(env, "agent", None)
        if agent is None:
            return None
        space = env.action_space(agent)
        nvec = getattr(space, "nvec", None)
        if nvec is None or len(nvec) == 0:
            return None
        return int(nvec[0])
    except Exception as exc:
        print(f"[warn] failed to probe action space size via DoublesEnv: {exc}")
        return None
    finally:
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()


def _resolve_action_space_size(settings, server_cfg):
    inferred = _probe_action_space_size(settings.battle_format, server_cfg)
    return inferred or action_space_size(settings.battle_format)


def _action_to_tuple(order, battle):
    raw = DoublesEnv.order_to_action(order, battle, fake=True, strict=False)
    first, second = (int(raw[0]), int(raw[1]))
    return first, second


__all__ = [
    "_action_to_tuple",
    "_probe_action_space_size",
    "_resolve_action_space_size",
    "_resolve_server_configuration",
]
