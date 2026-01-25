import socket
from urllib.parse import urlparse

from poke_env import (
    AccountConfiguration,
    LocalhostServerConfiguration,
    ServerConfiguration,
    ShowdownServerConfiguration,
)
from poke_env.player.baselines import SimpleHeuristicsPlayer
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.core.env import action_space_size
from src.online.env import make_maskable_env
from src.online.opponents import PolicyOpponentPlayer, _unique_username


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


def _ensure_server_available(server_cfg):
    ws_url = getattr(server_cfg, "websocket_url", "")
    parsed = urlparse(ws_url)
    host = parsed.hostname
    if not host:
        return
    port = parsed.port or (443 if parsed.scheme == "wss" else 80)
    try:
        with socket.create_connection((host, port), timeout=3):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"Could not reach Showdown server at {ws_url} "
            "(start the server or update online.server_url)."
        ) from exc


def _build_env(settings, *, team_text, server_cfg):
    player_prefix = settings.get("player_name_prefix", "RLAgent")
    opponent_prefix = settings.get("opponent_name_prefix", "SimpleHeuristics")
    opponent_kinds = settings.get("opponent_kinds") or ["simple"]
    if not isinstance(opponent_kinds, list | tuple) or not opponent_kinds:
        opponent_kinds = ["simple"]
    opponent_kinds = [str(k).lower() for k in opponent_kinds]
    selfplay_path = settings.get("selfplay_opponent_path") or settings.get("policy_path")
    act_size = action_space_size(settings.get("battle_format", "gen9doublesou"))

    def _make_env_fn(env_idx: int):
        def _thunk():
            # Choose opponent based on env index to diversify across workers deterministically
            kind = opponent_kinds[env_idx % len(opponent_kinds)]
            opponent_account = AccountConfiguration(_unique_username(opponent_prefix), None)
            if kind == "maxbp":
                from poke_env.player.baselines import MaxBasePowerPlayer

                opponent = MaxBasePowerPlayer(
                    account_configuration=opponent_account,
                    battle_format=settings["battle_format"],
                    max_concurrent_battles=1,
                    server_configuration=server_cfg,
                )
            elif kind == "selfplay":
                opponent = PolicyOpponentPlayer(
                    model_path=selfplay_path,
                    act_size=act_size,
                    account_configuration=opponent_account,
                    battle_format=settings["battle_format"],
                    max_concurrent_battles=1,
                    server_configuration=server_cfg,
                )
            else:
                opponent = SimpleHeuristicsPlayer(
                    account_configuration=opponent_account,
                    battle_format=settings["battle_format"],
                    max_concurrent_battles=1,
                    server_configuration=server_cfg,
                )
            player_account = AccountConfiguration(_unique_username(player_prefix), None)
            limit = settings["rate_limit_per_second"]
            step_delay = 0.0 if not limit else 1.0 / max(limit, 1e-6)

            return make_maskable_env(
                opponent=opponent,
                battle_format=settings["battle_format"],
                rewards=settings["rewards"],
                team=team_text,
                account_configuration1=player_account,
                server_configuration=server_cfg,
                step_delay=step_delay,
                console_log_mode=settings.get("console_log_mode", "summary"),
                console_log_interval_sec=settings.get("console_log_interval_sec", 2.0),
            )

        return _thunk

    env_fns = [_make_env_fn(i) for i in range(settings["parallel_battles"])]
    if settings["parallel_battles"] > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


__all__ = ["_build_env", "_ensure_server_available", "_resolve_server_configuration"]
