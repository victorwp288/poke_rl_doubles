# Small helpers shared by scripts when talking to poke-env.

from __future__ import annotations

from urllib.parse import urlparse


def act_size_for_format(battle_format: str) -> int:
    # Return per-slot action space size used by DoublesEnv.

    fmt = battle_format.lower()
    gimmicks_by_gen = {
        "gen6": 1,
        "gen7": 2,
        "gen8": 3,
        "gen9": 4,
    }
    gimmicks = next(
        (count for prefix, count in gimmicks_by_gen.items() if fmt.startswith(prefix)), 0
    )
    return 1 + 6 + 4 * 5 * (gimmicks + 1)


def _import_server_symbols() -> tuple[type, object | None, object | None]:
    modules = [
        "poke_env.ps_client.server_configuration",
        "poke_env.server_configuration",
        "poke_env.player",
    ]
    for name in modules:
        try:
            module = __import__(
                name,
                fromlist=[
                    "ServerConfiguration",
                    "LocalhostServerConfiguration",
                    "ShowdownServerConfiguration",
                ],
            )
        except ImportError:
            continue
        server_cfg = getattr(module, "ServerConfiguration", None)
        if server_cfg is None:
            continue
        localhost_cfg = getattr(module, "LocalhostServerConfiguration", None)
        showdown_cfg = getattr(module, "ShowdownServerConfiguration", None)
        return server_cfg, localhost_cfg, showdown_cfg
    raise RuntimeError("Could not import poke_env server configuration symbols")


def server_configuration_for_url(url: str):
    # Return a poke-env ServerConfiguration (or predefined constant) for a URL.

    server_cfg, localhost_cfg, showdown_cfg = _import_server_symbols()
    parsed = urlparse(url)
    hostname = parsed.hostname or ""

    if ("localhost" in url or hostname in {"127.0.0.1", "::1"}) and localhost_cfg is not None:
        return localhost_cfg

    if any(domain in hostname for domain in ("psim.us", "pokemonshowdown.com")) and (
        showdown_cfg is not None
    ):
        return showdown_cfg

    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    websocket_url = f"{scheme}://{netloc}/showdown/websocket"
    auth_url = "https://play.pokemonshowdown.com/action.php?"
    return server_cfg(websocket_url, auth_url)
