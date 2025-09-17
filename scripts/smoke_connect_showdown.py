#!/usr/bin/env python3
"""Tiny connectivity smoke test for a local Pokémon Showdown server."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import typer

try:
    from poke_env.player import Player, RandomPlayer
except Exception as exc:  # pragma: no cover
    raise RuntimeError("poke_env not installed or incompatible: " + str(exc)) from exc

try:
    from poke_env.ps_client.server_configuration import (
        LocalhostServerConfiguration,
        ServerConfiguration,
        ShowdownServerConfiguration,
    )
except ImportError:
    try:
        from poke_env.server_configuration import (
            LocalhostServerConfiguration,
            ServerConfiguration,
            ShowdownServerConfiguration,
        )
    except ImportError:
        # Ancient versions expose the class on poke_env.player.
        ServerConfiguration = getattr(Player, "ServerConfiguration", None)  # type: ignore[attr-defined]
        LocalhostServerConfiguration = getattr(  # type: ignore[attr-defined]
            Player, "LocalhostServerConfiguration", None
        )
        ShowdownServerConfiguration = getattr(  # type: ignore[attr-defined]
            Player, "ShowdownServerConfiguration", None
        )

try:
    from poke_env.ps_client.account_configuration import AccountConfiguration
except ImportError:
    try:
        from poke_env.account_configuration import AccountConfiguration
    except ImportError:
        try:
            from poke_env.player_configuration import PlayerConfiguration as AccountConfiguration
        except ImportError:
            AccountConfiguration = None  # type: ignore[assignment]


class Settings:
    server_url: str = "http://localhost:8000"
    username: str = "rl-bot-1"
    opponent: str = "rl-bot-2"
    password: str | None = None
    battle_format: str = "gen9doublesou"
    team_path: Path = Path("teams/gen9dou_fixed.txt")
    replays_dir: Path = Path("replays")
    n_battles: int = 2


def load_team_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Team file not found: {path}")
    return path.read_text(encoding="utf-8")


def resolve_server_configuration(url: str) -> Any:
    if "localhost" in url and LocalhostServerConfiguration is not None:
        return LocalhostServerConfiguration
    if any(host in url for host in ("psim.us", "pokemonshowdown.com")) and (
        ShowdownServerConfiguration is not None
    ):
        return ShowdownServerConfiguration

    parsed = urlparse(url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    websocket_url = f"{scheme}://{netloc}/showdown/websocket"
    auth_url = "https://play.pokemonshowdown.com/action.php?"
    if ServerConfiguration is None:
        raise RuntimeError("poke_env ServerConfiguration class not available")
    return ServerConfiguration(websocket_url, auth_url)


def build_account_pair(
    account_cls: type[Any] | None, username: str, opponent: str, password: str | None
) -> tuple[Any | None, Any | None]:
    if account_cls is None:
        return (None, None)
    try:
        return (account_cls(username, password), account_cls(opponent, password))
    except Exception:
        return (None, None)


def make_players(
    team_text: str,
    battle_format: str,
    server_configuration: Any,
    accounts: tuple[Any | None, Any | None],
) -> tuple[RandomPlayer, RandomPlayer]:
    base_kwargs = dict(
        battle_format=battle_format,
        team=team_text,
        server_configuration=server_configuration,
        max_concurrent_battles=1,
    )
    p1_kwargs = base_kwargs.copy()
    p2_kwargs = base_kwargs.copy()
    acc1, acc2 = accounts
    if acc1 is not None:
        p1_kwargs["account_configuration"] = acc1
    if acc2 is not None:
        p2_kwargs["account_configuration"] = acc2
    return RandomPlayer(**p1_kwargs), RandomPlayer(**p2_kwargs)


def write_battle_logs(player: RandomPlayer, destination: Path) -> None:
    for tag, battle in getattr(player, "battles", {}).items():
        log_path = destination / f"{tag}.log"
        lines = [
            f"battle_tag: {tag}",
            f"finished: {getattr(battle, 'finished', True)}",
            f"won: {getattr(battle, 'won', False)}",
            f"turns: {getattr(battle, 'turn', 'NA')}",
            f"opponent_name: {getattr(battle, 'opponent_username', 'unknown')}",
        ]
        with contextlib.suppress(Exception):
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def run(settings: Settings) -> None:
    server_cfg = resolve_server_configuration(settings.server_url)
    settings.replays_dir.mkdir(parents=True, exist_ok=True)
    team_text = load_team_text(settings.team_path)
    accounts = build_account_pair(
        AccountConfiguration,
        username=settings.username,
        opponent=settings.opponent,
        password=settings.password,
    )
    player_one, player_two = make_players(
        team_text=team_text,
        battle_format=settings.battle_format,
        server_configuration=server_cfg,
        accounts=accounts,
    )
    await player_one.battle_against(player_two, n_battles=settings.n_battles)
    write_battle_logs(player_one, settings.replays_dir)
    print(
        f"Done. {settings.n_battles} battles on {settings.server_url} in {settings.battle_format}. "
        f"P1 won {player_one.n_won_battles} / {settings.n_battles}. Logs in {settings.replays_dir}."
    )


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    n_battles: int = typer.Option(2, help="Number of battles to play"),  # noqa: B008
    server_url: str = typer.Option("http://localhost:8000", help="Showdown server URL"),  # noqa: B008
    format: str = typer.Option("gen9doublesou", help="Battle format"),  # noqa: B008
    username: str = typer.Option("rl-bot-1", help="Primary bot username"),  # noqa: B008
    opponent: str = typer.Option("rl-bot-2", help="Opponent bot username"),  # noqa: B008
    password: str | None = typer.Option(None, help="Password (if server requires it)"),  # noqa: B008
    team: Path = typer.Option(Path("teams/gen9dou_fixed.txt"), help="Path to team text file"),  # noqa: B008
    replays_dir: Path = typer.Option(Path("replays"), help="Directory to write replay logs"),  # noqa: B008
):
    settings = Settings()
    settings.n_battles = n_battles
    settings.server_url = server_url
    settings.battle_format = format
    settings.username = username
    settings.opponent = opponent
    settings.password = password
    settings.team_path = team
    settings.replays_dir = replays_dir
    asyncio.run(run(settings))


if __name__ == "__main__":
    app()
