#!/usr/bin/env python3
"""Collect imitation tuples from heuristic self-play in Gen 9 Doubles."""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from poke_env import (
    LocalhostServerConfiguration,
    ServerConfiguration,
    ShowdownServerConfiguration,
)
from poke_env.battle import DoubleBattle
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player import Player
from poke_env.player.baselines import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.env import action_space_size  # noqa: E402
from src.utils.teambuilders import (  # noqa: E402
    RotatingTeambuilder,
    constant_team_from_text,
    load_showdown_teams_from_dir,
    read_showdown_team,
)

STATUS_NAMES: tuple[str, ...] = ("SLP", "PAR", "BRN", "FRZ", "PSN", "TOX")
TYPE_NAMES: tuple[str, ...] = (
    "NORMAL",
    "FIRE",
    "WATER",
    "ELECTRIC",
    "GRASS",
    "ICE",
    "FIGHTING",
    "POISON",
    "GROUND",
    "FLYING",
    "PSYCHIC",
    "BUG",
    "ROCK",
    "GHOST",
    "DRAGON",
    "DARK",
    "STEEL",
    "FAIRY",
)


def _resolve_server_configuration(url: str) -> ServerConfiguration:
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


def _hp_ratio(mon: object | None) -> float:
    if mon is None:
        return 0.0
    current = getattr(mon, "current_hp", 0) or 0
    maximum = getattr(mon, "max_hp", 0) or 0
    return float(current) / float(maximum) if maximum else 0.0


def _encode_status(mon: object | None) -> list[int]:
    onehot = [0] * len(STATUS_NAMES)
    if mon is None:
        return onehot
    status = getattr(mon, "status", None)
    name = getattr(status, "name", None)
    if isinstance(name, str):
        with contextlib.suppress(ValueError):
            idx = STATUS_NAMES.index(name.upper())
            onehot[idx] = 1
    return onehot


def _encode_types(mon: object | None) -> list[int]:
    onehot = [0] * len(TYPE_NAMES)
    if mon is None:
        return onehot
    types = getattr(mon, "types", None) or []
    for entry in types:
        name = getattr(entry, "name", str(entry)).upper()
        with contextlib.suppress(ValueError):
            idx = TYPE_NAMES.index(name)
            onehot[idx] = 1
    return onehot


def _encode_obs_v0(battle: DoubleBattle) -> list[float]:
    feats: list[float] = []
    slots = list(battle.active_pokemon) + list(battle.opponent_active_pokemon)
    for mon in slots:
        feats.append(_hp_ratio(mon))
        feats.extend(_encode_status(mon))
        feats.extend(_encode_types(mon))
    return feats


def _per_slot_mask(battle: DoubleBattle, slot: int, act_size: int) -> list[int]:
    mask = [0] * act_size
    for action_idx in range(act_size):
        try:
            DoublesEnv._action_to_order_individual(np.int64(action_idx), battle, False, slot)
            mask[action_idx] = 1
        except Exception:
            mask[action_idx] = 0
    return mask


def _action_to_tuple(order, battle: DoubleBattle) -> tuple[int, int]:
    raw = DoublesEnv.order_to_action(order, battle, fake=True, strict=False)
    first, second = (int(raw[0]), int(raw[1]))
    return first, second


class Recorder:
    def __init__(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = out_path.open("a", encoding="utf-8")

    def __enter__(self) -> Recorder:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def write(self, payload: dict[str, object]) -> None:
        self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._handle.close()


class RecordingHeuristics(SimpleHeuristicsPlayer):
    def __init__(self, recorder: Recorder, act_size: int, teacher_name: str, **kwargs):
        self._recorder = recorder
        self._act_size = act_size
        self._teacher_name = teacher_name
        super().__init__(**kwargs)

    def choose_move(self, battle: DoubleBattle):
        order = super().choose_move(battle)
        mask0 = _per_slot_mask(battle, 0, self._act_size)
        mask1 = _per_slot_mask(battle, 1, self._act_size)
        first, second = _action_to_tuple(order, battle)
        record = {
            "battle_tag": battle.battle_tag,
            "turn": battle.turn,
            "teacher": self._teacher_name,
            "format": battle.format,
            "obs_v0": _encode_obs_v0(battle),
            "action": [first, second],
            "mask": [mask0, mask1],
        }
        self._recorder.write(record)
        return order


def _make_player(kind: str, *, record: bool = False, **kwargs) -> Player:
    kind = kind.lower()
    if kind in {"simple", "heuristic", "simpleheuristics"}:
        if record:
            recorder: Recorder = kwargs.pop("recorder")
            act_size: int = kwargs.pop("act_size")
            return RecordingHeuristics(
                recorder=recorder,
                act_size=act_size,
                teacher_name="SimpleHeuristicsPlayer",
                **kwargs,
            )
        return SimpleHeuristicsPlayer(**kwargs)
    if kind in {"maxbp", "maxbasepower"}:
        return MaxBasePowerPlayer(**kwargs)
    if kind == "random":
        return RandomPlayer(**kwargs)
    raise ValueError(f"unknown player kind: {kind}")


@dataclass(slots=True)
class Settings:
    n_battles: int
    server_url: str
    battle_format: str
    our_team_path: Path
    opponent_teams_dir: Path
    teacher_kind: str
    opponents_kinds: list[str]
    out_path: Path
    battle_timeout: float = 60.0


async def play_dataset(settings: Settings) -> None:
    act_size = action_space_size(settings.battle_format)
    server_cfg = _resolve_server_configuration(settings.server_url)

    our_team_text = read_showdown_team(settings.our_team_path)
    opponent_pool = [
        text
        for text in load_showdown_teams_from_dir(settings.opponent_teams_dir)
        if text.strip() and text.strip() != our_team_text.strip()
    ]
    if not opponent_pool:
        opponent_pool = [our_team_text]

    def _cleanup(player: Player) -> None:
        with contextlib.suppress(Exception):
            player.reset_battles()
        with contextlib.suppress(Exception):
            close_fn = getattr(player, "close", None)
            if callable(close_fn):
                close_fn()

    with Recorder(settings.out_path) as recorder:
        teacher = _make_player(
            settings.teacher_kind,
            record=True,
            recorder=recorder,
            act_size=act_size,
            battle_format=settings.battle_format,
            team=constant_team_from_text(our_team_text),
            server_configuration=server_cfg,
            max_concurrent_battles=1,
        )

        try:
            for battle_idx in range(settings.n_battles):
                opponent_kind = random.choice(settings.opponents_kinds)
                opponent = _make_player(
                    opponent_kind,
                    battle_format=settings.battle_format,
                    team=RotatingTeambuilder(opponent_pool),
                    server_configuration=server_cfg,
                    max_concurrent_battles=1,
                )
                print(
                    f"[info] starting battle {battle_idx + 1}/{settings.n_battles} vs {opponent_kind}",
                    flush=True,
                )
                try:
                    await asyncio.wait_for(
                        teacher.battle_against(opponent, n_battles=1),
                        timeout=settings.battle_timeout,
                    )
                    print(
                        f"[info] finished battle {battle_idx + 1}/{settings.n_battles}",
                        flush=True,
                    )
                except TimeoutError:
                    print(f"[warn] battle {battle_idx + 1}: timeout (likely rejected team)")
                finally:
                    _cleanup(opponent)
        finally:
            _cleanup(teacher)

    print(
        f"Collected {settings.n_battles} battles at {settings.out_path}. "
        f"Teacher={settings.teacher_kind} format={settings.battle_format} act_size={act_size}"
    )


DEFAULT_SETTINGS = Settings(
    n_battles=50,
    server_url="http://localhost:8000",
    battle_format="gen9doublesou",
    our_team_path=Path("teams/gen9dou_fixed.txt"),
    opponent_teams_dir=Path("teams"),
    teacher_kind="simple",
    opponents_kinds=["simple", "maxbp", "random"],
    out_path=Path("data/processed/imitation.jsonl"),
    battle_timeout=60.0,
)


# Run the asynchronous collection loop with the provided settings.
def collect_imitation(settings: Settings = DEFAULT_SETTINGS) -> None:
    asyncio.run(play_dataset(settings))


def main() -> None:
    # Edit DEFAULT_SETTINGS above or call collect_imitation with custom Settings.
    collect_imitation(DEFAULT_SETTINGS)


if __name__ == "__main__":
    main()
