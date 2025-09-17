#!/usr/bin/env python3
"""Collect imitation tuples from a heuristic teacher in Gen 9 Doubles."""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import typer
from poke_env.battle import DoubleBattle
from poke_env.environment.doubles_env import DoublesEnv
from poke_env.player import Player
from poke_env.player.baselines import (
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)

try:
    from src.utils.poke_env_utils import act_size_for_format, server_configuration_for_url
    from src.utils.teambuilders import (
        RotatingTeambuilder,
        constant_team_from_text,
        load_showdown_teams_from_dir,
        read_showdown_team,
    )
except ImportError:  # allow running as a script without installing the package
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.utils.poke_env_utils import act_size_for_format, server_configuration_for_url
    from src.utils.teambuilders import (
        RotatingTeambuilder,
        constant_team_from_text,
        load_showdown_teams_from_dir,
        read_showdown_team,
    )

STATUS_NAMES = ["SLP", "PAR", "BRN", "FRZ", "PSN", "TOX"]
TYPE_NAMES = [
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
]


def encode_status(mon) -> list[int]:
    onehot = [0] * len(STATUS_NAMES)
    if mon:
        status = getattr(mon, "status", None)
        name = getattr(status, "name", None)
        if isinstance(name, str) and name in STATUS_NAMES:
            onehot[STATUS_NAMES.index(name)] = 1
    return onehot


def encode_types(mon) -> list[int]:
    onehot = [0] * len(TYPE_NAMES)
    if mon is None:
        return onehot
    types = getattr(mon, "types", []) or []
    for t in types:
        name = getattr(t, "name", str(t)).upper()
        if name in TYPE_NAMES:
            onehot[TYPE_NAMES.index(name)] = 1
    return onehot


def hp_ratio(mon) -> float:
    if mon is None:
        return 0.0
    current = getattr(mon, "current_hp", None) or 0
    maximum = getattr(mon, "max_hp", None) or 0
    return float(current) / float(maximum) if maximum else 0.0


def per_slot_mask(battle: DoubleBattle, slot: int, act_size: int) -> list[int]:
    mask = [0] * act_size
    for action_idx in range(act_size):
        try:
            DoublesEnv._action_to_order_individual(np.int64(action_idx), battle, False, slot)
            mask[action_idx] = 1
        except AssertionError:
            mask[action_idx] = 0
        except Exception:
            mask[action_idx] = 0
    return mask


def action_to_tuple(order, battle: DoubleBattle) -> tuple[int, int]:
    arr = DoublesEnv.order_to_action(order, battle, fake=True, strict=False)
    pair = cast("tuple[int, int]", tuple(int(x) for x in arr))
    return pair[0], pair[1]


def encode_obs_v0(battle: DoubleBattle) -> list[float]:
    feats: list[float] = []
    slots = list(battle.active_pokemon) + list(battle.opponent_active_pokemon)
    for mon in slots:
        feats.append(hp_ratio(mon))
        feats.extend(encode_status(mon))
        feats.extend(encode_types(mon))
    return feats


class Recorder:
    def __init__(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = out_path.open("a", encoding="utf-8")

    def write(self, payload: dict[str, object]) -> None:
        self._fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._fh.close()


class RecordingHeuristics(SimpleHeuristicsPlayer):
    def __init__(self, recorder: Recorder, act_size: int, teacher_name: str, **kwargs):
        self._recorder = recorder
        self._act_size = act_size
        self._teacher_name = teacher_name
        super().__init__(**kwargs)

    def choose_move(self, battle: DoubleBattle):
        order = super().choose_move(battle)
        mask0 = per_slot_mask(battle, 0, self._act_size)
        mask1 = per_slot_mask(battle, 1, self._act_size)
        obs = encode_obs_v0(battle)
        first, second = action_to_tuple(order, battle)
        record = {
            "battle_tag": battle.battle_tag,
            "turn": battle.turn,
            "teacher": self._teacher_name,
            "format": battle.battle_tag.split("-")[1] if "-" in battle.battle_tag else None,
            "obs_v0": obs,
            "action": [first, second],
            "mask": [mask0, mask1],
        }
        self._recorder.write(record)
        return order


def make_player(kind: str, record: bool = False, **kwargs) -> Player:
    kind = kind.lower()
    if kind in {"simple", "heuristic", "simpleheuristics"}:
        if record:
            recorder = kwargs.pop("recorder")
            act_size = kwargs.pop("act_size")
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
    raise ValueError(f"Unknown player kind: {kind}")


@dataclass
class Settings:
    n_battles: int
    server_url: str
    battle_format: str
    our_team_path: Path
    opponent_teams_dir: Path
    teacher_kind: str
    opponents_kinds: list[str]
    out_path: Path


async def play_dataset(settings: Settings) -> None:
    act_size = act_size_for_format(settings.battle_format)
    server_cfg = server_configuration_for_url(settings.server_url)

    our_team_text = read_showdown_team(settings.our_team_path)
    opponents = [
        team
        for team in load_showdown_teams_from_dir(settings.opponent_teams_dir)
        if team.strip() and team.strip() != our_team_text.strip()
    ]
    if not opponents:
        opponents = [our_team_text]

    recorder = Recorder(settings.out_path)
    teacher = make_player(
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
            weights = [0.5, 0.3, 0.2][: len(settings.opponents_kinds)]
            opponent_kind = random.choices(settings.opponents_kinds, weights=weights)[0]
            opponent = make_player(
                opponent_kind,
                battle_format=settings.battle_format,
                team=RotatingTeambuilder(opponents),
                server_configuration=server_cfg,
                max_concurrent_battles=1,
            )
            try:
                await asyncio.wait_for(teacher.battle_against(opponent, n_battles=1), timeout=60)
            except TimeoutError:
                print(f"[warn] battle {battle_idx + 1}: timeout, probably a rejected team")
                continue
    finally:
        recorder.close()

    print(
        f"Collected {settings.n_battles} battles in {settings.out_path}. "
        f"Teacher={settings.teacher_kind} format={settings.battle_format} act_size={act_size}"
    )


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    n_battles: int = typer.Option(50, help="Number of battles to collect"),  # noqa: B008
    server_url: str = typer.Option("http://localhost:8000", help="Showdown server URL"),  # noqa: B008
    format: str = typer.Option("gen9doublesou", help="Battle format"),  # noqa: B008
    our_team: Path = typer.Option(Path("teams/gen9dou_fixed.txt"), help="Fixed team"),  # noqa: B008
    opponent_teams_dir: Path = typer.Option(Path("teams"), help="Directory with opponent teams"),  # noqa: B008
    teacher: str = typer.Option("simple", help="Teacher kind"),  # noqa: B008
    opponents: str = typer.Option("simple,maxbp,random", help="Opponent kinds"),  # noqa: B008
    out: Path = typer.Option(Path("data/imitation.jsonl"), help="Output JSONL path"),  # noqa: B008
) -> None:
    opponent_list = [token.strip() for token in opponents.split(",") if token.strip()]
    if not opponent_list:
        opponent_list = ["simple", "maxbp", "random"]
    settings = Settings(
        n_battles=n_battles,
        server_url=server_url,
        battle_format=format,
        our_team_path=our_team,
        opponent_teams_dir=opponent_teams_dir,
        teacher_kind=teacher,
        opponents_kinds=opponent_list,
        out_path=out,
    )
    asyncio.run(play_dataset(settings))


if __name__ == "__main__":
    app()
