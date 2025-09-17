from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path

from poke_env.teambuilder import ConstantTeambuilder, Teambuilder, TeambuilderPokemon


def read_showdown_team(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_showdown_teams_from_dir(directory: Path) -> list[str]:
    teams: list[str] = []
    for file_path in sorted(directory.glob("*.txt")):
        try:
            team = read_showdown_team(file_path)
        except OSError:
            continue
        if team:
            teams.append(team)
    return teams


class RotatingTeambuilder(Teambuilder):
    def __init__(self, teams: Iterable[str]):
        parsed: list[list[TeambuilderPokemon]] = []
        for raw in teams:
            if not raw or not raw.strip():
                continue
            try:
                mons = self.parse_packed_team(raw) if "|" in raw else self.parse_showdown_team(raw)
            except Exception:
                continue
            if mons:
                parsed.append(mons)
        if not parsed:
            raise ValueError("RotatingTeambuilder needs at least one valid team text")
        self._teams = [self.join_team(mons) for mons in parsed]

    def yield_team(self) -> str:
        return random.choice(self._teams)


def constant_team_from_text(team_text: str) -> ConstantTeambuilder:
    return ConstantTeambuilder(team_text)
