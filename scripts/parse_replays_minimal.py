#!/usr/bin/env python3
"""Skim Pokémon Showdown replays for easy-to-spot tactical hints."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

import typer

MOVE_LINE = re.compile(r"^\|move\|(?P<side>p[12])[ab]?: [^|]+\|(?P<move>[^|]+)")
SWITCH_LINE = re.compile(r"^\|switch\|(?P<side>p[12])(?P<slot>[ab]): ")
TURN_LINE = re.compile(r"^\|turn\|(?P<n>\d+)")


def _lines_from_blob(blob: bytes, ext: str) -> list[str]:
    if ext == ".json":
        try:
            payload = json.loads(blob.decode("utf-8", errors="ignore"))
        except Exception:
            payload = None
        if isinstance(payload, dict):
            for key in ("log", "logdata", "replay", "payload"):
                value = payload.get(key)
                if isinstance(value, str):
                    return [line for line in value.splitlines() if line.strip()]
    text = blob.decode("utf-8", errors="ignore")
    if "|init|battle" in text:
        text = text[text.index("|init|battle") :]
    return [line for line in text.splitlines() if line.strip()]


def iter_replay_files(raw_dir: Path) -> Iterable[Path]:
    for path in sorted(raw_dir.glob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".log", ".html", ""}:
            yield path


def parse_replay(path: Path) -> list[dict[str, object]]:
    tag = path.stem
    lines = _lines_from_blob(path.read_bytes(), path.suffix.lower())
    hints: list[dict[str, object]] = []
    turn: int | None = None
    for line in lines:
        turn_match = TURN_LINE.match(line)
        if turn_match:
            turn = int(turn_match.group("n"))
            continue
        move_match = MOVE_LINE.match(line)
        if move_match:
            if turn is None:
                continue
            move = move_match.group("move").strip().lower()
            side = move_match.group("side")
            if move in {"protect", "detect"}:
                hints.append(
                    {
                        "replay_id": tag,
                        "battle_tag": f"battle-{tag}",
                        "format": tag.split("-")[0],
                        "turn": turn,
                        "event": "protect",
                        "hint": "protect",
                        "side": side,
                    }
                )
            if move == "tailwind":
                hints.append(
                    {
                        "replay_id": tag,
                        "battle_tag": f"battle-{tag}",
                        "format": tag.split("-")[0],
                        "turn": turn,
                        "event": "tailwind",
                        "hint": "tailwind",
                        "side": side,
                    }
                )
            continue
        switch_match = SWITCH_LINE.match(line)
        if switch_match:
            if turn is None:
                continue
            hints.append(
                {
                    "replay_id": tag,
                    "battle_tag": f"battle-{tag}",
                    "format": tag.split("-")[0],
                    "turn": turn,
                    "event": "switch",
                    "hint": "switch",
                    "side": switch_match.group("side"),
                    "slot": switch_match.group("slot"),
                }
            )
    return hints


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    raw_dir: Path = typer.Option(Path("data/replays_raw"), help="Directory with raw replays"),  # noqa: B008
    out: Path = typer.Option(Path("data/human_hints.jsonl"), help="Output JSONL path"),  # noqa: B008
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    n_files = 0
    n_events = 0
    with out.open("w", encoding="utf-8") as handle:
        for replay_path in iter_replay_files(raw_dir):
            try:
                for event in parse_replay(replay_path):
                    handle.write(json.dumps(event) + "\n")
                    n_events += 1
                n_files += 1
            except Exception:
                continue
    print(f"Parsed {n_events} events from {n_files} files -> {out}")


if __name__ == "__main__":
    app()
