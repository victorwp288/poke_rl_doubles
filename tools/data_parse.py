#!/usr/bin/env python3
# Collect tactical hints from showdown replay logs

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

MOVE_LINE = re.compile(r"^\|move\|(?P<side>p[12])(?P<slot>[ab])?: [^|]+\|(?P<move>[^|]+)")
SWITCH_LINE = re.compile(r"^\|switch\|(?P<side>p[12])(?P<slot>[ab]): ")
TURN_LINE = re.compile(r"^\|turn\|(?P<n>\d+)")
PROTECT_MOVES = {"protect", "detect", "kingsshield", "silktrap", "spikyshield"}
TAILWIND_MOVES = {"tailwind"}


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


def iter_replay_files(raw_dir: Path) -> Iterator[Path]:
    for path in sorted(raw_dir.glob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".log", ".html", ""}:
            yield path


def _normalise_battle_tag(tag: str) -> str:
    return tag if tag.startswith("battle-") else f"battle-{tag}"


def _infer_format(tag: str) -> str | None:
    parts = tag.split("-")
    if not parts:
        return None
    if parts[0] == "battle" and len(parts) >= 2:
        return parts[1]
    return parts[0]


@dataclass(slots=True)
class HintEvent:
    turn: int
    event: str
    hint: str
    side: str
    slot: str | None

    def to_json(self, replay_id: str, battle_tag: str, fmt: str | None) -> dict[str, object]:
        payload: dict[str, object] = {
            "replay_id": replay_id,
            "battle_tag": battle_tag,
            "format": fmt,
            "turn": self.turn,
            "event": self.event,
            "hint": self.hint,
            "side": self.side,
        }
        if self.slot is not None:
            payload["slot"] = self.slot
        return payload


def _hint_from_move(turn: int, line: re.Match[str]) -> Iterable[HintEvent]:
    move = line.group("move").strip().lower()
    slot = line.group("slot")
    side = line.group("side")
    if slot is None:
        return []
    if move in PROTECT_MOVES:
        return [HintEvent(turn, event="protect", hint="protect", side=side, slot=slot)]
    if move in TAILWIND_MOVES:
        return [HintEvent(turn, event="tailwind", hint="tailwind", side=side, slot=slot)]
    return []


def _hint_from_switch(turn: int, line: re.Match[str]) -> Iterable[HintEvent]:
    slot = line.group("slot")
    if slot is None:
        return []
    return [
        HintEvent(
            turn,
            event="switch",
            hint="switch",
            side=line.group("side"),
            slot=slot,
        )
    ]


def parse_replay(path: Path) -> list[HintEvent]:
    tag = path.stem
    battle_tag = _normalise_battle_tag(tag)
    fmt = _infer_format(tag)
    lines = _lines_from_blob(path.read_bytes(), path.suffix.lower())
    hints: list[HintEvent] = []
    turn: int | None = None
    for line in lines:
        turn_match = TURN_LINE.match(line)
        if turn_match:
            turn = int(turn_match.group("n"))
            continue
        if turn is None:
            continue
        move_match = MOVE_LINE.match(line)
        if move_match:
            hints.extend(_hint_from_move(turn, move_match))
            continue
        switch_match = SWITCH_LINE.match(line)
        if switch_match:
            hints.extend(_hint_from_switch(turn, switch_match))
    return [event.to_json(tag, battle_tag, fmt) for event in hints]


def main() -> None:
    raw_dir = Path("data/raw/downloaded")
    out_path = Path("data/processed/human_hints.jsonl")
    focus_side = "p1"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_files = 0
    n_events = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for replay_path in iter_replay_files(raw_dir):
            try:
                events = parse_replay(replay_path)
            except Exception:
                continue
            if not events:
                continue
            for event in events:
                if focus_side is not None and event.get("side") != focus_side:
                    continue
                handle.write(json.dumps(event) + "\n")
                n_events += 1
            n_files += 1
    print(f"parsed {n_events} events from {n_files} files -> {out_path}")


if __name__ == "__main__":
    main()
