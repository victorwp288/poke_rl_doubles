import argparse
import json
import re
from pathlib import Path

from src.config import section

MOVE_LINE = re.compile(r"^\|move\|(?P<side>p[12])(?P<slot>[ab])?: [^|]+\|(?P<move>[^|]+)")
SWITCH_LINE = re.compile(r"^\|switch\|(?P<side>p[12])(?P<slot>[ab]): ")
TURN_LINE = re.compile(r"^\|turn\|(?P<n>\d+)")
PROTECT_MOVES = {"protect", "detect", "kingsshield", "silktrap", "spikyshield"}
TAILWIND_MOVES = {"tailwind"}


def _lines_from_blob(blob, ext):
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


def iter_replay_files(raw_dir):
    for path in sorted(raw_dir.glob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".log", ".html", ""}:
            yield path


def _normalise_battle_tag(tag):
    return tag if tag.startswith("battle-") else f"battle-{tag}"


def _infer_format(tag):
    parts = tag.split("-")
    if not parts:
        return None
    if parts[0] == "battle" and len(parts) >= 2:
        return parts[1]
    return parts[0]


def _make_hint(turn, event, hint, side, slot):
    payload = {"turn": turn, "event": event, "hint": hint, "side": side}
    if slot is not None:
        payload["slot"] = slot
    return payload


def _hint_from_move(turn, line):
    move = line.group("move").strip().lower()
    slot = line.group("slot")
    side = line.group("side")
    if slot is None:
        return []
    if move in PROTECT_MOVES:
        return [_make_hint(turn, "protect", "protect", side, slot)]
    if move in TAILWIND_MOVES:
        return [_make_hint(turn, "tailwind", "tailwind", side, slot)]
    return []


def _hint_from_switch(turn, line):
    slot = line.group("slot")
    if slot is None:
        return []
    return [_make_hint(turn, "switch", "switch", line.group("side"), slot)]


def parse_replay(path):
    tag = path.stem
    battle_tag = _normalise_battle_tag(tag)
    fmt = _infer_format(tag)
    lines = _lines_from_blob(path.read_bytes(), path.suffix.lower())
    hints = []
    turn = None
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
    payloads = []
    for event in hints:
        entry = {
            "replay_id": tag,
            "battle_tag": battle_tag,
            "format": fmt,
            "turn": event["turn"],
            "event": event["event"],
            "hint": event["hint"],
            "side": event["side"],
        }
        if "slot" in event:
            entry["slot"] = event["slot"]
        payloads.append(entry)
    return payloads


def load_settings():
    config = section("data_parse")
    return {
        "raw_dir": Path(config.get("raw_dir", "data/raw/downloaded")),
        "out_path": Path(config.get("out_path", "data/processed/human_hints.jsonl")),
        "focus_side": config.get("focus_side"),
    }


def build_arg_parser(defaults):
    parser = argparse.ArgumentParser(description="Parse tactical hints from showdown replay logs")

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=defaults.get("raw_dir", Path("data/raw/downloaded")),
        help="Directory containing raw replay files",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=defaults.get("out_path", Path("data/processed/human_hints.jsonl")),
        help="Output path for parsed hints",
    )
    parser.add_argument(
        "--focus-side",
        type=str,
        choices=["p1", "p2"],
        default=defaults.get("focus_side"),
        help="If set, only include hints for the specified side",
    )

    return parser


def merge_cli_overrides(defaults, args):
    settings = defaults.copy()

    for key in ["raw_dir", "out_path", "focus_side"]:
        val = getattr(args, key)
        if val is not None:
            settings[key] = val

    return settings


def main(argv: list[str] | None = None):
    _ = argv
    config = section("data_parse")

    raw_dir = Path(config.get("raw_dir", "data/raw/downloaded"))
    out_path = Path(config.get("out_path", "data/processed/human_hints.jsonl"))
    focus_side = config.get("focus_side")

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


__all__ = ["main", "parse_replay"]
