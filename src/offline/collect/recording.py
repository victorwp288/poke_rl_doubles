import contextlib
import json


class Recorder:
    def __init__(self, out_path, *, rotate_every=None):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._base_path = out_path
        self._path = out_path
        self._handle = out_path.open("a", encoding="utf-8")
        self._written = 0
        self._rotate_every = rotate_every
        self._since_rotate = 0

    @property
    def written(self):
        return self._written

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def write(self, payload):
        self._handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._handle.flush()
        self._written += 1
        self._since_rotate += 1
        if self._rotate_every and self._since_rotate >= self._rotate_every:
            self.rotate()

    def close(self):
        with contextlib.suppress(Exception):
            self._handle.close()

    def rotate(self):
        self.close()
        suffix = self._base_path.suffix or ".jsonl"
        stem_path = self._base_path.with_suffix("")
        rotated = stem_path.with_name(f"{stem_path.name}.{self._written:07d}{suffix}")
        with contextlib.suppress(FileNotFoundError):
            self._base_path.rename(rotated)
        self._handle = self._base_path.open("a", encoding="utf-8")
        self._since_rotate = 0


def _battle_summary(battle, outcome, opponent_kind, settings, *, timeout=False, error=None):
    result = "draw"
    if getattr(battle, "won", False):
        result = "win"
    elif getattr(battle, "lost", False):
        result = "loss"
    elif outcome:
        result = outcome
    return {
        "battle_tag": battle.battle_tag,
        "result": result,
        "turns": getattr(battle, "turn", None),
        "timeout": bool(timeout),
        "error": error,
        "opponent": opponent_kind,
        "teacher": settings.teacher_kind,
        "format": getattr(battle, "format", None),
    }


__all__ = ["Recorder", "_battle_summary"]
