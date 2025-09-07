from pathlib import Path


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)
