import threading
from pathlib import Path

import yaml  # type: ignore[import-untyped]

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "config" / "defaults.yaml"
_CACHE = None
_CACHE_PATH = None
_CACHE_LOCK = threading.Lock()


def _read_config(path):
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return data


def load(path=None):
    global _CACHE, _CACHE_PATH
    target = Path(path) if path else _CONFIG_PATH
    with _CACHE_LOCK:
        if _CACHE is None or target != _CACHE_PATH:
            _CACHE = _read_config(target)
            _CACHE_PATH = target
        cached = dict(_CACHE)
    return cached


def section(name, path=None):
    config = load(path)
    value = config.get(name, {})
    if isinstance(value, dict):
        return dict(value)
    return value
