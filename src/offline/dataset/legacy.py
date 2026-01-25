import json
import random
from pathlib import Path

from .parsing import _parse_payload


def load_samples(dataset_path, max_samples=None, seed=0, shuffle=True):
    """Legacy helper retained for compatibility with older tooling."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"missing dataset: {path}")
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                sample = _parse_payload(payload)
            except Exception:
                continue
            samples.append(sample)
            if max_samples and len(samples) >= max_samples:
                break
    if not samples:
        raise ValueError(f"no usable records in {path}")
    if shuffle:
        random.Random(seed).shuffle(samples)
    return samples


def split_train_val(samples, val_fraction):
    items = list(samples)
    if not items or val_fraction <= 0:
        return items, []
    if len(items) == 1:
        return items, []
    val_count = max(1, int(len(items) * val_fraction))
    val_count = min(val_count, len(items) - 1)
    val_split = items[:val_count]
    train_split = items[val_count:]
    return train_split, val_split


__all__ = ["load_samples", "split_train_val"]
