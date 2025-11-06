import json
import random
from pathlib import Path


def _as_float_list(raw):
    if not isinstance(raw, list):
        raise ValueError
    values = []
    for value in raw:
        values.append(float(value))
    return values


def _as_int_list(raw):
    if not isinstance(raw, list):
        raise ValueError
    values = []
    for value in raw:
        values.append(int(value))
    return values


def _as_mask(raw):
    if not isinstance(raw, list):
        raise ValueError
    parsed = []
    width = None
    for slot in raw:
        if not isinstance(slot, list):
            raise ValueError
        slot_mask = []
        for entry in slot:
            slot_mask.append(1 if int(entry) else 0)
        if width is None:
            width = len(slot_mask)
        elif len(slot_mask) != width:
            raise ValueError
        parsed.append(slot_mask)
    if not parsed:
        raise ValueError
    return parsed


def _valid_actions(actions, mask):
    if len(actions) != len(mask):
        return False
    for choice, slot_mask in zip(actions, mask, strict=False):
        if choice < 0 or choice >= len(slot_mask):
            return False
        if not slot_mask[choice]:
            return False
    return True


def _parse_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError
    observation = _as_float_list(payload.get("observation"))
    actions = _as_int_list(payload.get("action"))
    mask = _as_mask(payload.get("mask"))
    if len(actions) != 2:
        raise ValueError
    if not _valid_actions(actions, mask):
        raise ValueError
    return observation, actions, mask


def load_samples(dataset_path, max_samples=None, seed=0, shuffle=True):
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
    if not samples:
        raise ValueError(f"no usable records in {path}")
    if shuffle:
        random.Random(seed).shuffle(samples)
    if max_samples and max_samples > 0:
        samples = samples[:max_samples]
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
