# Helpers to read the imitation dataset JSONL files

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ImitationSample:
    observation: list[float]
    actions: list[int]
    mask: list[list[int]]


def _coerce_observation(raw_obs: object) -> list[float] | None:
    if not isinstance(raw_obs, list):
        return None
    try:
        return [float(value) for value in raw_obs]
    except (TypeError, ValueError):
        return None


def _coerce_actions(raw_actions: object) -> list[int] | None:
    if not isinstance(raw_actions, list):
        return None
    try:
        choices = [int(value) for value in raw_actions]
    except (TypeError, ValueError):
        return None
    if len(choices) != 2:
        return None
    return choices


def _coerce_mask(raw_mask: object) -> list[list[int]] | None:
    if not isinstance(raw_mask, list):
        return None
    parsed: list[list[int]] = []
    width: int | None = None
    for slot in raw_mask:
        if not isinstance(slot, list):
            return None
        slot_mask: list[int] = []
        for entry in slot:
            try:
                slot_mask.append(1 if int(entry) else 0)
            except (TypeError, ValueError):
                return None
        if width is None:
            width = len(slot_mask)
        elif len(slot_mask) != width:
            return None
        parsed.append(slot_mask)
    return parsed


def _parse_payload(payload: dict[str, object]) -> ImitationSample | None:
    observation = _coerce_observation(payload.get("obs_v0"))
    actions = _coerce_actions(payload.get("action"))
    mask = _coerce_mask(payload.get("mask"))
    if observation is None or actions is None or mask is None:
        return None
    for choice, slot_mask in zip(actions, mask, strict=False):
        if not (0 <= choice < len(slot_mask)) or not slot_mask[choice]:
            return None
    return ImitationSample(observation=observation, actions=actions, mask=mask)


def load_samples(
    dataset_path: Path,
    *,
    max_samples: int | None = None,
    seed: int = 0,
    shuffle: bool = True,
) -> list[ImitationSample]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples: list[ImitationSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                sample = _parse_payload(payload)
                if sample:
                    samples.append(sample)

    if not samples:
        raise ValueError(f"No usable records in {path}")

    if shuffle:
        random.Random(seed).shuffle(samples)

    if max_samples and max_samples > 0:
        samples = samples[:max_samples]

    return samples


def split_train_val(
    samples: Iterable[ImitationSample], *, val_fraction: float
) -> tuple[list[ImitationSample], list[ImitationSample]]:
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


__all__ = ["ImitationSample", "load_samples", "split_train_val"]
