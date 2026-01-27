"""
Dataset parsing contract for imitation JSONL.

High-signal invariants:
- Training rows must contain `observation` (float list), `action` (len 2 int list), and `mask`
  (2 × act_size int list).
- Actions must be legal under the corresponding slot mask.
- The collector may write battle-level summary records into the same JSONL; scanning/training code
  skips records that do not satisfy this step-record schema.

See docs/codebase_overview.md and docs/DATA_SOURCES.md for examples.
"""

def _as_float_list(raw):
    if not isinstance(raw, list):
        raise ValueError
    return [float(value) for value in raw]


def _as_int_list(raw):
    if not isinstance(raw, list):
        raise ValueError
    return [int(value) for value in raw]


def _as_mask(raw):
    if not isinstance(raw, list):
        raise ValueError
    parsed: list[list[int]] = []
    width: int | None = None
    for slot in raw:
        if not isinstance(slot, list):
            raise ValueError
        slot_mask = [1 if int(entry) else 0 for entry in slot]
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


__all__ = ["_parse_payload"]
