import math

import torch
from torch.utils.data import DataLoader


def _batch_collate(batch):
    # samples already come as tensors; stack to preserve shapes and avoid scalar conversion
    observations = torch.stack([sample[0] for sample in batch], dim=0)
    actions = torch.stack([sample[1] for sample in batch], dim=0)
    mask = torch.stack([sample[2] for sample in batch], dim=0)
    weights = None
    if batch and len(batch[0]) > 3:
        weights = torch.stack([sample[3] for sample in batch], dim=0)
    return observations, actions, mask, weights


def _binary_feature_mask(binary_flags, device):
    if not binary_flags:
        return None
    numeric_flags = [not flag for flag in binary_flags]
    if not any(numeric_flags):
        return None
    return torch.tensor(numeric_flags, dtype=torch.float32, device=device)


def _optimizer_steps_per_epoch(steps_per_epoch, grad_accum_steps):
    if steps_per_epoch <= 0:
        return 0
    return math.ceil(steps_per_epoch / max(1, grad_accum_steps))


def _pick_device(preference):
    if preference:
        return torch.device(preference)
    checks = [("cuda", torch.cuda.is_available)]
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None:
        checks.append(("mps", mps_backend.is_available))
    for name, available in checks:
        if available():
            return torch.device(name)
    return torch.device("cpu")


def _slot_loss(logits, actions, mask, loss_fn, sample_weights=None):
    # Ensure the target action is always legal and avoid all-false rows.
    safe_mask = mask.clone()
    for idx in range(safe_mask.shape[1]):  # per slot
        slot_mask = safe_mask[:, idx]
        target = actions[:, idx].unsqueeze(1)
        # allow the target action even if original mask said False
        slot_mask.scatter_(1, target, True)
        # if still all-false, allow all actions for that row
        missing = ~slot_mask.any(dim=-1, keepdim=True)
        if missing.any():
            slot_mask |= missing
        safe_mask[:, idx] = slot_mask

    masked = logits.masked_fill(~safe_mask, -1e9)
    losses = []
    for index, slot in enumerate(masked.unbind(dim=1)):
        raw_loss = loss_fn(slot, actions[:, index])
        if sample_weights is not None:
            raw_loss = raw_loss * sample_weights
        losses.append(raw_loss.mean())
    return torch.stack(losses).mean()


def _data_loader(dataset, batch_size, device, shuffle, num_workers):
    if dataset is None or len(dataset) == 0:
        return None
    pin_memory = getattr(device, "type", "cpu") == "cuda"
    persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_batch_collate,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )


__all__ = [
    "_batch_collate",
    "_binary_feature_mask",
    "_data_loader",
    "_optimizer_steps_per_epoch",
    "_pick_device",
    "_slot_loss",
]
