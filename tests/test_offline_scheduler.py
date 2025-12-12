#!/usr/bin/env python3

import math
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.offline.trainer import _optimizer_steps_per_epoch


def _build_scheduler(base_lr, steps_per_epoch, grad_accum_steps, epochs, warmup_epochs, lr_cosine):
    # Mirrors train_offline scheduler setup
    optimizer_steps = _optimizer_steps_per_epoch(steps_per_epoch, grad_accum_steps)
    total_steps = optimizer_steps * epochs
    warmup_steps = warmup_epochs * optimizer_steps
    optimizer = torch.optim.Adam([nn.Parameter(torch.ones(1))], lr=base_lr)

    if total_steps == 0:
        return optimizer, None, optimizer_steps, warmup_steps

    def lr_lambda(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if not lr_cosine:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), optimizer_steps, warmup_steps


def test_scheduler_counts_use_optimizer_steps_with_accumulation():
    base_lr = 1e-3
    steps_per_epoch = 5  # mini-batches
    grad_accum_steps = 2
    epochs = 2
    warmup_epochs = 1

    optimizer, scheduler, optimizer_steps, warmup_steps = _build_scheduler(
        base_lr, steps_per_epoch, grad_accum_steps, epochs, warmup_epochs, lr_cosine=True
    )

    # With accumulation=2, we expect ceil(5/2)=3 optimizer steps per epoch.
    assert optimizer_steps == 3
    assert warmup_steps == 3  # one warmup epoch worth of optimizer steps

    lrs = []
    for _ in range(optimizer_steps * epochs):
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        lrs.append(optimizer.param_groups[0]["lr"])

    assert len(lrs) == 6  # 3 steps per epoch * 2 epochs
    # Warmup should finish at step index warmup_steps-1
    assert math.isclose(lrs[warmup_steps - 1], base_lr, rel_tol=1e-6)
    # Cosine decay should not have finished before the final step
    assert lrs[-1] < lrs[warmup_steps - 1]
