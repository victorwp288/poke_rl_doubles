"""
Offline behavior cloning (BC) training loop.

Summary:
- Scans a JSONL dataset to compute mean/std normalization and byte offsets (streaming).
- Builds an indexed dataset that can train on files larger than RAM.
- Trains `BehaviorCloningPolicy` with masked cross-entropy loss (illegal actions are masked out).
- Exports:
  - `outputs/models/bc_policy*.pt` (checkpoint payload with state_dict + metadata)
  - `outputs/models/bc_stats*.json` (normalization stats used for parity/warmstart)
"""

import math
import random

import torch
from torch import nn

from .config import OfflineConfig
from .dataset import IndexedJsonlDataset, scan_samples, split_train_val
from .model import BehaviorCloningPolicy
from .trainer_io import _tensorboard_writer, _write_stats
from .trainer_steps import _evaluate, _train_epoch
from .utils import (
    _binary_feature_mask,
    _data_loader,
    _optimizer_steps_per_epoch,
    _pick_device,
)


class BehaviorCloningTrainer:
    def __init__(self, config: OfflineConfig):
        self.cfg = config
        self.device = _pick_device(config.device)
        torch.manual_seed(config.seed)
        random.seed(config.seed)

        self.scan = scan_samples(
            config.dataset_path,
            max_samples=config.max_samples,
            seed=config.seed,
            shuffle=config.shuffle,
            filters=config.filters,
        )
        train_offsets, val_offsets = split_train_val(self.scan.offsets, config.val_fraction)
        if not train_offsets:
            raise ValueError("training split must contain samples")

        self.train_offsets = train_offsets
        self.val_offsets = val_offsets
        self.train_dataset = IndexedJsonlDataset(
            self.scan.path,
            train_offsets,
            self.scan.mean,
            self.scan.std,
            weight_cfg=config.weighting,
        )
        self.val_dataset = (
            IndexedJsonlDataset(
                self.scan.path,
                val_offsets,
                self.scan.mean,
                self.scan.std,
                weight_cfg=config.weighting,
            )
            if val_offsets
            else None
        )

        sample_obs, _, sample_mask, _ = self.train_dataset[0]
        self.obs_dim = self.scan.obs_dim or len(sample_obs)
        self.act_dim = self.scan.action_dim or (len(sample_mask[0]) if sample_mask else 0)

        self.numeric_feature_mask = _binary_feature_mask(self.scan.binary_flags, self.device)

        self.train_loader = _data_loader(
            self.train_dataset, config.batch_size, self.device, True, config.num_workers
        )
        self.val_loader = _data_loader(
            self.val_dataset, config.batch_size, self.device, False, config.num_workers
        )

        self.writer = _tensorboard_writer(config.tensorboard_dir)

        self.model = BehaviorCloningPolicy(
            self.obs_dim,
            self.act_dim,
            hidden_dim=config.hidden_dim,
            hidden_layers=config.hidden_layers,
            dropout=config.dropout,
            attn_heads=config.attn_heads,
            slot_mlp_layers=config.slot_mlp_layers,
            head_mlp_dim=config.head_mlp_dim,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing, reduction="none")

        steps_per_epoch = len(self.train_loader) if self.train_loader is not None else 0
        self.optimizer_steps_per_epoch = _optimizer_steps_per_epoch(
            steps_per_epoch, config.grad_accum_steps
        )
        total_steps = self.optimizer_steps_per_epoch * config.epochs
        warmup_steps = config.lr_warmup_epochs * self.optimizer_steps_per_epoch

        self.scheduler = None
        if total_steps > 0 and (config.lr_cosine or warmup_steps):

            def lr_lambda(step):
                if warmup_steps and step < warmup_steps:
                    return float(step + 1) / float(max(1, warmup_steps))
                if not config.lr_cosine:
                    return 1.0
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.metrics: list[dict[str, float]] = []
        self.best_loss: float | None = None
        self.normalization = {
            "mean": self.scan.mean,
            "std": self.scan.std,
            "count": self.scan.count,
        }
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16

    def train_epoch(self):
        return _train_epoch(
            self.model,
            self.train_loader,
            self.device,
            self.loss_fn,
            self.optimizer,
            self.cfg.log_every,
            self.cfg.grad_clip_norm,
            self.cfg.obs_noise_std,
            self.numeric_feature_mask,
            self.scheduler,
            self.cfg.grad_accum_steps,
            self.use_amp,
            self.amp_dtype,
        )

    def evaluate(self):
        return _evaluate(
            self.model,
            self.val_loader,
            self.device,
            self.loss_fn,
            self.use_amp,
            self.amp_dtype,
        )

    def close(self):
        self.train_dataset.close()
        if self.val_dataset is not None:
            self.val_dataset.close()
        if self.writer is not None:
            self.writer.close()

    def train(self):
        config = self.cfg
        config.policy_path.parent.mkdir(parents=True, exist_ok=True)
        if config.stats_path:
            config.stats_path.parent.mkdir(parents=True, exist_ok=True)

        print(
            "offline setup: "
            f"total={len(self.scan.offsets)} train={len(self.train_offsets)} "
            f"val={len(self.val_offsets)} batch={config.batch_size} "
            f"epochs={config.epochs} device={self.device} "
            f"amp={config.use_amp}/{config.amp_dtype} grad_accum={config.grad_accum_steps}",
            flush=True,
        )

        try:
            self.optimizer.zero_grad(set_to_none=True)
            for epoch in range(1, config.epochs + 1):
                train_loss = self.train_epoch()
                val_loss = self.evaluate()
                self.metrics.append(
                    {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
                )
                if val_loss is None:
                    print(f"epoch {epoch}: train {train_loss:.4f}")
                else:
                    print(f"epoch {epoch}: train {train_loss:.4f} val {val_loss:.4f}")

                if self.writer is not None:
                    self.writer.add_scalar("loss/train", train_loss, epoch)
                    if val_loss is not None:
                        self.writer.add_scalar("loss/val", val_loss, epoch)
                    self.writer.flush()

                if val_loss is not None and (self.best_loss is None or val_loss < self.best_loss):
                    self.best_loss = val_loss
                    payload = {
                        "state_dict": self.model.state_dict(),
                        "config": config.as_dict(),
                        "obs_dim": self.obs_dim,
                        "action_dim": self.act_dim,
                        "normalization": self.normalization,
                    }
                    config.best_policy_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(payload, config.best_policy_path)
                    print(
                        f"saved best policy to {config.best_policy_path} (val_loss={val_loss:.4f})"
                    )
                    if config.best_stats_path is not None:
                        _write_stats(config.best_stats_path, self.normalization)
                        print(f"saved best normalization stats to {config.best_stats_path}")

            payload = {
                "state_dict": self.model.state_dict(),
                "config": config.as_dict(),
                "obs_dim": self.obs_dim,
                "action_dim": self.act_dim,
                "normalization": self.normalization,
            }
            torch.save(payload, config.policy_path)
            print(f"saved policy to {config.policy_path}")

            if config.stats_path:
                _write_stats(config.stats_path, self.normalization)
                print(f"saved normalization stats to {config.stats_path}")

            return self.metrics
        finally:
            self.close()


def train_offline(settings):
    trainer = BehaviorCloningTrainer(OfflineConfig(settings))
    return trainer.train()


__all__ = [
    "BehaviorCloningTrainer",
    "train_offline",
    "_optimizer_steps_per_epoch",
]
