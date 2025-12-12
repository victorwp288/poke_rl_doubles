import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import IndexedJsonlDataset, scan_samples, split_train_val
from .model import BehaviorCloningPolicy


class OfflineConfig:
    def __init__(self, source):
        data = dict(source or {})
        self.raw = dict(data)
        self.device = data.get("device")
        self.seed = int(data.get("seed", 0))
        self.num_workers = int(data.get("num_workers", 0))
        self.dataset_path = data.get("dataset_path")
        self.max_samples = data.get("max_samples")
        self.shuffle = bool(data.get("shuffle", True))
        self.val_fraction = float(data.get("val_fraction", 0.0))
        batch_size = data.get("batch_size")
        self.batch_size = int(batch_size) if batch_size is not None else 32
        hidden_dim = data.get("hidden_dim")
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else 384
        hidden_layers = data.get("hidden_layers")
        self.hidden_layers = int(hidden_layers) if hidden_layers is not None else 3
        dropout = data.get("dropout")
        self.dropout = float(dropout) if dropout is not None else 0.0
        learning_rate = data.get("learning_rate")
        self.learning_rate = float(learning_rate) if learning_rate is not None else 1e-3
        weight_decay = data.get("weight_decay")
        self.weight_decay = float(weight_decay) if weight_decay is not None else 0.0
        label_smoothing = data.get("label_smoothing")
        self.label_smoothing = float(label_smoothing) if label_smoothing is not None else 0.0
        obs_noise_std = data.get("obs_noise_std")
        self.obs_noise_std = float(obs_noise_std) if obs_noise_std is not None else 0.0
        grad_clip = data.get("grad_clip_norm")
        self.grad_clip_norm = float(grad_clip) if grad_clip is not None else None
        self.log_every = int(data.get("log_every") or 0)
        self.epochs = int(data.get("epochs") or 1)
        self.filters = data.get("filters") or {}
        self.weighting = data.get("weighting") or {}
        self.lr_warmup_epochs = int(data.get("lr_warmup_epochs") or 0)
        self.lr_cosine = bool(data.get("lr_cosine", True))
        self.grad_accum_steps = int(data.get("grad_accum_steps") or 1)
        amp_dtype = str(data.get("amp_dtype") or "bf16").lower()
        self.amp_dtype = "bf16" if amp_dtype not in {"fp16", "float16"} else "fp16"
        self.use_amp = bool(data.get("use_amp", True))
        attn_heads = data.get("attn_heads")
        self.attn_heads = int(attn_heads) if attn_heads is not None else 4
        slot_layers = data.get("slot_mlp_layers")
        self.slot_mlp_layers = int(slot_layers) if slot_layers is not None else 1
        head_mlp_dim = data.get("head_mlp_dim")
        self.head_mlp_dim = int(head_mlp_dim) if head_mlp_dim is not None else 512
        policy_path = data.get("policy_path") or "outputs/models/bc_policy.pt"
        self.policy_path = Path(policy_path)
        stats_path = data.get("stats_path")
        self.stats_path = Path(stats_path) if stats_path else None
        best_policy = data.get("best_policy_path")
        if best_policy:
            self.best_policy_path = Path(best_policy)
        else:
            self.best_policy_path = self.policy_path.with_name(
                f"{self.policy_path.stem}_best{self.policy_path.suffix}"
            )
        best_stats = data.get("best_stats_path")
        if best_stats:
            self.best_stats_path = Path(best_stats)
        elif self.stats_path is not None:
            self.best_stats_path = self.stats_path.with_name(
                f"{self.stats_path.stem}_best{self.stats_path.suffix}"
            )
        else:
            self.best_stats_path = None
        tensorboard_dir = data.get("tensorboard_dir")
        self.tensorboard_dir = Path(tensorboard_dir) if tensorboard_dir else None

    def as_dict(self):
        return dict(self.raw)


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


def _train_epoch(
    model,
    loader,
    device,
    loss_fn,
    optimizer,
    log_every,
    grad_clip_norm,
    obs_noise_std,
    numeric_feature_mask,
    scheduler,
    grad_accum_steps,
    use_amp,
    amp_dtype,
):
    model.train()
    total_loss = 0.0
    total_items = 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    steps = 0
    total_steps = len(loader)
    remainder = total_steps % grad_accum_steps
    last_window = remainder if remainder > 0 else grad_accum_steps
    last_window_start = total_steps - last_window + 1

    for step, (obs, actions, mask, weights) in enumerate(loader, 1):
        steps = step
        obs = obs.to(device)
        actions = actions.to(device)
        mask = mask.to(device)
        if weights is not None:
            weights = weights.to(device)
        if obs_noise_std:
            noise = torch.randn_like(obs) * obs_noise_std
            if numeric_feature_mask is not None:
                noise = noise * numeric_feature_mask
            inputs = obs + noise
        else:
            inputs = obs
        is_step = step % grad_accum_steps == 0
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            loss = _slot_loss(model(inputs), actions, mask, loss_fn, sample_weights=weights)
            # Use the actual accumulation window size for the final partial window
            accum_divisor = (
                last_window if (remainder > 0 and step >= last_window_start) else grad_accum_steps
            )
            loss = loss / accum_divisor

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if is_step:
            if grad_clip_norm:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * accum_divisor * obs.size(0)
        total_items += obs.size(0)
        if log_every and step % log_every == 0:
            print(f"step {step}: loss {loss.item() * accum_divisor:.4f}")
    remainder = steps % grad_accum_steps
    if remainder:
        if grad_clip_norm:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
    return total_loss / max(1, total_items)


def _evaluate(model, loader, device, loss_fn, use_amp, amp_dtype):
    if loader is None:
        return None
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for obs, actions, mask, weights in loader:
            obs = obs.to(device)
            actions = actions.to(device)
            mask = mask.to(device)
            if weights is not None:
                weights = weights.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                loss = _slot_loss(model(obs), actions, mask, loss_fn, sample_weights=weights)
            total_loss += loss.item() * obs.size(0)
            total_items += obs.size(0)
    if total_items == 0:
        return None
    return total_loss / total_items


def _tensorboard_writer(path):
    if not path:
        return None
    path.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        print(f"[warn] tensorboard unavailable: {exc}")
        return None
    print(f"tensorboard logging -> {path}")
    return SummaryWriter(log_dir=str(path))


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


def _write_stats(path, stats):
    def _to_jsonable(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(stats, default=_to_jsonable, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def train_offline(settings):
    config = OfflineConfig(settings)
    device = _pick_device(config.device)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    scan = scan_samples(
        config.dataset_path,
        max_samples=config.max_samples,
        seed=config.seed,
        shuffle=config.shuffle,
        filters=config.filters,
    )
    train_offsets, val_offsets = split_train_val(scan.offsets, config.val_fraction)
    if not train_offsets:
        raise ValueError("training split must contain samples")

    train_dataset = IndexedJsonlDataset(
        scan.path,
        train_offsets,
        scan.mean,
        scan.std,
        weight_cfg=config.weighting,
    )
    val_dataset = (
        IndexedJsonlDataset(
            scan.path, val_offsets, scan.mean, scan.std, weight_cfg=config.weighting
        )
        if val_offsets
        else None
    )

    sample_obs, _, sample_mask, _ = train_dataset[0]
    obs_dim = scan.obs_dim or len(sample_obs)
    act_dim = scan.action_dim or (len(sample_mask[0]) if sample_mask else 0)

    numeric_feature_mask = _binary_feature_mask(scan.binary_flags, device)

    train_loader = _data_loader(train_dataset, config.batch_size, device, True, config.num_workers)
    val_loader = _data_loader(val_dataset, config.batch_size, device, False, config.num_workers)

    writer = _tensorboard_writer(config.tensorboard_dir)

    model = BehaviorCloningPolicy(
        obs_dim,
        act_dim,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
        attn_heads=config.attn_heads,
        slot_mlp_layers=config.slot_mlp_layers,
        head_mlp_dim=config.head_mlp_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing, reduction="none")

    steps_per_epoch = len(train_loader) if train_loader is not None else 0
    optimizer_steps_per_epoch = _optimizer_steps_per_epoch(
        steps_per_epoch, config.grad_accum_steps
    )
    total_steps = optimizer_steps_per_epoch * config.epochs
    warmup_steps = config.lr_warmup_epochs * optimizer_steps_per_epoch

    scheduler = None
    if total_steps > 0 and (config.lr_cosine or warmup_steps):

        def lr_lambda(step):
            if warmup_steps and step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            if not config.lr_cosine:
                return 1.0
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    metrics = []
    best_loss = None
    normalization = {"mean": scan.mean, "std": scan.std, "count": scan.count}

    config.policy_path.parent.mkdir(parents=True, exist_ok=True)
    if config.stats_path:
        config.stats_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "offline setup: "
        f"total={len(scan.offsets)} train={len(train_offsets)} val={len(val_offsets)} "
        f"batch={config.batch_size} epochs={config.epochs} device={device} "
        f"amp={config.use_amp}/{config.amp_dtype} grad_accum={config.grad_accum_steps}",
        flush=True,
    )

    try:
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, config.epochs + 1):
            train_loss = _train_epoch(
                model,
                train_loader,
                device,
                loss_fn,
                optimizer,
                config.log_every,
                config.grad_clip_norm,
                config.obs_noise_std,
                numeric_feature_mask,
                scheduler,
                config.grad_accum_steps,
                config.use_amp and device.type == "cuda",
                torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16,
            )
            val_loss = _evaluate(
                model,
                val_loader,
                device,
                loss_fn,
                config.use_amp and device.type == "cuda",
                torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16,
            )
            metrics.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if val_loss is None:
                print(f"epoch {epoch}: train {train_loss:.4f}")
            else:
                print(f"epoch {epoch}: train {train_loss:.4f} val {val_loss:.4f}")

            if writer is not None:
                writer.add_scalar("loss/train", train_loss, epoch)
                if val_loss is not None:
                    writer.add_scalar("loss/val", val_loss, epoch)
                writer.flush()

            if val_loss is not None and (best_loss is None or val_loss < best_loss):
                best_loss = val_loss
                payload = {
                    "state_dict": model.state_dict(),
                    "config": config.as_dict(),
                    "obs_dim": obs_dim,
                    "action_dim": act_dim,
                    "normalization": normalization,
                }
                config.best_policy_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, config.best_policy_path)
                print(f"saved best policy to {config.best_policy_path} (val_loss={val_loss:.4f})")
                if config.best_stats_path is not None:
                    _write_stats(config.best_stats_path, normalization)
                    print(f"saved best normalization stats to {config.best_stats_path}")

        payload = {
            "state_dict": model.state_dict(),
            "config": config.as_dict(),
            "obs_dim": obs_dim,
            "action_dim": act_dim,
            "normalization": normalization,
        }
        torch.save(payload, config.policy_path)
        print(f"saved policy to {config.policy_path}")

        if config.stats_path:
            _write_stats(config.stats_path, normalization)
            print(f"saved normalization stats to {config.stats_path}")

        if writer is not None:
            writer.close()

        return metrics
    finally:
        train_dataset.close()
        if val_dataset is not None:
            val_dataset.close()
