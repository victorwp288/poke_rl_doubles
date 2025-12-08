import json
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
    observations = torch.tensor([sample[0] for sample in batch], dtype=torch.float32)
    actions = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
    mask = torch.tensor([sample[2] for sample in batch], dtype=torch.bool)
    return observations, actions, mask


def _binary_feature_mask(binary_flags, device):
    if not binary_flags:
        return None
    numeric_flags = [not flag for flag in binary_flags]
    if not any(numeric_flags):
        return None
    return torch.tensor(numeric_flags, dtype=torch.float32, device=device)


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


def _slot_loss(logits, actions, mask, loss_fn):
    masked = logits.masked_fill(~mask, -1e9)
    losses = []
    for index, slot in enumerate(masked.unbind(dim=1)):
        losses.append(loss_fn(slot, actions[:, index]))
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
):
    model.train()
    total_loss = 0.0
    total_items = 0
    for step, (obs, actions, mask) in enumerate(loader, 1):
        obs = obs.to(device)
        actions = actions.to(device)
        mask = mask.to(device)
        if obs_noise_std:
            noise = torch.randn_like(obs) * obs_noise_std
            if numeric_feature_mask is not None:
                noise = noise * numeric_feature_mask
            inputs = obs + noise
        else:
            inputs = obs
        optimizer.zero_grad()
        loss = _slot_loss(model(inputs), actions, mask, loss_fn)
        loss.backward()
        if grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += loss.item() * obs.size(0)
        total_items += obs.size(0)
        if log_every and step % log_every == 0:
            print(f"step {step}: loss {loss.item():.4f}")
    return total_loss / max(1, total_items)


def _evaluate(model, loader, device, loss_fn):
    if loader is None:
        return None
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for obs, actions, mask in loader:
            obs = obs.to(device)
            actions = actions.to(device)
            mask = mask.to(device)
            loss = _slot_loss(model(obs), actions, mask, loss_fn)
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


def _data_loader(dataset, batch_size, device, shuffle):
    if dataset is None or len(dataset) == 0:
        return None
    pin_memory = getattr(device, "type", "cpu") == "cuda"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_batch_collate,
        pin_memory=pin_memory,
    )


def _write_stats(path, stats):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


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
    )
    train_offsets, val_offsets = split_train_val(scan.offsets, config.val_fraction)
    if not train_offsets:
        raise ValueError("training split must contain samples")

    train_dataset = IndexedJsonlDataset(scan.path, train_offsets, scan.mean, scan.std)
    val_dataset = (
        IndexedJsonlDataset(scan.path, val_offsets, scan.mean, scan.std) if val_offsets else None
    )

    sample_obs, _, sample_mask = train_dataset[0]
    obs_dim = scan.obs_dim or len(sample_obs)
    act_dim = scan.action_dim or (len(sample_mask[0]) if sample_mask else 0)

    numeric_feature_mask = _binary_feature_mask(scan.binary_flags, device)

    train_loader = _data_loader(train_dataset, config.batch_size, device, True)
    val_loader = _data_loader(val_dataset, config.batch_size, device, False)

    writer = _tensorboard_writer(config.tensorboard_dir)

    model = BehaviorCloningPolicy(
        obs_dim,
        act_dim,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    metrics = []
    best_loss = None
    normalization = {"mean": scan.mean, "std": scan.std, "count": scan.count}

    config.policy_path.parent.mkdir(parents=True, exist_ok=True)
    if config.stats_path:
        config.stats_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "offline setup: "
        f"total={len(scan.offsets)} train={len(train_offsets)} val={len(val_offsets)} "
        f"batch={config.batch_size} epochs={config.epochs} device={device}",
        flush=True,
    )

    try:
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
            )
            val_loss = _evaluate(model, val_loader, device, loss_fn)
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
