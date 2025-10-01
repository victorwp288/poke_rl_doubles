# Training loop for the doubles behavior cloning policy

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import OfflineConfig
from .dataset import ImitationSample, load_samples, split_train_val
from .model import BehaviorCloningPolicy


@dataclass(slots=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float | None


class _SampleDataset(Dataset[ImitationSample]):
    def __init__(self, samples: list[ImitationSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> ImitationSample:
        return self._samples[index]


def _batch_collate(batch: list[ImitationSample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observations = torch.tensor([sample.observation for sample in batch], dtype=torch.float32)
    actions = torch.tensor([sample.actions for sample in batch], dtype=torch.long)
    mask = torch.tensor([sample.mask for sample in batch], dtype=torch.bool)
    return observations, actions, mask


def _pick_device(preference: str | None) -> torch.device:
    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _step_losses(
    logits: torch.Tensor,
    actions: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: nn.Module,
) -> torch.Tensor:
    losses = []
    for slot in range(logits.shape[1]):
        slot_logits = logits[:, slot, :]
        slot_targets = actions[:, slot]
        slot_mask = mask[:, slot, :]
        slot_logits = slot_logits.masked_fill(~slot_mask, -1e9)
        losses.append(loss_fn(slot_logits, slot_targets))
    return sum(losses) / len(losses)


def _run_epoch(
    *,
    model: BehaviorCloningPolicy,
    loader: DataLoader[ImitationSample],
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    log_every: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for step, (obs, actions, mask) in enumerate(loader, 1):
        obs = obs.to(device)
        actions = actions.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        logits = model(obs)["logits"]
        loss = _step_losses(logits, actions, mask, loss_fn)
        loss.backward()
        optimizer.step()

        batch_size = obs.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

        if log_every and step % log_every == 0:
            print(f"train step {step}: loss {loss.item():.4f}")

    return total_loss / total_items if total_items else 0.0


def _evaluate(
    *,
    model: BehaviorCloningPolicy,
    loader: DataLoader[ImitationSample] | None,
    device: torch.device,
    loss_fn: nn.Module,
) -> float | None:
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
            logits = model(obs)["logits"]
            loss = _step_losses(logits, actions, mask, loss_fn)
            batch_size = obs.size(0)
            total_loss += loss.item() * batch_size
            total_items += batch_size
    return total_loss / total_items if total_items else None


def train_offline(config: OfflineConfig) -> list[EpochMetrics]:
    device = _pick_device(config.device)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    samples = load_samples(
        config.dataset_path,
        max_samples=config.max_samples,
        seed=config.seed,
        shuffle=config.shuffle,
    )
    train_samples, val_samples = split_train_val(samples, val_fraction=config.val_fraction)

    print(
        "offline setup: "
        f"total={len(samples)} train={len(train_samples)} val={len(val_samples)} "
        f"batch={config.batch_size} epochs={config.epochs} device={device}",
        flush=True,
    )

    train_loader = DataLoader(
        _SampleDataset(train_samples),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_batch_collate,
    )

    val_loader = (
        DataLoader(
            _SampleDataset(val_samples), batch_size=config.batch_size, collate_fn=_batch_collate
        )
        if val_samples
        else None
    )

    writer = None
    if config.tensorboard_dir is not None:
        config.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError(
                "tensorboard package missing; install with `pip install tensorboard`"
            ) from exc
        writer = SummaryWriter(log_dir=str(config.tensorboard_dir))
        print(f"tensorboard logging -> {config.tensorboard_dir}")

    obs_dim = len(train_samples[0].observation)
    act_dim = len(train_samples[0].mask[0])
    model = BehaviorCloningPolicy(obs_dim=obs_dim, action_dim=act_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()

    metrics: list[EpochMetrics] = []

    for epoch in range(1, config.epochs + 1):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            loss_fn=loss_fn,
            optimizer=optimizer,
            log_every=config.log_every,
        )
        val_loss = _evaluate(
            model=model,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
        )
        metrics.append(EpochMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss))

        if val_loss is None:
            print(f"epoch {epoch}: train {train_loss:.4f}")
        else:
            print(f"epoch {epoch}: train {train_loss:.4f} val {val_loss:.4f}")

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            if val_loss is not None:
                writer.add_scalar("loss/val", val_loss, epoch)
            writer.flush()

    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "obs_dim": obs_dim,
        "action_dim": act_dim,
    }
    Path(config.policy_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, config.policy_path)
    print(f"saved policy to {config.policy_path}")

    if writer is not None:
        writer.close()

    return metrics


__all__ = ["train_offline"]
