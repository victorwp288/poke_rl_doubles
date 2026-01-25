from pathlib import Path

from src.config import section
from src.online.policy.warmstart import load_behavior_clone_weights


def _load_bc_if_requested(model, settings):
    checkpoint = settings.get("bc_checkpoint")
    if not checkpoint:
        return
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"BC checkpoint not found: {checkpoint}")

    offline_config = section("offline")
    stats_path = offline_config.get("stats_path")
    stats_path = Path(stats_path) if stats_path else None
    try:
        stats = load_behavior_clone_weights(
            policy=model.policy,
            checkpoint_path=checkpoint,
            stats_path=stats_path,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load BC weights from {checkpoint}: {exc}") from exc

    if stats is not None:
        print(
            f"loaded behavior clone weights (count={stats.count} mean_dim={stats.mean.numel()})",
            flush=True,
        )


__all__ = ["_load_bc_if_requested"]
