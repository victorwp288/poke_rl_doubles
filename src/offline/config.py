from pathlib import Path


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


__all__ = ["OfflineConfig"]
