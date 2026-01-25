import json


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


def _write_stats(path, stats):
    def _to_jsonable(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(stats, default=_to_jsonable, ensure_ascii=False, indent=2), encoding="utf-8"
    )


__all__ = ["_tensorboard_writer", "_write_stats"]
