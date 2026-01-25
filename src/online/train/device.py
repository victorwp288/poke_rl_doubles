import torch


def _resolve_device(preferred=None):
    candidates: list[str] = []
    if isinstance(preferred, str) and preferred.strip():
        candidates.append(preferred.strip().lower())
    candidates.extend(["cuda", "mps", "cpu"])

    for name in candidates:
        if name.startswith("cuda"):
            if not torch.cuda.is_available():
                continue
            device = torch.device("cuda")
        elif name.startswith("mps"):
            if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
                continue
            if hasattr(torch.backends.mps, "is_built") and not torch.backends.mps.is_built():
                continue
            device = torch.device("mps")
        elif name.startswith("cpu"):
            device = torch.device("cpu")
        else:
            try:
                device = torch.device(name)
            except Exception:
                continue
        print(f"Using {device.type} device", flush=True)
        return device

    fallback = torch.device("cpu")
    print(f"Using {fallback.type} device", flush=True)
    return fallback


__all__ = ["_resolve_device"]
