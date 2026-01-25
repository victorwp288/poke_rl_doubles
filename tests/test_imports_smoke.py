import importlib.util
import sys
from pathlib import Path


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)
    return module


def test_imports_smoke():
    root = Path(__file__).resolve().parents[1]
    sys.path.append(str(root))

    tools = [
        root / "tools" / "online.py",
        root / "tools" / "eval_models.py",
        root / "tools" / "offline_train.py",
        root / "tools" / "collect_dataset.py",
    ]
    for idx, tool in enumerate(tools):
        _load_module(tool, f"tool_{idx}")

    import src.core.action_mask  # noqa: F401
    import src.core.observation  # noqa: F401
    import src.offline.dataset  # noqa: F401
    import src.online.env  # noqa: F401
    import src.online.policy.head  # noqa: F401
    import src.online.runner  # noqa: F401
