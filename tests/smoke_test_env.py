#!/usr/bin/env python3
"""
Smoke test for RL project dependencies.

Usage:
  python smoke_test_env.py

What it does:
- Prints Python version, platform, and pip-installed lib versions (where available)
- Verifies PyTorch imports and runs a tiny matmul
- Imports (and lightly exercises) core libs: numpy, gymnasium, poke_env
- Writes two small plots to ./artifacts: matplotlib_smoke.png and seaborn_smoke.png
- Tries optional extras if installed: gradio, sqlmodel
"""

import importlib
import platform
import sys
import time
from pathlib import Path


def imp(name: str):
    try:
        m = importlib.import_module(name)
        ver = getattr(m, "__version__", getattr(m, "version", "unknown"))
        print(f"[OK] import {name:<20} ver={ver}")
        return m
    except Exception as e:
        print(f"[FAIL] import {name:<20} err={e}")
        return None


def main():
    print("=== Python & Platform ===")
    print(f"Python: {sys.version.split()[0]}  impl={platform.python_implementation()}")
    print(f"Platform: {platform.platform()}")
    artifacts = Path("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)

    print("\n=== PyTorch ===")
    torch = imp("torch")
    if torch:
        try:
            cuda_ok = torch.cuda.is_available()
            mps = getattr(torch.backends, "mps", None)
            mps_ok = bool(mps and mps.is_available())
            device = torch.device("mps" if mps_ok else ("cuda" if cuda_ok else "cpu"))
            print(f"Device: {device}")
            a = torch.randn(1024, 1024, device=device)
            b = torch.randn(1024, 1024, device=device)
            t0 = time.time()
            c = a @ b
            _ = c.cpu().numpy()
            dt = time.time() - t0
            print(f"Matmul on {device} OK in {dt:.3f}s (MPS={mps_ok}, CUDA={cuda_ok})")
        except Exception as e:
            print("  PyTorch compute failed:", e)

    print("\n=== Core Numerics ===")
    np = imp("numpy")

    print("\n=== Gymnasium (smoke) ===")
    gym = imp("gymnasium")
    if gym:
        try:
            env = gym.make("CartPole-v1")
            obs, info = env.reset(seed=0)
            for _ in range(5):
                obs, r, terminated, truncated, info = env.step(env.action_space.sample())
                if terminated or truncated:
                    obs, info = env.reset()
            env.close()
            print("Gymnasium CartPole-v1 step OK")
        except Exception as e:
            print("  Gymnasium simple env failed:", e)

    print("\n=== poke-env (import only) ===")
    imp("poke_env")

    print("\n=== Logging, CLI, JSON, Boards ===")
    imp("rich")
    imp("tqdm")
    imp("typer")
    imp("orjson")
    imp("pydantic_settings")
    imp("tensorboard")

    print("\n=== Plotting (writes to ./artifacts) ===")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot([0, 1, 2], [0, 1, 0])
        plt.title("matplotlib smoke")
        out = artifacts / "matplotlib_smoke.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"matplotlib wrote {out}")
    except Exception as e:
        print("  matplotlib failed:", e)

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        sns.set_theme()
        x = np.random.randn(1000)
        plt.figure()
        sns.histplot(x, bins=30)
        plt.title("seaborn smoke")
        out = artifacts / "seaborn_smoke.png"
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"seaborn wrote {out}")
    except Exception as e:
        print("  seaborn failed:", e)

    print("\n=== Optional extras (import if present) ===")
    imp("gradio")
    imp("sqlmodel")

    print("\nAll checks attempted. If you saw only [OK] lines, you're good.")


if __name__ == "__main__":
    main()
