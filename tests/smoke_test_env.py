#!/usr/bin/env python3

import contextlib
import importlib
from pathlib import Path

REQUIRED_IMPORTS = (
    "torch",
    "numpy",
    "poke_env",
)


def try_import(name):
    try:
        module = importlib.import_module(name)
        print(f"[OK]   {name}")
        return module
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        return None


def check_torch_devices(torch_module):
    if torch_module is None:
        return
    cuda_ok = torch_module.cuda.is_available()
    mps = getattr(torch_module.backends, "mps", None)
    mps_ok = bool(mps and mps.is_available())
    device = "cuda" if cuda_ok else "mps" if mps_ok else "cpu"
    print(f"torch device preference: {device} (cuda={cuda_ok}, mps={mps_ok})")
    try:
        tensor = torch_module.randn(64, 64, device=device)
        result = tensor @ tensor
        _ = result.sum().item()
        print("[OK]   torch matmul on", device)
    except Exception as exc:
        print(f"[WARN] torch compute failed on {device}: {exc}")


def connect_showdown():
    print("\n=== showdown connectivity ===")
    try:
        from poke_env import LocalhostServerConfiguration, ShowdownServerConfiguration
        from poke_env.player import RandomPlayer
    except Exception as exc:
        print(f"[WARN] poke-env random player not available: {exc}")
        return

    def _try_connection(kind, configuration):
        try:
            player = RandomPlayer(
                battle_format="gen9doublesou",
                server_configuration=configuration,
                start_listening=False,
            )
            if kind == "localhost":
                print("[INFO] created localhost player (no connection attempted)")
            with contextlib.suppress(Exception):
                player.reset_battles()
            close_fn = getattr(player, "close", None)
            if callable(close_fn):
                with contextlib.suppress(Exception):
                    close_fn()
            print(f"[OK]   {kind} configuration available")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {kind} connection failed: {exc}")

    _try_connection("localhost", LocalhostServerConfiguration)
    _try_connection("showdown", ShowdownServerConfiguration)


def write_plots():
    out_dir = Path("outputs/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        sns.set_theme(style="darkgrid")
        xs = np.linspace(0, 2 * np.pi, 200)
        plt.figure()
        plt.plot(xs, np.sin(xs))
        plt.title("matplotlib smoke")
        plt.savefig(out_dir / "matplotlib_smoke.png", dpi=120)
        plt.close()

        plt.figure()
        sns.histplot(np.random.randn(500), bins=25)
        plt.title("seaborn smoke")
        plt.savefig(out_dir / "seaborn_smoke.png", dpi=120)
        plt.close()
        print(f"[OK]   wrote plots to {out_dir}")
    except Exception as exc:
        print(f"[WARN] plotting failed: {exc}")


def main():
    print("=== required imports ===")
    modules = {name: try_import(name) for name in REQUIRED_IMPORTS}
    if modules.get("torch") is not None:
        check_torch_devices(modules["torch"])

    connect_showdown()

    print("\n=== plotting ===")
    write_plots()


if __name__ == "__main__":
    main()
