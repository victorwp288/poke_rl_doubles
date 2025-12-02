#!/usr/bin/env python3
"""
CLI for offline training sweeps over learning rates and batch sites.
This script extends the single run 'cli_offline.py' by running a grid of trials.
If no flags are provided, it defaults to the same four trialds defined in 
'offline_sweep.py' (Combination of [5e-4, 1e-3] x [1024, 2048])
"""

from __future__ import annotations
import sys
import argparse
from dataclasses import fields, MISSING
from pathlib import Path

#Ensurement of the project Root in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.offline import OfflineConfig, train_offline

#Infer an argpaser-compatible type for dataclass fields.
def _infer_arg_type(arg_type):
    origin = getattr(arg_type, "__origin__", None)
    if arg_type == Path or origin == Path:
        return Path
    if arg_type in {int, float, str}:
        return arg_type
    return str

#Parse CLI arguments for both OfflineConfig fields and sweep parameters.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a hyperparameter sweep for offline training."
    )

    #Add OfflineConfig fields dynamically
    for f in fields(OfflineConfig):
        arg_name = f"--{f.name.replace('_', '-')}"
        arg_type = f.type
        default = f.default if f.default is not MISSING else None
        
        if arg_type == bool:
            if default:
                parser.add_argument(
                    f"--no{f.name.replace('_', '-')}",
                    dest=f.name,
                    action="store_false", 
                    help=f"Disable {f.name.replace('_', '-')} (default enabled)", 
                )
            else:
                parser.add_argument(
                    arg_name,
                    type= _infer_arg_type(arg_type),
                    default=None,
                    help=f"(default: {default})",
                )

#Add sweep grid arguments            
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[5e-4, 1e-3],
        help="Learning rates to sweep over (default: 5e-4, 1e-3)",
    )

    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1024, 2048],
        help="Batch sizes to sweep over (default: 1024, 2048)",
    )

    return parser.parse_args()

#Entry point: parse CLI args, build sweep grid, and run each trial.
def main() -> None:
    args = parse_args()

    #Extract overrides for OfflineConfig
    overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k not in {"learning_rates", "batch_sizes"}
    }    

    base_config = OfflineConfig(**overrides)
    learning_rates = args.learning_rates
    batch_sizes = args.batch_sizes

    print("Starting offline sweep with:")
    print(f"  learning_rates = {learning_rates}")
    print(f"  batch_sizes = {batch_sizes}")
    print(f"  base_config = {base_config}")

    #Run each trial in the sweep grid
    for lr in learning_rates:
        for bs in batch_sizes:
            config = OfflineConfig(
                **{**vars(base_config), "learning_rate": lr, "batch size": bs}
            )

            print(f"\n[trial ] lr=[lr], batch_size={bs}")
            train_offline(config)

if __name__ == "__main__":
    main()    

"""
To run the script in terminal write: 

1. Default sweep (uses defaults from OfflineConfig):
    python tools/cli_offline-sweep.py

2. Sweep over custom values:
    python tools/cli_offline-sweep.py --learning-rates 1e-3 2e-3 --batch-sizes 512 1024

3. Override config fields:
    python tools/cli_offline-sweep.py --epochs 100 --device cuda --learning-rates 1e-4 1e-3
"""