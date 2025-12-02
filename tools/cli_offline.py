#!/usr/bin/env python3
""" 
CLI for offline training with configurable hyperparameters.
The script allows overriding fields in 'OfflineConfig' from the command line.
If no flags are provided, it uses the default values defined in 'src/offline/config.py'
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
from dataclasses import fields, MISSING

#Ensurement of the project Root in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.offline import OfflineConfig, train_offline

#Infer dataclass field types to argparse-compatible types.
def _infer_arg_type(arg_type):
    origin = getattr(arg_type, "__origin__", None)
    if arg_type == Path or origin == Path:
        return Path
    if arg_type in {int, float, str}:
        return arg_type
    return str #Fallback for complex or optional types        

#Dynamically add CLI flags for all fields in src/offline/config.py 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline training with configurable hyperparameters"
    )

    for f in fields(OfflineConfig):
        arg_name = f"--{f.name.replace('_', '-')}"
        arg_type = f.type
        default = f.default if f.default is not MISSING else None

        #Boolean flags: allow Shuffle/No Shuffle toogle
        if arg_type == bool:
            if default:
                parser.add_argument(
                    f"--no-{f.name.replace('_', '-')}",
                    dest=f.name,
                    action="store_false",
                    help=f"Disable {f.name.replace('_', '-')} (default: Enabled)",
                )
            else:
                parser.add_argument(
                    arg_name,
                    type=_infer_arg_type(arg_type),
                    default=None,
                    help=f"(default: {default})",
                )

    return parser.parse_args()

#Main entry point - parse CLI args, create config, and run offline training
def main() -> None:
    args = parse_args()

    #Only override fields that the user actually set
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    
    #Use defaults from OfflineConfig for everything else
    config = OfflineConfig(**overrides)

    print("Starting offline training with config:")
    print(config)
    train_offline(config)

if __name__ == "__main__":
    main()    

"""
To run the script write in terminal: 
    python tools/cli_offline.py
    python tools/cli_offline.py --epochs 100
    python tools/cli_offline.py --learning-rate 0.0005 --batch-size 4096
    python tools/cli_offline.py --no-shuffle
"""