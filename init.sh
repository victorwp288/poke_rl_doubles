#!/usr/bin/env bash

set -euo pipefail

python_bin="${PYTHON_BIN:-python3}"

"${python_bin}" -m pip install --upgrade pip

echo "Installing core packages..."
core_packages=(
  "torch>=2.3"
  "stable-baselines3>=2.3.2"
  "sb3-contrib>=2.3.0"
  "poke_env==0.10.0"
  "asyncio>=4.0.0"
  "gymnasium>=0.29"
  "numpy>=2.3"
  "pandas==2.3.2"
  "matplotlib==3.10.5"
  "seaborn>=0.13"
  "tensorboard==2.19.0"
  "pyyaml>=6.0"
  "typer>=0.12"
  "pydantic-settings>=2.5"
  "orjson>=3.10"
  "rich>=13.8"
  "tqdm>=4.66"
  "ruff>=0.12"
  "pre-commit>=4.3.0"
)

for package in "${core_packages[@]}"; do
  if [[ "${package}" == "asyncio>=4.0.0" ]]; then
    echo "  -> skipping asyncio (stdlib module ships with Python 3.11)"
    continue
  fi
  echo "  -> ${package}"
  "${python_bin}" -m pip install "${package}"
done

echo "Installing optional web viewer packages..."
web_packages=(
  "gradio>=4.44"
)

for package in "${web_packages[@]}"; do
  echo "  -> ${package}"
  "${python_bin}" -m pip install "${package}"
done

echo "Installing optional database packages..."
db_packages=(
  "sqlmodel>=0.0.22"
  "pydantic>=2.8"
)

for package in "${db_packages[@]}"; do
  echo "  -> ${package}"
  "${python_bin}" -m pip install "${package}"
done

echo "Environment setup complete."

