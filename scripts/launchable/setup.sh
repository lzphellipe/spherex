#!/usr/bin/env bash
# Ambiente do projeto numa instância NVIDIA Brev (ou qualquer Ubuntu com Python 3.11+).
# Uso: bash scripts/launchable/setup.sh   (idempotente)
set -euo pipefail
cd "$(dirname "$0")/../.."

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

command -v rclone >/dev/null || curl -fsSL https://rclone.org/install.sh | sudo bash

[ -f .env ] || cp .env.example .env
mkdir -p data/raw data/cache data/processed artifacts/checkpoints artifacts/ledger artifacts/reports

python -m pytest -q
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available())"
echo "Pronto. Configure o rclone (rclone config), defina BACKUP_REMOTE no .env e rode: make guard"
