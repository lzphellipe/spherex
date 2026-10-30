#!/usr/bin/env bash
# Backup contínuo da instância Brev para um remoto fora dela.
# Local no repo: scripts/brev_guard.sh
#
# Uso:
#   ./scripts/brev_guard.sh loop      # roda em background, copia a cada INTERVAL
#   ./scripts/brev_guard.sh once      # um backup agora (usado pelo make safe-delete)
#   ./scripts/brev_guard.sh check     # falha se o último backup for antigo
#
# Configuração (variáveis de ambiente ou .env):
#   BACKUP_REMOTE   remoto rclone, ex.: backup:spherex   (S3, R2, GCS, Drive...)
#   INTERVAL        segundos entre backups (padrão 900 = 15 min)
#   MAX_AGE         idade máxima aceitável do último backup em segundos (padrão 1200)
set -euo pipefail

cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a

: "${BACKUP_REMOTE:?defina BACKUP_REMOTE, ex.: backup:spherex}"
INTERVAL="${INTERVAL:-900}"
MAX_AGE="${MAX_AGE:-1200}"
STAMP=".last_backup"
TODAY="$(date +%Y%m%d)"

# O que proteger (dados brutos, cache, resultados, checkpoints, ledger, configs)
PATHS=(data/raw data/cache data/processed artifacts/checkpoints artifacts/ledger configs)

backup_once() {
  for p in "${PATHS[@]}"; do
    [ -e "$p" ] || continue
    # copy, não sync: nada apagado localmente some do remoto.
    # --backup-dir: se um arquivo remoto for sobrescrito, a versão antiga é preservada.
    rclone copy "$p" "$BACKUP_REMOTE/$p" \
      --backup-dir "$BACKUP_REMOTE/_versions/$TODAY/$p" \
      --transfers 8 --checkers 16 --fast-list --quiet
  done
  date +%s > "$STAMP"
  echo "[guard] backup ok $(date '+%H:%M:%S') -> $BACKUP_REMOTE"
}

protect_raw() {
  # Dados brutos baixados do IRSA viram somente leitura: nenhum script os sobrescreve.
  [ -d data/raw ] && find data/raw -type f -exec chmod a-w {} + || true
}

check_fresh() {
  [ -f "$STAMP" ] || { echo "[guard] nenhum backup registrado"; exit 1; }
  age=$(( $(date +%s) - $(cat "$STAMP") ))
  if [ "$age" -gt "$MAX_AGE" ]; then
    echo "[guard] último backup há ${age}s (limite ${MAX_AGE}s)"; exit 1
  fi
  echo "[guard] último backup há ${age}s"
}

case "${1:-loop}" in
  once)  protect_raw; backup_once ;;
  check) check_fresh ;;
  loop)
    # Backup final ao receber stop/shutdown da instância ou Ctrl+C.
    trap 'echo "[guard] encerrando, backup final"; backup_once; exit 0' TERM INT
    while true; do
      protect_raw
      backup_once || echo "[guard] backup falhou, tentando no próximo ciclo" >&2
      sleep "$INTERVAL" & wait $!
    done ;;
  *) echo "uso: $0 {loop|once|check}"; exit 2 ;;
esac
