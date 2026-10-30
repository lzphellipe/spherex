# Comandos do projeto. Os alvos de segurança protegem dados e treino na instância Brev.
INSTANCE ?= spherex-dev
PY ?= python3

.PHONY: install test lint demo detect train eval serve worker guard backup check-backup safe-stop safe-delete restore

install:        ## dependências (editável) + dev
	$(PY) -m pip install -e ".[dev]"

test:           ## testes offline (rede excluída por padrão)
	$(PY) -m pytest -q

test-network:   ## testes contra IRSA/JPL reais
	$(PY) -m pytest -q -m network

lint:
	ruff check src tests

demo:           ## pipeline inteiro no céu sintético
	spherex demo

detect train eval:
	spherex $@ $(ARGS)

serve:          ## API + UI de revisão em http://127.0.0.1:8000
	spherex serve

worker:         ## executa jobs enfileirados pela API
	spherex worker

# ---------- segurança de dados ----------
guard:          ## backup contínuo em background
	mkdir -p artifacts && nohup ./scripts/brev_guard.sh loop > artifacts/guard.log 2>&1 &

backup:         ## backup imediato
	./scripts/brev_guard.sh once

check-backup:   ## falha se o último backup estiver velho
	./scripts/brev_guard.sh check

safe-stop: backup   ## para a instância só depois de um backup completo
	brev stop $(INSTANCE)

safe-delete: backup check-backup   ## deletar apaga TUDO na instância: exige backup recente e confirmação
	@read -p "Digite o nome da instância para confirmar a exclusão: " ans; \
	  [ "$$ans" = "$(INSTANCE)" ] || { echo "Cancelado."; exit 1; }
	brev delete $(INSTANCE)

restore:        ## nova instância: traz tudo de volta do remoto
	@: $${BACKUP_REMOTE:?defina BACKUP_REMOTE}
	for p in data/raw data/cache data/processed artifacts/checkpoints artifacts/ledger configs; do \
	  rclone copy "$$BACKUP_REMOTE/$$p" "$$p" --transfers 8; done
