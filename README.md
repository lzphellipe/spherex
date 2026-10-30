# SPHEREx Moving Objects

Sistema de detecção de objetos em movimento nas imagens multiépoca do SPHEREx, com avaliação honesta.
Projeto para o NASA Space Apps Challenge 2026, desafio **"Planet X and SPHEREx"**.

> **Este sistema não detecta nem identifica o Planeta X.** Ele produz *candidatos para revisão humana*
> e mede, por injeção sintética, o que conseguiria ver. Planet X é a motivação, não o resultado.

## A pergunta que o projeto responde

*Quanto do que existe nós conseguimos recuperar, com qual taxa de falsos positivos, e o que um objeto
distante precisaria ter para ser visto?*

## Resultado da demo (céu sintético, `spherex demo`, ~15 s em CPU)

| Métrica | Valor |
|---|---|
| AUC da CNN real/bogus (validação em campos nunca vistos) | 0,973 |
| AUC do baseline (regressão logística sobre features) | 0,875 |
| Precisão nos 10 candidatos de maior score | 100% |
| Asteroides "conhecidos" recuperados | 40 de 48 |
| Magnitude a 50% de completude (taxa ≥ 4″/dia) | 18,53 [IC 95%: 18,02 – 18,94] |
| Calibração (ECE) | 0,072 |

A curva por taxa mostra o limite físico do cadenciamento: objetos com menos de ~2″/dia se
autossubtraem na diferença de imagens (14% de completude). É exatamente o regime de um corpo
distante, e é por isso que `pipeline/linking.py` inclui *shift-and-stack*.

Esses números são do céu sintético e servem para provar o pipeline, não o SPHEREx.
Com dados reais (`--source irsa`), as métricas são recalculadas do zero.

## Como rodar

```bash
make install          # pip install -e ".[dev]"
make test             # 32 testes offline
spherex demo          # detect → train → eval no céu sintético
spherex serve         # API + UI de revisão em http://127.0.0.1:8000
spherex worker        # executa jobs que a API enfileira
```

Dados reais (requer rede; confira antes os endpoints em `docs/dados.md`):

```bash
make test-network                       # valida IRSA/JPL
spherex detect --source irsa -c configs/meu_campo.yaml
```

## Arquitetura (ports & adapters)

```
src/spherex/
├── domain/          entidades + ports (Protocols). Não importa nada de fora.
├── pipeline/        preprocess (banda LVF, reprojeção, PSF), detection, linking,
│                    injection, stamps, models (CNN + baseline), evaluation
├── application/     detect / train / eval / review services
├── infrastructure/  IRSA, Horizons, sb_ident, AllWISE, céu sintético, cache,
│                    checkpoint, ledger
└── interface/       CLI (Typer), API (FastAPI), UI de revisão
```

O pipeline só conhece os `ports`. Trocar o IRSA real pelo cache ou pelo céu sintético é trocar o adapter.

### Decisões técnicas que importam

- **Casamento de banda (LVF):** o comprimento de onda varia ao longo do detector do SPHEREx.
  Comparar épocas de bandas diferentes cria fontes falsas. O teste `test_band_mismatch_creates_spurious_sources`
  mede isso. Épocas sem referência na mesma banda são puladas e registradas.
- **Spline cúbica na reprojeção:** a PSF é subamostrada. A interpolação bilinear achatava os picos e
  dobrava os resíduos de estrelas (medido).
- **Rótulos gratuitos:** positivos vêm de objetos conhecidos e de injeções. Negativos vêm de raios cósmicos,
  colunas ruins e resíduos de subtração.
- **Sem vazamento:** o split é agrupado por campo do céu, e o baseline sempre aparece ao lado da CNN.
- **Reprodutível:** a seed é fixa e há desempates determinísticos. Duas execuções com `PYTHONHASHSEED`
  diferentes geram relatórios idênticos.

## Segurança de dados (nada se perde)

| Risco | Proteção | Teste |
|---|---|---|
| Queda no meio de um save | escrita atômica + `fsync` | `test_crash_during_save_keeps_previous_checkpoint` |
| Checkpoint corrompido | SHA-256 no manifesto, volta ao anterior | `test_corrupted_latest_falls_back_to_previous` |
| Preempção da instância spot | SIGTERM salva a época parcial e a retomada refaz essa época | `test_sigterm_during_training_saves_and_resumes` |
| Processo morre no meio do lote | ledger SQLite retoma só o que falta | `test_ledger_resumes_and_isolates_failures` |
| Reprocessar com outra config | `config_hash` no caminho, nunca sobrescreve | idem |
| Dado bruto alterado por engano | cache em `data/raw` somente leitura | `test_raw_cache_is_read_only` |
| Deletar a instância Brev | `make safe-delete` exige backup recente + nome digitado | `test_guard_*` |

Backup contínuo para fora da instância: configure um remoto no `rclone` e `BACKUP_REMOTE` no `.env`,
depois rode `make guard`. Numa instância nova, `make restore` traz tudo de volta.

## NVIDIA Brev

O gargalo é o download do IRSA e as etapas em CPU, não a GPU. Escolha a instância por vCPUs, RAM e rede.
Uma GPU de entrada (T4/L4) sobra para a CNN. Use `scripts/launchable/setup.sh` para montar o ambiente.

## Créditos de dados

SPHEREx (NASA/JPL-Caltech/IPAC), via NASA/IPAC Infrared Science Archive (IRSA); JPL Horizons e
Small-Body Identification; AllWISE (NASA/IPAC). Inclua o agradecimento e o DOI exigidos pelo SPHEREx/IRSA
na submissão: confira o texto exato na página do IRSA.

## Licença

MIT.
