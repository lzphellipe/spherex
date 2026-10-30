# Prompt — SPHEREx Moving Objects (NASA Space Apps 2026 · "Planet X and SPHEREx")

> Cole este prompt inteiro no agente de código (Claude Code ou similar), na raiz de um repositório vazio.
> Se os arquivos de referência `checkpoint.py`, `ledger.py`, `brev_guard.sh` e `Makefile` estiverem disponíveis, coloque-os na raiz antes de começar.

---

## 1. Papel

Você é um engenheiro de sistemas de ML sênior, com experiência em astronomia de survey (astropy, imagem diferença, detecção de objetos em movimento) e em engenharia de software de produção (arquitetura limpa, testes, MLOps). Trabalhe em Python 3.11+. Priorize código correto, testado e reproduzível em vez de funcionalidades vistosas.

## 2. Contexto

- Evento: NASA Space Apps Challenge 2026, hackathon de 48h em 14–15/nov/2026, desafio "Planet X and SPHEREx".
- O SPHEREx (NASA/JPL/IPAC) mapeia o céu inteiro a cada ~6 meses em 102 bandas no infravermelho próximo (0,75–5 µm). Objetos do Sistema Solar se revelam por mudar de posição entre épocas.
- Dados: IRSA (busca SIA na coleção `spherex_qr2`, serviço de cutouts, datalink), JPL Horizons (posição do próprio SPHEREx, ID −163182; efemérides), JPL Small-Body Identification (`sb_ident`, asteroides conhecidos no campo) e AllWISE via TAP (fontes estáticas).
- Time de 4–5 pessoas. Processamento em instância GPU na NVIDIA Brev, ou local quando não precisar de GPU.
- Antes de codar qualquer regra de negócio, confira o enunciado oficial em spaceappschallenge.org e as regras do evento sobre o que pode ser preparado antes de 14/11. Se algo neste prompt conflitar com o enunciado oficial, o enunciado vence: avise e ajuste.

## 3. Objetivo do produto

Construir um **sistema de detecção de objetos em movimento nas imagens multiépoca do SPHEREx, com avaliação honesta**, e uma interface pública para revisar candidatos.

A pergunta que o sistema responde: *"Quanto do que existe nós conseguimos recuperar, com qual taxa de falsos positivos, e o que um objeto distante tipo Planeta X precisaria ter para ser visto?"*

### Restrições científicas (inegociáveis)

1. **Nunca** afirmar que o sistema detectou o Planeta X. Planet X é motivação. O que o sistema produz são **candidatos para revisão humana**, nunca detecções.
2. Todo número exibido (completude, precisão, limites) tem que vir de medição reproduzível no código, nunca de estimativa.
3. Cada candidato carrega proveniência: exposições usadas, versão do modelo, configuração e limiares.
4. Deixar explícito na UI e na documentação o limite físico: o brilho refletido cai com ~1/r⁴, então um corpo muito distante provavelmente está abaixo do limite de detecção do SPHEREx.

## 4. Reformulação do problema (rótulos sem anotação manual)

- **Positivos:** asteroides conhecidos previstos pelo JPL `sb_ident`/Horizons dentro do campo e da época de cada exposição.
- **Negativos:** fontes estáticas do AllWISE, além de artefatos (raios cósmicos, bordas, resíduos de subtração).
- **Sintéticos:** injeção de fontes pontuais com PSF realista, magnitude e taxa de movimento controladas. Inclua o regime lento (objetos distantes) para medir completude.
- **Validação em objeto real distante:** recuperar um planeta anão conhecido (por exemplo, Makemake ou Haumea) quando houver cobertura.

## 5. Arquitetura (ports & adapters / arquitetura limpa)

Regra central: **`domain` não importa nada de fora**. `pipeline` e `application` dependem de `ports` (Python `Protocol`), e `infrastructure` implementa essas portas. O mesmo código precisa rodar com o IRSA real, com o cache local ou com uma fonte sintética, sem mudanças.

```
spherex-moving-objects/
├── src/spherex/
│   ├── interface/          # api/ (FastAPI), web/ (UI de revisão), cli/ (Typer), schemas/ (Pydantic)
│   ├── application/        # detect_service, review_service, train_service, eval_service
│   ├── pipeline/           # preprocess, detection, linking, models, injection
│   ├── domain/             # entidades + ports (Protocols)
│   └── infrastructure/     # irsa_client, horizons_client, sbident_client, allwise_client,
│                           # storage, registry, checkpoint, ledger
├── configs/                # YAML por experimento (campos do céu, limiares, hiperparâmetros)
├── data/                   # raw/ (somente leitura), cache/, processed/  — fora do git (DVC)
├── artifacts/              # checkpoints/, ledger/, reports/, guard.log
├── scripts/                # brev_guard.sh, launchable/ (setup do ambiente Brev)
├── notebooks/              # exploração; nunca importados por src/
├── tests/                  # unit/ (domain, pipeline) + integration/ (infra, com rede marcada)
├── docs/                   # model card, curva de completude, ADRs, guia de dados
├── Makefile
├── pyproject.toml          # uv ou poetry; versões fixadas
└── .env.example
```

### 5.1 Domain (`src/spherex/domain/`)

- `Exposure`: id, época (MJD), detector, faixa de comprimento de onda, WCS, caminho do FITS e máscara.
- `Detection`: posição (RA/Dec), fluxo, SNR e exposição de origem.
- `Tracklet`: lista de `Detection` ligadas entre épocas, taxa angular e ajuste de órbita/paralaxe.
- `Candidate`: tracklet + score + classe + status de revisão + `Provenance`.
- `Provenance`: ids de exposição, hash da config, versão do modelo, versão do código (commit) e timestamp.
- `ports.py`: `ExposureSource`, `EphemerisSource`, `KnownObjectSource`, `StaticCatalog`, `ArtifactStore`, `ModelRegistry`, `CheckpointStore`, `WorkLedger`.

### 5.2 Pipeline (`src/spherex/pipeline/`)

1. **preprocess:** os filtros do SPHEREx são lineares variáveis (LVF), então o comprimento de onda varia ao longo do detector. **Antes de qualquer subtração, case a banda** entre épocas e reprojete para a mesma WCS (`reproject`). Comparar quadros brutos gera movimento falso por diferença de cor. Aplique máscara de pixels ruins e normalize o fundo.
2. **detection:** imagem diferença (casamento de PSF), extração de fontes (`photutils` ou `sep`) e cross-match com o AllWISE para descartar o que é estático.
3. **linking:** ligue detecções em tracklets com KD-tree, restringindo pela velocidade angular máxima (nada de força bruta). Para objetos lentos, use *shift-and-stack* ao longo de trajetórias hipotéticas, com a paralaxe derivada da posição do SPHEREx no Horizons. Trate fronteiras de tile com borda de sobreposição.
4. **models:**
   - CNN pequena *real/bogus* sobre trios de recortes (referência, nova, diferença) de ~32×32 px.
   - Opcional: classificador espectral sobre as 102 bandas (asteroide, anã marrom, objeto rico em gelo).
5. **injection:** gerador de fontes sintéticas (magnitude, taxa angular, direção) para treino e para a curva de completude.

### 5.3 Application (`src/spherex/application/`)

- `detect_service`: orquestra o pipeline por tile do céu via `WorkLedger` (retomável).
- `review_service`: fila de revisão ordenada por aprendizado ativo (incerteza e diversidade) e consenso entre revisores.
- `train_service`: treino com checkpoints seguros, retomada automática e registro de métricas.
- `eval_service`: curva de completude, recall de asteroides conhecidos, precisão no top-K e calibração.

### 5.4 Interface (`src/spherex/interface/`)

- **API (FastAPI), stateless e somente leitura sobre resultados pré-calculados.** Nenhuma requisição dispara processamento pesado: a API enfileira o job e um worker grava o resultado.
- **Web:** mapa do céu (Aladin Lite), comparador blink/diferença, painel de candidato com proveniência e espectro, botões de rótulo (objeto real, artefato, estrela) e uma página de métricas com a curva de completude.
- **CLI:** `spherex ingest | detect | train | eval | report`.

## 6. Avaliação (o diferencial do projeto)

- **Curva de completude:** fração de fontes injetadas recuperadas por magnitude e por taxa angular, com intervalo de confiança. Gerar `artifacts/reports/completeness.png` e `.json`.
- **Recall de asteroides conhecidos** por faixa de magnitude.
- **Precisão no top-K** de candidatos, a partir dos rótulos de revisão.
- **Calibração** do score do modelo (curva de confiabilidade).
- **Model card** em `docs/model_card.md`: dados, métricas, limitações e o aviso sobre o Planet X.

## 7. Segurança de dados e de trabalho (obrigatório)

Nenhum dado baixado, processamento ou treino pode ser perdido por queda, preempção, erro humano ou exclusão da instância.

### 7.1 Checkpoints de treino — `src/spherex/infrastructure/checkpoint.py`

- **Escrita atômica:** grava em arquivo temporário no mesmo diretório, faz `fsync` do arquivo e do diretório e só então `os.replace`.
- **Manifesto** `manifest.json` (também gravado de forma atômica) com SHA-256, step, métricas e data de cada checkpoint.
- `load_latest()`: verifica o hash e, se o mais recente estiver ausente ou corrompido, cai para o anterior.
- **Retenção:** mantém os N mais recentes + `best.pt` (pela métrica configurada), que nunca é apagado pela limpeza.
- `install_preemption_handler()`: em SIGTERM/SIGINT, salva antes de sair (instância spot, shutdown, Ctrl+C).
- **Backup remoto** de cada checkpoint via `rclone copy`, nunca `sync`.

### 7.2 Registro de progresso — `src/spherex/infrastructure/ledger.py`

- SQLite em modo WAL; chave `(stage, tile, config_hash)`; status `running | done | failed`, com caminho do resultado e erro.
- `run()` pula unidades já concluídas com o mesmo `config_hash` (e cujo arquivo existe), refaz só o que falta e isola falhas: um tile com erro não derruba o lote.
- **Mudou a configuração, mudou o hash:** o tile é reprocessado. Os resultados incluem o `config_hash` no nome do arquivo, para nunca sobrescrever saídas anteriores.
- Toda saída é gravada com escrita atômica antes de marcar `done`.

### 7.3 Backup contínuo — `scripts/brev_guard.sh`

- Modos: `loop` (a cada `INTERVAL`, padrão 15 min), `once` e `check` (falha se o último backup for mais antigo que `MAX_AGE`).
- Protege `data/raw`, `data/cache`, `data/processed`, `artifacts/checkpoints`, `artifacts/ledger` e `configs`.
- Usa `rclone copy` com `--backup-dir $REMOTE/_versions/<data>/`, para que versões sobrescritas sejam preservadas.
- `trap` em TERM/INT faz um backup final antes de encerrar.
- Torna `data/raw` somente leitura após o download.
- O remoto é configurado em `.env` (`BACKUP_REMOTE`), e o bucket deve ter versionamento ativado.

### 7.4 Comandos seguros — `Makefile`

- `make guard`: backup contínuo em background, com log em `artifacts/guard.log`.
- `make backup` e `make check-backup`.
- `make safe-stop`: backup antes de `brev stop`.
- `make safe-delete`: backup, depois verificação de que o backup é recente, depois exige digitar o nome da instância e só então roda `brev delete`.
- `make restore`: numa instância nova, traz tudo de volta do remoto.

### 7.5 Regras gerais de segurança

- Código sempre no git, com commits pequenos e push frequente. Dados grandes via DVC apontando para o mesmo remoto.
- Nunca usar `rm -rf` em `data/` ou `artifacts/` dentro de scripts. Limpeza só por comando explícito e documentado.
- Segredos só em `.env` (no `.gitignore`), com `.env.example` versionado.
- Pesos de modelo publicados no `registry` (MLflow local ou arquivos versionados) com o hash do dataset de treino.

### 7.6 Testes obrigatórios da camada de segurança

- Queda simulada no meio de um save deixa o checkpoint anterior intacto e nenhum `.tmp` órfão.
- Checkpoint corrompido faz `load_latest()` voltar ao anterior.
- A retenção preserva o `best.pt`.
- O ledger retoma: a segunda execução pula os tiles concluídos e refaz só os que falharam. Uma config nova reprocessa sem sobrescrever.
- `bash -n scripts/brev_guard.sh` passa, e o modo `check` falha sem backup registrado.

## 8. Infraestrutura e ambiente

- **NVIDIA Brev:** escolha a instância pelas vCPUs, pela RAM e pela rede, não pela GPU mais forte. Uma GPU de entrada (classe T4/L4) basta para a CNN. Crie um *Launchable* em `scripts/launchable/` que instala o ambiente com versões fixadas e um único comando.
- O gargalo esperado é o download do IRSA e as etapas de CPU (reprojeção, extração de fontes), não a GPU. Baixe e cacheie os campos escolhidos nas primeiras horas.
- **Escalabilidade (documentar, não construir em 48h):** particionamento do céu em tiles HEALPix, jobs paralelos (Ray/Dask ou Kubernetes), dados em object storage particionados por tile e época. Mostrar num slide que a troca é só de adapters.

## 9. Plano de entrega (48h)

| Janela | Entrega |
|---|---|
| H0–H2 | Repo, ambiente Brev, `make guard` rodando, remoto de backup testado com `make backup` e `make restore` |
| H2–H8 | Ingestão e cache de 2–3 campos com asteroides conhecidos; baseline de diferença + cross-match AllWISE |
| H8–H24 | Correção de banda LVF, injeção sintética, linking e **primeira curva de completude** |
| H24–H36 | CNN real/bogus com rótulos gratuitos + sintéticos, com checkpoints seguros; UI de revisão |
| H36–H44 | Validação em planeta anão conhecido, model card, página de métricas |
| H44–H48 | Vídeo, slides (incluindo escalabilidade), README e submissão. **Congelar código às H46** |

## 10. Critérios de aceite (definição de pronto)

- [ ] `make restore` numa instância nova reconstrói o estado sem perda.
- [ ] Interromper o treino com SIGTERM e rodar de novo retoma do último checkpoint íntegro.
- [ ] Interromper o `detect` e rodar de novo processa só os tiles pendentes.
- [ ] A curva de completude é gerada por um único comando e é reproduzível com a mesma seed.
- [ ] Pelo menos um asteroide conhecido é recuperado e exibido com proveniência completa.
- [ ] Nenhum texto do produto afirma detecção do Planeta X.
- [ ] Testes de `domain`, `pipeline` e da camada de segurança passando; lint (`ruff`) e tipos (`mypy`) limpos em `src/`.
- [ ] README com: problema, arquitetura, como rodar, métricas obtidas, limitações e créditos de dados (incluindo o agradecimento e o DOI exigidos pelo SPHEREx/IRSA).

## 11. Como trabalhar

1. Comece pela fundação: estrutura do repo, `pyproject.toml`, `domain/` com `ports`, camada de segurança (seção 7) com seus testes, `Makefile` e `.env.example`. **Nada de pipeline antes de o backup e o restore funcionarem.**
2. Depois implemente os adapters de infraestrutura com testes de integração marcados (`@pytest.mark.network`), e uma `ExposureSource` sintética para testar sem rede.
3. Em seguida, o pipeline etapa por etapa, cada uma passando pelo `WorkLedger`.
4. Faça commits pequenos com mensagens claras. Ao fim de cada etapa, rode os testes e registre em `docs/PROGRESS.md` o que foi feito, o que falta e as decisões tomadas (ADRs curtos em `docs/adr/`).
5. Quando houver ambiguidade científica, escolha a opção mais conservadora, documente e siga em frente.
6. Não invente endpoints, parâmetros de API ou nomes de coleção: confirme na documentação do IRSA/JPL e registre a fonte em `docs/dados.md`.
