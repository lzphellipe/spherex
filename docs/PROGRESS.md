# Progresso

## Feito (2026-10-29)

- Fundação: domínio, ports, config com hash, CLI, API, UI de revisão, worker com fila.
- Segurança: checkpoints atômicos, ledger, cache somente leitura, backup contínuo, `safe-delete`, `restore`.
- Pipeline: casamento de banda LVF, reprojeção cúbica, casamento de PSF, fundo, detecção com deblending,
  remoção de estáticas, linking por KD-tree, shift-and-stack, injeção sintética.
- Modelo: CNN real/bogus + baseline, split por campo, retomada após SIGTERM.
- Avaliação: completude por magnitude e taxa (IC de Wilson), magnitude limite logística (IC bootstrap),
  precisão top-K, ECE, limite de ordem de grandeza para corpos distantes.
- 29 testes offline + 2 de rede; demo reprodutível em ~15 s.

## Pendente (por prioridade)

1. Rodar `make test-network` e corrigir o que divergir em `docs/dados.md`.
2. Escolher 2–3 campos reais perto da eclíptica com asteroides conhecidos; baixar e cachear.
3. Medir PSF e ponto zero reais por campo (hoje vêm da config).
4. Caso de validação real distante (ex.: Makemake), se houver cobertura.
5. Integrar shift-and-stack ao `detect_service` para o regime lento (hoje é função + teste).
6. Recalibrar o score (temperature scaling) com rótulos humanos.
