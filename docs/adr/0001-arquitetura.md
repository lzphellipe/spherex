# ADR 0001 — Ports & adapters, saídas imutáveis e segurança primeiro

**Status:** aceito · 2026-10-29

## Contexto

Hackathon de 48 h, dados remotos (IRSA) lentos, instância GPU alugada que pode ser preemptada ou
deletada por engano, e um tema (Planeta X) que convida a conclusões exageradas.

## Decisão

1. **Ports & adapters.** O `domain` define Protocols; `infrastructure` implementa (IRSA, cache, céu
   sintético). O pipeline roda igual com qualquer fonte, o que permite desenvolver e testar offline.
2. **Saídas imutáveis endereçadas por conteúdo.** Cada estágio grava em `.../<nome>__<config_hash>/`
   e só escreve `_SUCCESS.json` no fim. Mudar a config nunca sobrescreve resultados anteriores.
3. **Ledger retomável** (SQLite WAL) por (estágio, campo, config_hash).
4. **Checkpoints atômicos com SHA-256**, preempção salva a época parcial (nunca vira "best"), e a
   identidade do experimento exclui `epochs` para que estender o treino retome em vez de recomeçar.
5. **Avaliação antes de vitrine.** A completude é medida por injeção e o baseline aparece sempre ao lado.

## Consequências

- Mais diretórios em `data/processed` (limpeza manual e explícita, nunca automática).
- A API é somente leitura: processamento pesado só pelo worker, via fila.
