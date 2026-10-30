# Model card — classificador real/bogus

**O que faz:** dá a probabilidade de uma detecção na imagem diferença ser uma fonte pontual real
(e não raio cósmico, coluna ruim ou resíduo de subtração). O score de um candidato é a média das
suas detecções.

**O que não faz:** não identifica o tipo do objeto e não confirma descoberta. **Não detecta o Planeta X.**

## Dados

- Entrada: trio de recortes 21×21 (referência, science, diferença) normalizado pelo ruído da diferença.
- Positivos: detecções que coincidem (≤ 8″) com objetos conhecidos ou injetados.
- Negativos: o resto, que sobrevive à remoção de estáticas (raios cósmicos, resíduos, artefatos).
- Split **agrupado por campo do céu**: nenhum campo aparece no treino e na validação ao mesmo tempo.

## Métricas (céu sintético, seed 42)

| Modelo | AUC validação |
|---|---|
| Regressão logística (5 features: SNR, nitidez, anel, simetria, referência) | 0,875 |
| CNN (3 convoluções, ~16 mil parâmetros) | 0,973 |

Calibração (ECE, 10 faixas): 0,072. Para uso real, recalibre (temperature scaling) com rótulos humanos.

## Limitações

- Treinado só em dados sintéticos até agora: **retreine com dados reais** antes de qualquer conclusão.
- O céu sintético tem PSF gaussiana; a PSF real do SPHEREx não é.
- Positivos "conhecidos" herdam erros do catálogo do JPL (posições previstas, não observadas).

## Ciclo de melhoria

Os rótulos da UI de revisão (`ReviewStore.labeled()`) entram como novos exemplos. A fila prioriza os
candidatos em que o modelo está mais incerto (aprendizado ativo).
