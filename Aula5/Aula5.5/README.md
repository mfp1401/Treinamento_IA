# Aula 5.5 — Seleção de features

Mais colunas nem sempre é melhor: colunas sem informação deixam o modelo mais lento e podem atrapalhar. Para testar os métodos de seleção, o California Housing recebe **8 colunas de ruído aleatório**, e o objetivo é ver se cada método consegue descartá-las mantendo o desempenho.

## Conteúdos

- Baseline com todas as colunas (`HistGradientBoostingRegressor`, R² na validação cruzada)
- Filtros: `SelectKBest` com `f_regression` (ANOVA F) e `mutual_info_regression`
- Wrapper: `RFECV` com `Ridge`
- Escolher o menor número de colunas dentro da faixa aceitável de R²
- Teste "trancado": usado só na avaliação final

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado.py` | compara filtros e RFECV, com gráficos de R² por número de colunas |
| `exercicio1.py` | o fluxo completo em 21 passos: ruído, seleção, pipeline final, avaliação no teste e conclusão |

---

**← Anterior:** [Aula 5.4](../Aula5.4/README.md) · **Voltar:** [Aula 5](../README.md)
