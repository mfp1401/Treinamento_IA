# Aula 7.1 — Regularização

A regularização adiciona ao custo da regressão uma **penalidade pelo tamanho dos coeficientes**. Isso segura coeficientes exagerados e ajuda a evitar overfitting. O Lasso ainda consegue zerar coeficientes, fazendo seleção de variáveis automaticamente.

## Conteúdos

- Regressão linear comum (MQO) como baseline
- Ridge (penalidade L2), Lasso (L1) e ElasticNet (L1 + L2)
- `Pipeline` com `StandardScaler`: a penalidade só é justa com as variáveis na mesma escala
- Busca de `alpha` e `l1_ratio` com `GridSearchCV` e `KFold`
- Métrica: RMSE na validação cruzada

## Resultados (`load_diabetes`: 442 pacientes, 10 variáveis)

| Modelo | RMSE (validação cruzada) |
|---|---|
| Lasso | 55,35 |
| ElasticNet | 55,37 |
| Ridge | 55,37 |
| MQO | 55,39 |

Os quatro ficaram praticamente empatados: com só 10 variáveis e 442 amostras, a regressão comum não chega a sofrer overfitting. O Lasso zerou o coeficiente de `s3` e foi o escolhido, com **RMSE no teste = 53,68** e **R² = 0,456**.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado.py` | comparação MQO × Ridge × Lasso × ElasticNet, coeficientes e avaliação no teste |

---

**Voltar:** [Aula 7](../README.md)
