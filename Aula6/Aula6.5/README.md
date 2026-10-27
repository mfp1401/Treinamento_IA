# Aula 6.5 — Métricas de classificação

A acurácia sozinha engana, principalmente quando uma classe é rara. Esta aula constrói, a partir da **matriz de confusão**, as métricas que mostram onde o modelo acerta e onde erra.

## Conteúdos

- Verdadeiros/falsos positivos e negativos (TP, TN, FP, FN)
- Matriz de confusão
- Precision, recall e F1-score
- Extensão para várias classes (one-vs-rest)
- Médias macro e weighted
- Curva ROC e AUC (regra dos trapézios)

## Desafio de casa: `meu_classification_report(y_real, y_pred)`

Reproduz o `classification_report` do scikit-learn sem usar funções prontas de métrica.

| Exemplo | Resultado |
|---|---|
| Spam (2 classes) | matriz `[[3 2] [1 4]]`, accuracy 0,70 |
| Pinguins (3 classes) | matriz `[[8 2 0] [1 3 1] [0 1 9]]`, accuracy 0,80, F1 macro 0,76 e weighted 0,81 |
| ROC / AUC | AUC = 0,84, igual ao `roc_auc_score` |

As duas tabelas (a nossa e a do scikit-learn) saem idênticas.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `desafio.py` | `meu_classification_report`, `curva_roc` e `auc`, com a comparação com o scikit-learn |

---

**← Anterior:** [Aula 6.4](../Aula6.4/README.md) · **Voltar:** [Aula 6](../README.md) · **Próxima →** [Aula 6.6](../Aula6.6/README.md)
