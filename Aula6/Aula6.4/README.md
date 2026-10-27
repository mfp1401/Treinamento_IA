# Aula 6.4 — Regressão logística do zero

Para classificar em 0 ou 1, a regressão logística calcula z = Xβ e passa o resultado pela **sigmoide**, que transforma qualquer número numa probabilidade entre 0 e 1. O treino usa o mesmo gradiente descendente da 6.3, agora com a **entropia cruzada** como custo.

## Conteúdos

- Sigmoide: σ(z) = 1 / (1 + e⁻ᶻ)
- Entropia cruzada e por que ela pune previsões confiantes e erradas
- Gradiente: Xᵀ(p − y) / m
- Limiar de 0,5 para transformar probabilidade em classe
- Fronteira de decisão: x₂ = −(β₀ + β₁x₁) / β₂

## Desafio de casa: classe `RegressaoLogistica`

Métodos `fit`, `predict_proba`, `predict` e `score` (acurácia).

**Critérios de aceite e resultados:**

| Critério | Resultado |
|---|---|
| `len(historico)` igual ao número de épocas | ✓ 1000 |
| Primeiro custo = ln 2 ≈ 0,6931 (β começa em zero) | ✓ 0,6931 |
| Último custo menor que o primeiro | ✓ cerca de 0,19 |
| `predict_proba` em (0, 1) e `predict` só 0 e 1 | ✓ |
| Acurácia no teste ≥ 0,85 | ✓ 0,900 |
| Fronteira desenhada a partir do β aprendido | ✓ β = [0,257; 2,116; 2,653] |

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado.py` | sigmoide, custo, gradiente, treino e fronteira |
| `desafio.py` | classe `RegressaoLogistica` e verificação dos critérios |

---

**← Anterior:** [Aula 6.3](../Aula6.3/README.md) · **Voltar:** [Aula 6](../README.md) · **Próxima →** [Aula 6.5](../Aula6.5/README.md)
