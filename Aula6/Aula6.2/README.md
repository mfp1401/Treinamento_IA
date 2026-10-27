# Aula 6.2 — Regressão linear pela equação normal

A regressão linear procura os coeficientes β que minimizam a soma dos erros ao quadrado. Igualando o gradiente a zero, chega-se à **equação normal**, XᵀXβ = Xᵀy, que tem solução direta, sem iterações.

## Conteúdos

- Coluna de 1s para o intercepto (β₀)
- Três formas de calcular β: `np.linalg.inv`, `np.linalg.solve` e `np.linalg.pinv`
- R² = 1 − SQres / SQtot
- Número de condição (`np.linalg.cond`) e matrizes singulares
- Comparação com o `LinearRegression` do scikit-learn

## Desafio de casa: classe `MinimosQuadrados`

`MinimosQuadrados(metodo="pinv")` aceita `"inv"`, `"solve"` ou `"pinv"`, com `fit`, `predict` e `score` (R²).

**Resultados no California Housing (20 640 casas, 8 características):**

| Teste | Resultado |
|---|---|
| `coef_`, `predict` e `score` iguais ao `LinearRegression` | `True` nos três métodos |
| R² | 0,606 |
| Número de condição de Xb | ≈ 2,4 × 10⁵ |
| Número de condição com a coluna `MedInc` duplicada | ≈ 5,9 × 10¹⁷ |
| Coluna duplicada: `inv` e `solve` | erro de matriz singular |
| Coluna duplicada: `pinv` | funciona (solução de menor norma), R² = 0,606 |

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado.py` | equação normal no California Housing, comparada ao scikit-learn |
| `desafio.py` | classe `MinimosQuadrados`, testes de validação e caso da coluna duplicada |

---

**← Anterior:** [Aula 6.1](../Aula6.1/README.md) · **Voltar:** [Aula 6](../README.md) · **Próxima →** [Aula 6.3](../Aula6.3/README.md)
