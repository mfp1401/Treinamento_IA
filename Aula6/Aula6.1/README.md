# Aula 6.1 — Álgebra linear e cálculo para ML

Quase todo modelo de Machine Learning se resume a **multiplicar matrizes e seguir um gradiente**. Esta aula revisa as operações básicas com NumPy e mostra por que o código vetorizado (sem `for`) é muito mais rápido do que percorrer os dados com laços.

## Conteúdos

- Produto matriz × vetor (`X @ w`) como "transformação" dos dados
- Vetorização × laços em Python e medição de tempo com `timeit`
- Distâncias entre vetores com broadcasting (`A[:, None, :] - B[None, :, :]`)
- Erro quadrático médio (MSE) e seu gradiente na forma matricial: (2/n)·Xᵀ(Xw − y)
- Conferência do gradiente pela diferença central (gradiente numérico)

## Desafio de casa: o módulo `algebra`

| Função | O que faz |
|---|---|
| `norma(x, ord)` | normas L1 e L2 |
| `distancias(A, B)` | matriz (m, k) de distâncias euclidianas |
| `mse(X, y, w)` e `grad_mse(X, y, w)` | erro e gradiente na forma matricial |
| `grad_numerico(f, w, h)` | gradiente por diferença central, coordenada por coordenada |

**Critérios de aceite:** nenhum `for` fora de `grad_numerico` · `norma` igual a `np.linalg.norm` · `distancias` igual a `scipy.spatial.distance.cdist` · gradiente analítico igual ao numérico.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado.py` | produto matricial, tempo, distâncias, MSE e checagem do gradiente |
| `desafio.py` | implementação e testes das funções do desafio |

---

**Voltar:** [Aula 6](../README.md) · **Próxima →** [Aula 6.2](../Aula6.2/README.md)
