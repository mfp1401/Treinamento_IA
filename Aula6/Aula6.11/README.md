# Aula 6.11 — PCA por álgebra linear

Dá para representar um conjunto de dados com **menos características** sem perder quase nada da informação? Quando várias colunas variam juntas, elas carregam informação repetida. O **PCA** (Análise de Componentes Principais) encontra novas direções, ortogonais entre si, ordenadas da que tem **mais variância** para a que tem menos, e projeta os dados nas primeiras delas.

```text
centralizar → matriz de covariância → autovalores e autovetores → ordenar
            → variância explicada → escolher k componentes → projetar → reconstruir
```

## Conteúdos

- Dimensionalidade e redundância entre características
- Álgebra linear: produto escalar, norma, ortogonalidade e produto matricial
- Média por coluna e centralização: Xc = X − μ
- Variância e covariância amostral (n − 1)
- Matriz de covariância: C = XcᵀXc / (n − 1)
- Autovalores e autovetores com `np.linalg.eigh` (a covariância é simétrica)
- Variância explicada (EVR) e variância acumulada
- Projeção Z = Xc·W e reconstrução X̂ = Z·Wᵀ + μ
- PCA por SVD: autovalores = S² / (n − 1)
- Ambiguidade de sinal: um componente com o sinal invertido é o mesmo componente
- Efeito da padronização e PCA sem vazamento dentro de um `Pipeline`

## Exercício guiado: reduzir o Iris de 4D para 2D (20 etapas)

| Etapas | O que faz | Resultado |
|---|---|---|
| 01–04 | imports, carregar e explorar o Iris, gráficos dos 6 pares de características | `X` (150, 4) |
| 05–07 | média por coluna e centralização | médias ≈ 0 |
| 08–09 | covariância manual, conferida com `np.cov` | matriz 4 × 4 |
| 10–11 | autovalores e autovetores, ordenados | 4,228 · 0,243 · 0,078 · 0,024 |
| 12–13 | variância explicada e acumulada | PC1 92,46% · PC2 5,31% → **97,77%** |
| 14–17 | matriz W (4 × 2), projeção 150 × 2 e reconstrução | MSE de reconstrução 0,0253 |
| 18–19 | PCA por SVD e pelo scikit-learn | mesmos autovalores, EVR e reconstrução |
| 20 | efeito da padronização | PC1 + PC2 cai para 95,81% com `StandardScaler` |
| extra | PCA + regressão logística num `Pipeline` | acurácia no teste 0,90 |

## Desafio de casa: dataset Wine (178 vinhos, 13 características químicas)

Notebook executado: [`desafio.ipynb`](desafio.ipynb), com todas as saídas e gráficos.

| Etapas | O que faz |
|---|---|
| 1–3 | carregar, explorar as escalas das colunas e centralizar |
| 4–7 | covariância 13 × 13, autovalores, variância explicada e os dois primeiros componentes |
| 8–10 | projetar para 2D, gráfico com os cultivares identificados por cor e comparação com o scikit-learn |
| 11–12 | padronizado × não padronizado e separação dos cultivares no plano |

**Resultados:**

| | Sem padronização | Com padronização |
|---|---|---|
| PC1 + PC2 (variância explicada) | **99,98%** | 55,41% |
| O que PC1 mede | praticamente só `proline` (peso 1,00) | flavonoides, fenóis totais e OD280/OD315 |
| Cultivares separados no 2D* | 72,5% | **97,2%** |

\* Acerto ao classificar cada vinho pelo centro do cultivar mais próximo no plano PC1 × PC2. É só uma medida de separação visual, não um modelo avaliado em teste.

**Análise:** sem padronização, o PCA parece quase perfeito, mas é ilusão de escala. A variância de `proline` (≈ 99 000) é maior que a de todas as outras colunas juntas, então o PC1 vira basicamente essa coluna sozinha e os cultivares 1 e 2 ficam misturados no gráfico. Com padronização, todas as características pesam igual: a variância explicada cai para 55%, mas os componentes passam a combinar várias medidas químicas, e os três cultivares aparecem bem separados. **Variância explicada alta não garante uma projeção útil**: com colunas em escalas diferentes, padronize antes do PCA.

A implementação manual confere com o `PCA` do scikit-learn nos dois casos (mesma variância explicada, mesmas projeções a menos do sinal e mesma reconstrução).

## Erros comuns

| Erro | Correção |
|---|---|
| Esquecer de centralizar | `Xc = X - X.mean(axis=0)` antes da covariância |
| Média por linha em vez de coluna | `axis=0` |
| Covariância com o formato errado | `Xc.T @ Xc` dá p × p; `Xc @ Xc.T` daria n × n |
| Não ordenar os autovalores | `np.argsort(autovalores)[::-1]` e reordenar as **colunas** dos autovetores |
| Usar as linhas dos autovetores | os autovetores são as **colunas** da matriz |
| Achar que sinais opostos são erro | comparar valores absolutos ou a reconstrução |
| Ajustar o PCA antes de separar treino e teste | separar primeiro e usar `Pipeline` |

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado.py` | as 20 etapas da aula com o Iris (slides 42 a 51) |
| `desafio.ipynb` | desafio com o Wine, executado (Restart → Run All) |

---

**← Anterior:** [Aula 6.10](../Aula6.10/README.md) · **Voltar:** [Aula 6](../README.md) · **Próxima →** [Aula 7](../../Aula7/README.md)
