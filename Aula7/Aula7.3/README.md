# Aula 7.3 — Random Forest

Uma árvore de decisão sozinha é **instável**: poucos pontos diferentes no treino mudam a raiz e, com ela, tudo o que vem abaixo (alta variância). A **Random Forest** treina centenas de árvores, cada uma em uma amostra **bootstrap** do treino e sorteando **algumas características a cada divisão**, e depois faz a média das probabilidades delas. Como as árvores erram em lugares diferentes, a média cancela boa parte desses erros.

```text
treino → B amostras bootstrap → B árvores (max_features sorteadas por nó) → média de predict_proba → classe
                 └── amostras que ficaram de fora (≈36,8%) → OOB
```

## Conteúdos

- Ensemble learning: combinar modelos só ajuda se cada um for útil **e** se errarem de formas diferentes (3 modelos com 70% de acerto → 78,4% na votação)
- **Bootstrap**: amostragem com reposição, `rng.choice(n, n, replace=True)`
- **OOB**: P(ficar de fora) = (1 − 1/n)ⁿ ≈ e⁻¹ ≈ 36,8%
- **Bagging × Random Forest**: a floresta sorteia também as colunas (`max_features="sqrt"`) em **cada divisão**
- Variância da média: ρσ² + (1 − ρ)σ²/B. Mais árvores só derrubam o 2º termo; o 1º só cai com árvores menos correlacionadas
- Agregação no scikit-learn: `predict` = argmax da **média de `predict_proba`** (não é votação de rótulos)
- Hiperparâmetros: `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, `min_samples_leaf`, `bootstrap`, `oob_score`, `random_state`, `n_jobs`
- Importância: **MDI** (redução de Gini, no treino) × **permutação** (queda da métrica, no teste)

## Exercício guiado: uma árvore contra 500 (`make_moons`, 600 amostras, noise = 0,28, 20 etapas)

| Modelo | Árvores | max_depth | Treino | Teste | Matriz (teste) | OOB |
|---|---|---|---|---|---|---|
| Árvore de decisão | 1 | None | 1,000 | 0,907 | [[69, 6], [8, 67]] | n/a |
| Árvore rasa | 1 | 4 | 0,902 | 0,900 | [[67, 8], [7, 68]] | n/a |
| **Random Forest** | 500 | None | 1,000 | **0,920** | [[69, 6], [6, 69]] | **0,920** |
| Random Forest rasa | 500 | 4 | 0,911 | 0,893 | [[65, 10], [6, 69]] | 0,891 |

- A árvore profunda (13 níveis, 51 folhas) decora o treino e cai para 90,7% no teste. A floresta chega a **92,0%**: 2 acertos a mais em 150.
- O **OOB deu exatamente 0,920**, igual ao teste, estimado sem gastar dados de teste.
- Com 1 árvore: 84%; com 25 ou mais: 92%, e daí não sobe mais (ganho decrescente).
- Importância: MDI x1 = 0,454 e x2 = 0,546; a permutação concorda (x2 um pouco à frente).
- Custo: treino de 0,004 s (árvore) × 0,89 s (floresta); previsão de 0,0003 s × 0,095 s.

## Desafio de casa: investigando a Random Forest (`load_breast_cancer`)

Notebook executado: [`desafio.ipynb`](desafio.ipynb). São 569 tumores e 30 medidas das células, para prever **maligno × benigno**, com as 12 etapas do enunciado.

| Modelo | Treino | Teste | Precisão (maligno) | Recall (maligno) | F1 | OOB |
|---|---|---|---|---|---|---|
| Árvore (sem limite) | 1,000 | 0,923 | 0,875 | 0,925 | 0,899 | — |
| Árvore rasa (depth 4) | 0,988 | 0,944 | 0,925 | 0,925 | 0,925 | — |
| **Random Forest (500)** | 1,000 | **0,958** | **0,961** | 0,925 | **0,942** | 0,962 |
| Random Forest rasa (depth 4) | 0,993 | 0,958 | 0,961 | 0,925 | 0,942 | 0,955 |

**Análise:**
- A floresta acertou **5 casos a mais** que a árvore em 143. O ganho veio dos falsos alarmes (7 → 2 benignos chamados de malignos). Os 4 malignos não detectados são os mesmos em todos os modelos.
- **Profundidade:** a árvore piora quando fica profunda (94,4% com depth 3–4 → 92,3% sem limite). A floresta chega a 95,8% com depth 3 e **não piora mais**: a média das 500 árvores segura a variância.
- **Nº de árvores:** 100 árvores já dão 95,8%; com 500, o teste fica igual e o treino demora 5 vezes mais.
- **OOB 96,2% × teste 95,8%:** diferença menor que um acerto no teste (1 ponto = 0,70 p.p.).
- **Importâncias:** pelo MDI, as que mais pesam são as medidas de tamanho e concavidade (`worst area`, `worst perimeter`, `worst concave points`). A permutação deu **quase zero para todas**, porque o dataset tem 21 pares de variáveis com correlação acima de 0,9: embaralhando `worst area`, a floresta usa `worst perimeter`, que traz a mesma informação. É o ponto fraco da permutação mostrado na aula.

## Erros comuns

| Erro | Correção |
|---|---|
| Confundir bootstrap com a divisão treino/teste | o split é feito uma vez; o bootstrap acontece dentro do treino, para cada árvore |
| `replace=False` no bootstrap | `replace=True`: com reposição |
| Achar que `max_features` sorteia linhas | o bootstrap sorteia linhas; `max_features` sorteia colunas a cada divisão |
| `oob_score=True` com `bootstrap=False` | dá `ValueError`; o OOB exige bootstrap |
| Tratar o OOB como acurácia de teste | o OOB é uma estimativa interna; o teste é a avaliação final |
| Achar que mais árvores sempre aumentam a acurácia | o ganho é decrescente (aqui, parou em 100) |
| Ler importância como causalidade | ela mostra o que o modelo usou, não o que causa o resultado |
| Confiar na permutação com variáveis correlacionadas | ela subestima variáveis que têm "substitutas" (ver o desafio) |

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado.py` | as 20 etapas da aula: árvore × Random Forest no `make_moons`, fronteiras, OOB, importâncias, nº de árvores e profundidade |
| `desafio.ipynb` | desafio com o Breast Cancer, executado, com matrizes de confusão, gráficos, OOB, importâncias e conclusão |

---

**← Anterior:** [Aula 7.2](../Aula7.2/README.md) · **Voltar:** [Aula 7](../README.md)
