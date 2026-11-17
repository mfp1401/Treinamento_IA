# Aula 7.4 — Boosting: XGBoost, LightGBM e CatBoost

No **Bagging** (Random Forest), as árvores são treinadas de forma independente e depois combinadas. No **Boosting**, elas são treinadas **em sequência**: cada nova árvore aprende os erros (pseudo-resíduos) do conjunto até ali e soma uma pequena correção à previsão.

```text
F₀ = constante → resíduos → árvore h₁ → F₁ = F₀ + η·h₁ → novos resíduos → h₂ → … → F_M
```

## Conteúdos

- **Bagging × Boosting:** independente × sequencial; reduz variância × reduz viés
- **Resíduos e pseudo-resíduos:** r = −∂L/∂F. Na perda quadrática, r = y − F; na log loss (classificação), r = y − p
- **Gradient Boosting:** Fₘ = Fₘ₋₁ + η·hₘ. O `learning_rate` (η) encolhe cada correção, e η menor pede mais árvores
- **Early stopping:** para quando a métrica da **validação** deixa de melhorar e guarda a melhor iteração
- **XGBoost:** usa gradiente e Hessiana; objetivo regularizado Ω = γT + ½λΣw² + αΣ|w|; peso ótimo da folha w* = −G / (H + λ)
- **LightGBM:** histogramas (bins), crescimento **leaf-wise**, `num_leaves ≤ 2^max_depth`, GOSS e EFB. Atenção: `subsample` só funciona com `subsample_freq ≥ 1`
- **CatBoost:** árvores simétricas, categorias tratadas com estatísticas ordenadas e *Ordered Boosting* contra o prediction shift
- **Comparação justa:** mesmo split, mesma métrica de early stopping, mesmo limiar, teste fora do `eval_set`, versões registradas

## Exercício guiado: benchmark no mesmo split (`make_classification`, 2.400 amostras, 20 etapas)

Notebook: [`exercicio_guiado.ipynb`](exercicio_guiado.ipynb). Divisão 1.440 / 480 / 480, até 800 árvores, `learning_rate = 0.05`, early stopping de 40 rodadas pela AUC.

| Modelo | Acurácia | Precisão | Recall | F1 | ROC AUC | Log Loss | Árvores | Tempo (s) |
|---|---|---|---|---|---|---|---|---|
| Baseline | 0,548 | 0,000 | 0,000 | 0,000 | 0,500 | 0,689 | — | 0,001 |
| XGBoost | 0,894 | 0,903 | 0,857 | 0,879 | 0,964 | 0,246 | 173 | 0,38 |
| **LightGBM** | **0,908** | **0,926** | 0,866 | 0,895 | 0,966 | **0,233** | 288 | **0,28** |
| **CatBoost** | 0,906 | 0,902 | **0,889** | **0,896** | **0,971** | 0,251 | 152 | 0,58 |

- **Maior AUC e F1:** CatBoost. **Menor Log Loss e menor tempo:** LightGBM.
- **Gap de AUC treino − validação:** XGBoost 0,040, LightGBM 0,040 e CatBoost 0,027 (o menor).
- As importâncias por permutação concordam muito entre as três bibliotecas (Spearman de 0,95 a 0,98).
- Exemplo complementar: CatBoost com colunas categóricas (`plano`, `regiao`) direto no `cat_features`, com AUC de 0,829.

## Desafio final: investigando o Boosting (`load_breast_cancer`)

Notebook executado: [`desafio.ipynb`](desafio.ipynb). São 569 tumores com 30 medidas, e aqui a classe **1 = maligno**. A divisão é 341 / 114 / 114, com as 14 etapas do enunciado e uma Random Forest como referência da aula 7.3.

| Modelo | Acurácia | Precisão | Recall | F1 | ROC AUC | Log Loss | Árvores |
|---|---|---|---|---|---|---|---|
| Baseline | 0,632 | 0,000 | 0,000 | 0,000 | 0,500 | 0,658 | — |
| XGBoost | 0,974 | 1,000 | 0,929 | 0,963 | 0,9947 | **0,080** | 148 |
| LightGBM | 0,974 | 1,000 | 0,929 | 0,963 | **0,9954** | 0,085 | 111 |
| CatBoost | 0,974 | 1,000 | 0,929 | 0,963 | 0,9950 | 0,086 | 157 |
| Random Forest (ref.) | 0,956 | 0,974 | 0,905 | 0,938 | 0,9957 | 0,113 | 500 |

**Análise:**
- **Empate técnico:** os três boostings têm a mesma matriz de confusão (nenhum falso alarme e 3 malignos não detectados). As AUCs diferem só na terceira casa decimal, e 1 amostra do teste vale 0,88 p.p.
- **Contra a Random Forest:** o boosting teve mais acurácia e probabilidades melhores (Log Loss de 0,08 × 0,11).
- **Early stopping:** os modelos pararam entre 111 e 157 das 800 árvores possíveis. O LightGBM foi o mais rápido (≈ 0,03 s) e o CatBoost teve o menor gap treino × validação.
- **Dois hiperparâmetros por biblioteca** (grade 3 × 3, escolhida pela validação):
  - O CatBoost manteve `depth=4` e `l2_leaf_reg=3`.
  - O XGBoost escolheu `learning_rate=0.3`: acertou **1 caso a mais** no teste, mas a AUC e o Log Loss pioraram. Isso é ruído, não melhora.
- **Lição da grade:** com `learning_rate=0.03`, o early stopping **pela AUC** parou com só 10 a 29 árvores, e o Log Loss ficou em 0,48. A AUC mede só a ordenação, que estabiliza antes de as probabilidades estarem boas.

## Erros comuns

| Erro | Correção |
|---|---|
| Achar que as árvores do Boosting são independentes | a árvore m depende das árvores 1 … m − 1 |
| Usar o teste no `eval_set` / early stopping | parar pela validação; o teste fica para o fim |
| `subsample=0.8` no LightGBM sem `subsample_freq` | `subsample_freq=1`; o padrão 0 desativa a amostragem |
| `num_leaves` maior que `2^max_depth` | manter `num_leaves ≤ 2^max_depth` (o desafio usa 4, 8 e 15 com depth 4) |
| Confundir o `gamma` do XGBoost com o `gamma` da SVM | no XGBoost, é o ganho mínimo para dividir uma folha |
| Comparar importâncias nativas (gain, split) entre bibliotecas | usar a mesma permutation importance para as três |
| Tratar AUC como acurácia | a AUC mede a ordenação e não depende do limiar |
| Declarar um vencedor por 1 amostra de diferença | olhar várias métricas; 1/114 = 0,88 p.p. está dentro do ruído |
| Não registrar as versões | APIs mudam (ex.: `early_stopping_rounds` foi para o construtor no XGBoost ≥ 1.6) |

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado.ipynb` | benchmark XGBoost × LightGBM × CatBoost no `make_classification`, 20 etapas + exemplo com categorias |
| `desafio.ipynb` | desafio final com o Breast Cancer, executado: métricas, matrizes, tempo, curvas de validação, importâncias, ajuste de dois hiperparâmetros por biblioteca e conclusão |

Para rodar: `pip install xgboost lightgbm catboost`.

---

**← Anterior:** [Aula 7.3](../Aula7.3/README.md) · **Voltar:** [Aula 7](../README.md)
