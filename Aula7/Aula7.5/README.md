# Aula 7.5 — Otimização de hiperparâmetros: Grid Search, Random Search e Optuna

**Parâmetros** o modelo aprende no `fit()` (pesos, divisões das árvores). **Hiperparâmetros** nós escolhemos antes (`max_depth`, `learning_rate`, `C`...). Escolher hiperparâmetros é um problema de busca: θ* = argmax J(θ), em que J é a métrica de **validação** e cada avaliação exige treinar modelos.

```text
espaço de busca + métrica de validação + orçamento + estratégia = otimização de hiperparâmetros
```

## Conteúdos

- **Espaço de busca:** tipos de hiperparâmetro (categórico, inteiro, contínuo linear e contínuo log). `learning_rate` vai em **escala log**
- **Grid Search:** testa todas as combinações de uma grade fixa. Custo = ∏ tamanhos × folds, que cresce exponencialmente (4ᵏ)
- **Random Search:** sorteia `n_iter` configurações de distribuições (`randint`, `uniform`, `loguniform`). Testa mais valores distintos nas dimensões que importam
- **Otimização adaptativa:** a próxima configuração depende do histórico, equilibrando exploração e aproveitamento
- **TPE (Optuna):** separa os trials em bons e demais, e propõe onde l(x)/g(x) é maior. **Não** é um Processo Gaussiano
- **Optuna:** `Study` (o estudo), `Trial` (uma tentativa), `objective`, `suggest_int/float/categorical`, `TPESampler`
- **Pruning:** o `MedianPruner` interrompe um trial abaixo da mediana dos anteriores **na mesma etapa**. Exige `report()` + `should_prune()` e treino em etapas (`warm_start=True`)
- **Sampler × pruner:** o sampler escolhe os valores; o pruner decide se o trial continua
- **Protocolo justo:** mesmo split, mesmos folds, mesma métrica e mesmo orçamento. O teste só no fim

## Exercício guiado: mesmo orçamento, três estratégias (`make_classification`, 1.800 amostras, 22 etapas)

Notebook: [`exercicio_guiado.ipynb`](exercicio_guiado.ipynb). `GradientBoostingClassifier` com 150 árvores, otimizado pela ROC AUC em 3 folds, com 12 configurações por estratégia.

| Estratégia | Melhores parâmetros | AUC CV | AUC teste | Acurácia | Log Loss | Busca (s) |
|---|---|---|---|---|---|---|
| Grid Search | depth 3, lr 0,10, leaf 1 | 0,9285 | 0,9488 | 0,875 | 0,299 | 16,2 |
| **Random Search** | depth 3, lr 0,142, leaf 3 | **0,9309** | **0,9561** | 0,875 | **0,273** | 16,1 |
| Optuna (TPE + pruning) | depth 3, lr 0,043, leaf 2 | 0,9168 | 0,9380 | 0,875 | 0,344 | 18,9 |

- O **Random Search** venceu: sorteou um `learning_rate` (0,142) fora dos pontos da grade.
- O **Optuna interrompeu 2 trials** e treinou 4.800 árvores contra 5.400, mas com só 12 trials (5 deles iniciais e aleatórios) o TPE teve pouca informação e ficou atrás.
- A acurácia no teste foi a mesma nas três (0,875).

## Desafio de casa: otimizando um modelo no dataset Wine

Notebook executado: [`desafio.ipynb`](desafio.ipynb). São 178 vinhos, 13 medidas químicas e 3 classes, com **acurácia** como métrica. Usa `GradientBoostingClassifier`, 5 folds fixos, 12 configurações por estratégia e o mesmo critério de desempate (log loss). O teste tem 36 vinhos e só é usado no fim.

| Estratégia | Configuração escolhida | Acurácia CV | Log loss CV | Etapas treinadas | Acurácia teste | Log loss teste |
|---|---|---|---|---|---|---|
| Sem otimização | depth 3, lr 0,1, leaf 1 | 0,923 | — | — | 0,944 (2 erros) | 0,466 |
| Grid Search | depth 1, lr 0,1, leaf 1 | **0,972** | **0,073** | 9.000 | **1,000 (0 erros)** | **0,025** |
| Random Search | depth 2, lr 0,028, leaf 5 | 0,965 | 0,107 | 9.000 | 0,972 (1 erro) | 0,079 |
| **Optuna** | depth 1, lr 0,141, leaf 5 | **0,972** | 0,075 | **7.500** | 0,972 (1 erro) | 0,034 |

**Análise:**
- **Otimizar valeu:** as três estratégias trocaram as árvores de 3 níveis do padrão por árvores **rasas** (1 ou 2 níveis). O log loss no teste caiu de 0,47 para 0,02 a 0,08.
- **O Grid teve sorte:** a melhor combinação (`depth=1`, `lr=0.1`) estava na grade.
- **O Random achou a mesma região** (`depth=1`, `lr=0.142`, acurácia 0,9650), mas perdeu para outra configuração por **0,0003** de acurácia. Foi um empate decidido por arredondamento.
- **O Optuna:** depois dos 5 trials iniciais, o TPE concentrou as propostas em `depth=1` com `lr` entre 0,12 e 0,19. O pruner cortou 3 trials e economizou 17% do treino.
- **Tempo:** as três buscas ficaram entre 15 e 18 s. No Optuna, a avaliação a cada checkpoint consumiu a economia do pruning.
- **Teste:** com 36 vinhos, 1 erro vale 2,8 p.p., então o teste não separa as estratégias.
- **Conclusão:** o **Optuna** foi a estratégia mais interessante. Igualou a melhor validação com menos treino e sem depender de uma grade bem escolhida.

## Erros comuns

| Erro | Correção |
|---|---|
| Confundir parâmetros com hiperparâmetros | parâmetros vêm do `fit`; hiperparâmetros, do construtor |
| Escolher o modelo pelo teste | escolher pela validação; o teste mede uma única vez, no fim |
| Validação cruzada usando o teste | a CV acontece só dentro de `X_train` |
| Comparar estratégias com splits ou orçamentos diferentes | a mesma lista `splits` e o mesmo número de configurações nas três |
| `learning_rate` em escala linear | `loguniform` / `suggest_float(..., log=True)` |
| `randint(1, 3)` achando que inclui o 3 | o limite superior do `randint` é exclusivo; o `suggest_int` inclui os dois extremos |
| Confundir sampler com pruner | o sampler escolhe os valores; o pruner interrompe trials |
| Pruning sem etapas intermediárias | treinar em etapas (`warm_start`) e fazer `report()` a cada checkpoint |
| Contar trials como modelos treinados | contar folds × árvores; trials podados treinam menos |
| Achar que `best_score_` é o desempenho no teste | é o máximo de várias tentativas na validação, portanto otimista |
| Achar que Optuna ou Random sempre vencem o Grid | depende do espaço, do orçamento e do problema (no guiado venceu o Random; no Wine, empate entre o Grid e o Optuna) |

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado.ipynb` | Grid × Random × Optuna (TPE + MedianPruner) no `make_classification`, mesmo orçamento e mesmos folds |
| `desafio.ipynb` | desafio do Wine executado: exploração, espaço de busca, as três buscas, comparação, avaliação no teste, tempos e conclusão |

Para rodar: `pip install optuna`.

---

**← Anterior:** [Aula 7.4](../Aula7.4/README.md) · **Voltar:** [Aula 7](../README.md)
