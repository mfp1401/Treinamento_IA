# Aula 6.9 — Árvore de decisão do zero

A árvore de decisão classifica fazendo uma **sequência de perguntas** do tipo "característica ≤ limiar?". Nenhuma pergunta é escrita à mão: o algoritmo testa todos os cortes possíveis, escolhe o que mais reduz a impureza de **Gini** e repete o processo em cada lado, de forma recursiva.

## Conteúdos

- Impureza de Gini: 1 − Σ pᵢ²
- Gini ponderado e ganho de uma divisão
- Busca do melhor corte (pontos médios entre valores)
- Recursividade e caso base
- Critérios de parada: nó puro, `max_depth`, `min_samples_split`
- Overfitting e poda
- Comparação com o `DecisionTreeClassifier`

## Exercício guiado: 24 clientes (idade × visitas à loja)

```text
[Raiz] Visitas <= 3.5?
    [Sim] Classe 0  (4 amostras)
    [Não] Idade <= 53?
        [Sim] Idade <= 23.5?
            [Sim] Classe 0  (1 amostra)
            [Não] Classe 1  (10 amostras)
        [Não] Classe 0  (3 amostras)
```
Acurácia: treino 0,944 · teste 0,833 · previsões iguais às do scikit-learn.

## Desafio de casa: dataset Iris (150 flores, 3 espécies)

Mesma árvore aplicada a outro problema, agora com **3 classes**, usando o comprimento e a largura da pétala.

| max_depth | Treino | Teste |
|---|---|---|
| 1 | 0,667 | 0,667 |
| 2 | 0,971 | 0,889 |
| **3** | **0,981** | **0,933** |
| 4 | 0,981 | 0,956 |
| 5 | 0,990 | 0,956 |

Com `max_depth=3`, a árvore faz as mesmas previsões do `DecisionTreeClassifier` (3 níveis, 5 folhas). A primeira pergunta ("comprimento da pétala ≤ 2,45?") separa sozinha todas as setosas, e os erros ficam todos entre versicolor e virginica, que são parecidas.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado.py` | as 18 etapas da aula, do Gini à comparação com o scikit-learn |
| `desafio.py` | árvore no Iris: previsões, árvore impressa, regiões de decisão e análise da profundidade |

---

**← Anterior:** [Aula 6.8](../Aula6.8/README.md) · **Voltar:** [Aula 6](../README.md) · **Próxima →** [Aula 7](../../Aula7/README.md)
