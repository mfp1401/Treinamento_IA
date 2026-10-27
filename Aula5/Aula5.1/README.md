# Aula 5.1 — KNN, treino/teste e baseline

O primeiro classificador: o **KNN** (`KNeighborsClassifier`) prevê a classe de uma amostra olhando para os vizinhos mais próximos. Para saber se o modelo é bom de verdade, ele é comparado com um **baseline** (`DummyClassifier`), que sempre chuta a classe mais frequente.

## Conteúdos

- Separar treino e teste com `train_test_split` (`stratify`, `random_state`)
- Treinar e prever com `KNeighborsClassifier(n_neighbors=5)`
- Avaliar com `accuracy_score` e `confusion_matrix`
- Comparar com o `DummyClassifier` (baseline)

## Arquivos

| Arquivo | Dataset |
|---|---|
| `exercicio_guiado_iris.py` | Iris: 3 espécies de flores |
| `exercicio1_wine.py` | Wine: 3 tipos de vinho |
| `exercicio2_breast_cancer.py` | Breast Cancer: tumor maligno ou benigno |

---

**Voltar:** [Aula 5](../README.md) · **Próxima →** [Aula 5.3](../Aula5.3/README.md)
