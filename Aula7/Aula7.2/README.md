# Aula 7.2 — SVM e o truque do kernel

O **SVM** (Support Vector Machine) procura a fronteira que separa as classes com a **maior margem possível**, e essa fronteira depende só dos pontos mais próximos dela, os **vetores de suporte**. Quando uma reta não basta, o **truque do kernel** (RBF) mede a similaridade entre pontos pela distância, o que equivale a levá-los para uma dimensão maior, onde ficam separáveis, sem precisar calcular essa dimensão.

```text
dados → StandardScaler → SVC(kernel, C, gamma) → fronteira + margens + vetores de suporte
```

## Conteúdos

- Margem máxima e vetores de suporte
- Margem rígida × margem suave e o parâmetro **C** (custo de cada violação)
- Kernel linear × kernel **RBF**: K(x, x′) = exp(−γ‖x − x′‖²)
- **gamma**: o alcance da influência de cada vetor de suporte (`gamma="scale"` = 1 / (n_features · variância))
- `decision_function`: fronteira em 0 e margens em ±1
- Escalonamento obrigatório, sempre dentro do `Pipeline`
- Underfitting × overfitting controlados por C e gamma

## Exercício guiado: disco dentro de um anel (`make_circles`, 19 etapas)

| Etapas | O que faz | Resultado |
|---|---|---|
| 01–04 | imports, dataset (300 pontos), gráfico, treino/teste 75/25 | classes 150/150 |
| 05–10 | SVM linear: pipeline, treino, previsão, acurácia e fronteira | teste **0,493**: uma reta não separa disco e anel |
| 11–14 | SVM RBF e fronteiras lado a lado | teste **1,000**, fronteira fechada em volta do disco |
| 15 | vetores de suporte nas coordenadas originais | 44 vetores (22 por classe) |
| 16–17 | experimentos com C (0,01 · 1 · 100) e gamma (0,1 · 1 · 10) | C = 0,01 cai para 0,493 |
| 18–19 | tabela comparativa e conclusão | — |

## Desafio de casa: fronteiras de decisão no `make_moons`

Notebook executado: [`desafio.ipynb`](desafio.ipynb). Dataset `make_moons(n_samples=300, noise=0.20, random_state=42)`, com as 13 etapas do enunciado e a extensão opcional com `GridSearchCV`.

| Modelo | Treino | Teste | Vetores de suporte |
|---|---|---|---|
| Linear | 0,884 | 0,853 | 69 |
| RBF (C = 1, gamma = scale) | 0,942 | 0,907 | 62 |
| RBF C = 0,01 | 0,604 | 0,627 | 224 |
| RBF C = 10 | 0,982 | 0,973 | 39 |
| RBF gamma = 0,01 | 0,867 | 0,813 | 120 |
| RBF gamma = 100 | **1,000** | 0,893 | 216 |
| **GridSearchCV (C = 1, gamma = 1)** | 0,969 | **0,960** | 56 |

**Análise:**
- As luas são *parcialmente* separáveis por uma reta, por isso o linear ainda chega a 85%, mas erra onde uma lua entra na outra. O RBF consegue curvar a fronteira.
- **C pequeno** (0,01) aceita erros demais: quase todo ponto vira vetor de suporte e o modelo mal aprende (underfitting). **C grande** passa a perseguir o ruído.
- **gamma pequeno** deixa a fronteira quase reta. **gamma grande** (100) cria "ilhas" em volta de cada ponto: acerta **100% do treino e cai para 89% no teste** (overfitting), com 216 vetores de suporte.
- O `GridSearchCV` escolheu C = 1 e gamma = 1 **só com validação cruzada no treino** e acertou 96% do teste. O C = 10 teve 97,3% no teste, mas escolher por esse número seria vazamento: o teste deve ser usado uma única vez.

## Erros comuns

| Erro | Correção |
|---|---|
| Treinar o SVM sem escalonar | `StandardScaler` dentro do `Pipeline` |
| Escalonar antes de dividir treino e teste | dividir primeiro; o `Pipeline` aprende a escala só no treino |
| Achar que treino 100% é bom sinal | comparar sempre com o teste (gamma = 100: 1,000 × 0,893) |
| Escolher C e gamma olhando o teste | `GridSearchCV` com validação cruzada no treino |
| Desenhar `support_vectors_` direto no gráfico | eles estão escalonados: use `X_train[svc.support_]` ou `inverse_transform` |

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado.py` | as 19 etapas da aula: SVM linear × RBF no `make_circles` |
| `desafio.ipynb` | desafio com o `make_moons`, executado, com fronteiras, métricas, C, gamma, GridSearchCV e conclusão |

---

**← Anterior:** [Aula 7.1](../Aula7.1/README.md) · **Voltar:** [Aula 7](../README.md)
