# Aula 6.12 — Perceptron e MLP com backpropagation

O **Perceptron** é um único neurônio: soma as entradas com pesos, compara com um limiar e decide. Ele só consegue traçar **uma reta**, então resolve problemas como AND e OR, mas não o XOR. Uma **MLP** (rede com camada oculta) empilha neurônios com ativação não linear, e o **backpropagation** usa a regra da cadeia para descobrir como ajustar cada peso a partir do erro.

```text
entrada → camada oculta (pesos + sigmoid) → saída (pesos + sigmoid) → perda
        ← gradientes pela regra da cadeia (backpropagation) ←
```

## Conteúdos

- Perceptron: z = x·w + b, regra de atualização w ← w + η·y·x
- Separabilidade linear: AND e OR são separáveis por uma reta, o XOR não
- MLP: camadas, pesos W, bias b e ativação sigmoid
- Forward propagation: Z¹ = XW¹ + b¹, A¹ = σ(Z¹), Z² = A¹W² + b², A² = σ(Z²)
- Binary Cross-Entropy como função de perda
- Backpropagation e regra da cadeia: dZ² = A² − Y, dZ¹ = dZ²W²ᵀ ⊙ σ′
- Gradiente descendente: θ ← θ − η·∇L
- Inicialização aleatória para quebrar a simetria entre neurônios

## Exercício guiado: Perceptron no OR e MLP no XOR (20 etapas)

| Etapas | O que faz | Resultado |
|---|---|---|
| 01–03 | imports, tabela XOR e gráfico | classes em diagonais opostas |
| 04–06 | Perceptron no OR (rótulos −1/+1) | 4 de 4 acertos em 4 épocas |
| 07 | o mesmo Perceptron no XOR | só 2 de 4 depois de 100 épocas |
| 08–11 | sigmoid, parâmetros da MLP 2 → 4 → 1, forward e BCE | perda inicial 0,7285 |
| 12–14 | regra da cadeia num exemplo numérico, backward e atualização | dZ² = −0,2 · dA¹ = −0,4 · dZ¹ = −0,096 |
| 15–17 | treino (taxa 0,5, 20 000 épocas) e curva de perda | perda 0,7285 → **0,0007** |
| 18–20 | previsões, região de decisão e verificação | XOR `[0 1 1 0]` correto |

## Desafio de casa: construindo uma rede neural do zero

Notebook executado: [`desafio.ipynb`](desafio.ipynb), com todas as saídas, gráficos e a explicação matemática.

| Itens | O que faz |
|---|---|
| 1–2 | Perceptron para AND e a prova (experimento + desigualdades) de que ele não resolve o XOR |
| 3–6 | classe `MLP` em NumPy: forward, Binary Cross-Entropy e backpropagation, conferido com o gradiente numérico |
| 7–9 | treino 2 → 4 → 1, curva de perda e previsões |
| 10–11 | variação do número de neurônios ocultos e da taxa de aprendizado (10 sementes cada) |
| 12 | como a rede aprendeu: o que cada neurônio oculto faz e o "espaço oculto" |

**Resultados:**

- **AND:** o Perceptron acerta as 4 linhas em 4 épocas. **XOR:** depois de 100 épocas, ainda erra 2 de 4. As desigualdades mostram que não existe reta que resolva.
- **Backpropagation conferido:** o gradiente analítico bate com o numérico (diferença de 10⁻¹⁰).
- **MLP 2 → 4 → 1:** perda 0,7285 → 0,0016 em 10 000 épocas, com o XOR 100% correto.

| Neurônios ocultos | Acertou XOR (10 sementes) |
|---|---|
| 1 | 0/10 |
| 2 | 9/10 |
| 3, 4 e 8 | 10/10 |

| Taxa de aprendizado | Acertou XOR | O que acontece |
|---|---|---|
| 0,01 | 2/10 | lenta demais: perda quase não sai de 0,68 |
| 0,1 | 10/10 | fica num platô por ~5 000 épocas |
| 0,5 e 2 | 10/10 | rápida e estável |
| 10 | 10/10 | a mais rápida, mas com saltos no começo |

**Como a rede aprendeu:** com 2 neurônios ocultos, cada um vira uma **reta** no plano de entrada: um "detecta" a entrada (1,0) e o outro a (0,1). A camada oculta leva (0,0) e (1,1) para praticamente o mesmo ponto, e (0,1) e (1,0) para cantos opostos. Nesse novo espaço, **uma reta separa as classes**, e o neurônio de saída, que é um Perceptron com sigmoid, resolve. No espaço original, a combinação das duas retas forma uma **faixa diagonal**, a fronteira não linear do XOR.

## Erros comuns

| Erro | Correção |
|---|---|
| Iniciar todos os pesos com zero | pesos aleatórios: com zero, todos os neurônios ocultos aprendem a mesma coisa |
| Esquecer a derivada da sigmoid na camada oculta | `dZ1 = dA1 * A1 * (1 - A1)` |
| Usar `@` em vez de `*` no produto elemento a elemento | `@` multiplica matrizes; `*` multiplica posição a posição |
| Atualizar um parâmetro antes de calcular todos os gradientes | calcular todos os gradientes primeiro, depois atualizar |
| `log(0)` na perda | `np.clip(A, 1e-12, 1 - 1e-12)` |
| Comparar rótulos −1/+1 do Perceptron com 0/1 da MLP | converter a representação antes |

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado.py` | as 20 etapas da aula: Perceptron no OR e MLP no XOR |
| `desafio.ipynb` | desafio completo, executado, com explicação matemática e conclusão |

---

**← Anterior:** [Aula 6.11](../Aula6.11/README.md) · **Voltar:** [Aula 6](../README.md) · **Próxima →** [Aula 7](../../Aula7/README.md)
