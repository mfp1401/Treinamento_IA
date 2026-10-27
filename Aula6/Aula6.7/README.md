# Aula 6.7 — KNN do zero

O KNN classifica uma amostra nova olhando os **K exemplos de treino mais próximos** e fazendo uma votação. Não há treino propriamente dito: todo o trabalho acontece na previsão, ao calcular as distâncias.

## Conteúdos

- Distância euclidiana e de Manhattan
- Seleção dos K vizinhos com `np.argsort(kind="stable")`
- Votação majoritária com `np.bincount` e regra de empate
- KNN com loops × KNN com broadcasting
- Medição de tempo com `perf_counter`
- Escalonamento sem vazamento (μ e σ só do treino)
- Fronteiras de decisão para diferentes K

## Desafio de casa: loops × broadcasting

- **Versão 1:** loops, distância euclidiana
- **Versão 2:** broadcasting, distância de Manhattan

| Resultado | Valor |
|---|---|
| Previsões V1 × V2 (20 pontos de teste) | idênticas, acurácia 0,85 |
| Com a mesma métrica | distâncias e previsões iguais entre as versões |
| Tempo com 5 000 treinos | cerca de 3 s (loops) × 0,1 s (broadcasting), ~30× mais rápido |
| `load_breast_cancer` sem → com escalonamento | 0,930 → 0,979 (euclidiana) |

**Regra de empate:** em votos iguais, vence a menor classe; em distâncias iguais, vence o menor índice do treino.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado.py` | as 16 etapas da aula |
| `desafio.py` | as duas versões, prova de equivalência, tempos, escalonamento e fronteiras |

---

**← Anterior:** [Aula 6.6](../Aula6.6/README.md) · **Voltar:** [Aula 6](../README.md) · **Próxima →** [Aula 6.8](../Aula6.8/README.md)
