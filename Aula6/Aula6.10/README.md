# Aula 6.10 — K-Means do zero

Até aqui, todos os modelos aprenderam com **rótulos** (`y`). O K-Means é **não supervisionado**: recebe só `X` e procura sozinho grupos de amostras parecidas. Cada grupo é representado por um **centroide**, a média dos seus pontos, e o algoritmo alterna dois passos até os centroides pararem de se mover: atribuir cada amostra ao centroide mais próximo e recalcular os centroides.

```text
Definir K → inicializar centroides → calcular distâncias → atribuir clusters
          → atualizar centroides → verificar convergência → repetir
```

## Conteúdos

- Aprendizado supervisionado × não supervisionado (classificação × clusterização)
- KNN × K-Means: o mesmo "K" com significados diferentes (vizinhos × grupos)
- Centroide como média do grupo: μₖ = (1/|Cₖ|) Σ xᵢ
- Distâncias amostra × centroide com broadcasting: `X[:, None, :] - centroides[None, :, :]`
- Atribuição com `np.argmin(distancias, axis=1)`
- Clusters vazios: o centroide recebe a amostra mais distante, sem gerar `NaN`
- Inércia: J = Σ ‖xᵢ − μ₍ᵢ₎‖², que nunca aumenta de uma iteração para a outra
- Convergência pelo deslocamento máximo dos centroides (`tol`)
- Escalonamento: sem ele, a característica de maior escala domina a distância
- Método do cotovelo para escolher K
- Influência da semente: o K-Means pode parar num mínimo local
- Comparação com o `KMeans` do scikit-learn

## Exercício guiado: 150 clientes (compras por mês × gasto médio)

| Etapas | O que faz | Resultado |
|---|---|---|
| 01–06 | imports, gerador reprodutível, dataset, gráfico, padronização, K = 3 | `X` (150, 2) |
| 07–11 | inicializar, distâncias, atribuir, atualizar, teste de cluster vazio | cluster vazio sem `NaN` |
| 12–13 | laço iterativo e convergência | inércia 223,8 → 54,2 → **21,58**, convergiu em 3 iterações |
| 14–16 | classe `KMeansDoZero` e testes de robustez | K = 1, K > n, pontos repetidos, `max_iter = 1` |
| 17–19 | gráficos dos clusters, histórico dos centroides e inércia de duas formas | grupos de 50, 50 e 50 clientes |
| 20 | método do cotovelo | queda forte até K = 3 |
| 21 | comparar sementes | semente 1 cai num mínimo local (inércia 150,6) |
| 22 | comparar com o scikit-learn | mesma inércia (21,581), concordância de 100% |

Os três grupos encontrados, nas unidades originais:

| Grupo | Compras/mês | Gasto médio | Leitura |
|---|---|---|---|
| 0 | 9,3 | R$ 74,8 | frequentes, ticket baixo |
| 1 | 4,4 | R$ 255,2 | ticket alto |
| 2 | 2,0 | R$ 61,9 | ocasionais, ticket baixo |

## Desafio de casa: segmentação de clientes de uma loja online

Dataset sintético com **200 clientes** em 4 perfis (o algoritmo não sabe qual é qual): visitas ao site por mês × ticket médio por compra (R$).

**Método do cotovelo** (melhor inércia entre 10 sementes):

| K | 1 | 2 | 3 | **4** | 5 | 6 |
|---|---|---|---|---|---|---|
| Inércia | 400,0 | 216,1 | 89,2 | **33,6** | 28,7 | 23,9 |

A queda perde força depois de K = 4, então ficamos com 4 grupos.

**Segmentos encontrados** (do menor para o maior ticket):

| Grupo | Clientes | Visitas/mês | Ticket médio | Segmento |
|---|---|---|---|---|
| 3 | 51 | 3,0 | R$ 83,70 | ocasional econômico |
| 0 | 49 | 14,4 | R$ 97,30 | frequente econômico |
| 2 | 48 | 12,0 | R$ 378,40 | frequente premium |
| 1 | 52 | 4,3 | R$ 422,60 | ocasional premium |

**Análise:**
- **197 de 200 clientes** caíram no grupo do seu perfil verdadeiro. Os 3 "erros" são clientes na fronteira entre perfis vizinhos, como um frequente premium com poucas visitas, que ficou mais perto do grupo ocasional premium.
- 9 das 10 sementes chegaram à mesma solução (inércia 33,595). A semente 9 parou num **mínimo local** (inércia 84,0, grupos de 17, 51, 96 e 36 clientes). Por isso o código treina com várias sementes e fica com a menor inércia, a mesma ideia do `n_init` do scikit-learn.
- A nossa implementação dá a **mesma inércia e os mesmos centroides** do `KMeans` do scikit-learn, com 100% de concordância nos rótulos.
- Sem padronizar, o ticket (centenas de reais) dominaria a distância e as visitas quase não teriam peso.

**Critério de aceite:** o script roda de cima a baixo sem erros, sem `NaN`, e `labels_` e `inercia_` são recalculados com os `centroides_` finais (verificado com `assert`).

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado.py` | as 22 etapas da aula (slides 39 a 62) |
| `desafio.py` | classe `KMeansDoZero`, cotovelo, sementes, gráficos, comparação com o scikit-learn e interpretação dos grupos |

---

**← Anterior:** [Aula 6.9](../Aula6.9/README.md) · **Voltar:** [Aula 6](../README.md) · **Próxima →** [Aula 6.11](../Aula6.11/README.md)
