# Aula 6.8 — Naive Bayes do zero

O Naive Bayes classifica textos **contando palavras**: estima a probabilidade de cada palavra aparecer em cada classe (spam ou normal) e combina essas probabilidades com o teorema de Bayes, supondo que as palavras são independentes.

## Conteúdos

- Tokenização, vocabulário e vetores de contagem
- Prioris P(classe)
- Frequência das palavras por classe
- Suavização de Laplace (α = 1)
- Log-probabilidades para evitar números minúsculos
- `predict_proba` com normalização estável

## Desafio de casa: filtro de spam

Classe `NaiveBayesDoZero` treinada em 48 mensagens em português (24 spam e 24 normais).

| Resultado | Valor |
|---|---|
| Teste do núcleo: P(spam \| "grátis promoção") | 0,8815 ✓ |
| Acurácia no teste (12 mensagens) | 0,917 |
| Matriz de confusão | `[[5 1] [0 6]]`: todos os spams detectados |
| Probabilidades × `MultinomialNB` | idênticas |

**Análise do erro:** "Chego em casa por volta das sete" foi classificada como spam porque palavras comuns como "em", "casa" e "por" apareceram mais nos spams do treino. O modelo não entende o sentido da frase, só conta palavras.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado.py` | Naive Bayes passo a passo no mini dataset da aula |
| `desafio.py` | filtro de spam completo, mensagens novas e explicação das previsões |

---

**← Anterior:** [Aula 6.7](../Aula6.7/README.md) · **Voltar:** [Aula 6](../README.md) · **Próxima →** [Aula 6.9](../Aula6.9/README.md)
