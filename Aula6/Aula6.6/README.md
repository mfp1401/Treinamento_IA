# Aula 6.6 — Validação cruzada

Uma única divisão treino/teste pode dar sorte ou azar. A **validação cruzada** divide os dados em K partes (folds), treina K vezes e usa cada parte uma vez como validação. O resultado é uma média e um desvio-padrão, bem mais confiáveis.

## Conteúdos

- K-Fold: geradores (`yield`) de índices de treino e validação
- Embaralhamento (`shuffle`) e semente (`random_state`)
- Stratified K-Fold: cada fold mantém a proporção das classes
- `cross_val_score` com um modelo novo a cada rodada
- Média e desvio-padrão (`ddof=0`)

## Desafio de casa: o próprio `cross_val_score`

Funções `meu_kfold`, `meu_stratified_kfold` e `meu_cross_val_score`, testadas no `load_breast_cancer` (569 amostras: 212 malignos e 357 benignos).

| Divisão | Acurácia (média ± desvio) |
|---|---|
| K-Fold sem shuffle | 0,884 ± 0,084 |
| Stratified, shuffle (42) | 0,889 ± 0,027 |

Sem embaralhar, os folds ficam com proporções de classe muito diferentes, e o desvio é cerca de 3 vezes maior. Os índices do `meu_kfold` são iguais aos do `KFold`, e o `cross_val_score` do scikit-learn, rodando com os nossos folds, dá scores idênticos. Todos os `assert` dos critérios passam.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado.py` | as 12 etapas da aula (dataset 90/10, visualização dos folds, comparação com o scikit-learn) |
| `desafio.py` | as três funções no `load_breast_cancer`, gráfico, comparação e `assert`s |

---

**← Anterior:** [Aula 6.5](../Aula6.5/README.md) · **Voltar:** [Aula 6](../README.md) · **Próxima →** [Aula 6.7](../Aula6.7/README.md)
