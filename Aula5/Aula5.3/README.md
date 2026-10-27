# Aula 5.3 — Pipelines, validação cruzada e GridSearchCV

Dados reais têm valores faltando e colunas de texto. Esta aula monta um **Pipeline** que trata tudo isso automaticamente: preenche ausentes, padroniza números, transforma categorias em colunas e treina o modelo, sem vazar informação do teste para o treino.

## Conteúdos

- `SimpleImputer`, `StandardScaler` e `OneHotEncoder`
- `ColumnTransformer` para tratar colunas numéricas e categóricas separadamente
- `Pipeline` com pré-processamento + `LogisticRegression`
- Validação cruzada com `cross_val_score`
- Interpretação dos coeficientes da regressão logística
- Ajuste de hiperparâmetros com `GridSearchCV`
- Transformador personalizado (`BaseEstimator`, `TransformerMixin`)
- Salvar e carregar o modelo com `joblib`
- Prever a probabilidade para um exemplo novo (`predict_proba`)

## Arquivos

| Arquivo | O que faz |
|---|---|
| `exercicio_guiado_titanic.py` | pipeline completo no Titanic, com o transformador `TamanhoFamilia` |
| `exercicio1_adult_income.py` | o mesmo fluxo no Adult Income (renda acima de 50 mil/ano), com o transformador `RendaPorHora` |
| `titanic_pipeline.joblib` | modelo do Titanic salvo |
| `adult_income_pipeline.joblib` | modelo do Adult Income salvo |

---

**← Anterior:** [Aula 5.1](../Aula5.1/README.md) · **Voltar:** [Aula 5](../README.md) · **Próxima →** [Aula 5.4](../Aula5.4/README.md)
