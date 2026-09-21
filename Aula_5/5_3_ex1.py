import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin


# Carregando dataset, escolhendo colunas, convertendo a coluna income
# e verificando o shape do dataset

df = fetch_openml("adult", version=2, as_frame=True).frame
df = df[["class", "age", "workclass", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
         "capital-gain", "capital-loss", "hours-per-week", "native-country"]]

df["class"] = df["class"].str.strip().map({"<=50K": 0, ">50K": 1})


# Dividindo X e y

X = df.drop(columns="class")
y = df["class"]


# Dividindo os dados em treino e teste

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)


# Verificando valores ausentes

X_train.isna().sum().sort_values(ascending=False)
X_test.isna().sum().sort_values(ascending=False)


# Separando colunas numéricas e categóricas

num_cols = ["age", "education-num", "capital-gain",
            "capital-loss", "hours-per-week"]

cat_cols = ["workclass", "education", "marital-status",
            "occupation", "relationship", "race", "sex", "native-country"]


# Verificando se todas as colunas foram classificadas

assert set(num_cols + cat_cols) == set(X_train.columns)


# Pipeline numérico

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Pipeline categórico

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# Combinando os pipelines em um ColumnTransformer

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
])

# Testando o shape do dataset após o pré-processamento

preprocessor.fit_transform(X_train).shape

# Criando o pipeline completo com pré-processamento e classificador

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# Calculando a acurácia com validação cruzada

scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")

# Treinando o pipeline

pipeline.fit(X_train, y_train)

# Avaliando no conjunto de teste

pipeline.score(X_test, y_test)

# Pegando os nomes das features e os coeficientes do modelo

names = pipeline.named_steps["preprocessor"].get_feature_names_out()
coefs = pipeline.named_steps["classifier"].coef_[0]

# Colocando os coeficientes em um pandas Series

pd.Series(coefs, index=names).sort_values(ascending=False)


# Encontrando os melhores parâmetros

grid = {
    "preprocessor__num__imputer__strategy": ["median", "mean"],
    "preprocessor__cat__imputer__strategy": ["most_frequent", "constant"],
    "classifier__C": [0.1, 1.0, 10.0],
}


gs = GridSearchCV(pipeline, grid, cv=5, scoring="accuracy", n_jobs=-1)
gs.fit(X_train, y_train)


gs.best_params_, round(gs.best_score_, 4)

# Transformação logarítmica para capital-gain

log_capital_gain = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(np.log1p, validate=True)),
    ("scaler", StandardScaler())
])

# Transformador personalizado


class RendaPorHora(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        renda_hora = (
            X["capital-gain"] /
            (X["hours-per-week"] * 52 + 1)
        )

        return renda_hora.to_frame("renda_por_hora")

    def get_feature_names_out(self, input_features=None):

        return np.array(["renda_por_hora"])

# Salvando o modelo


joblib.dump(pipeline, "adult_income_pipeline.joblib")

# Carregando o modelo

modelo = joblib.load("adult_income_pipeline.joblib")

# Criando uma nova pessoa

nova = pd.DataFrame([
    {
        "age": 35,
        "workclass": "Private",
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Exec-managerial",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
])

# Probabilidade da nova pessoa ganhar mais de US$50K

modelo.predict_proba(nova)[0, 1].round(3)


print(df.shape)

print("Split dos dados em treino e teste")

print(X_train.isna().sum().sort_values(ascending=False))

print(X_test.isna().sum().sort_values(ascending=False))

print(set(num_cols + cat_cols) == set(X_train.columns))

print(num_pipeline)

print(cat_pipeline)

print(preprocessor.fit_transform(X_train).shape)

print(scores.mean().round(4), scores.std().round(4))

print(pipeline.score(X_test, y_test))

print(pd.Series(coefs, index=names).sort_values(ascending=False))

print(gs.best_params_)

print(round(gs.best_score_, 4))

print(log_capital_gain)

print(modelo.predict_proba(nova)[0, 1].round(3))

# Insights:
#
# O modelo apresentou uma acurácia média de 85,04% na validação cruzada,
# com desvio padrão de apenas 0,41%. No conjunto de
# teste, a acurácia foi de aproximadamente 85,30%, um resultado bastante
# próximo da validação cruzada.
#
# Um resultado que chamou atenção foi a variável "capital-gain", que
# apresentou o maior coeficiente positivo do modelo (2,2559). Isso indica
# que, dentro do modelo de regressão logística, essa característica possui
# uma forte associação com a classe de renda >50K. Também se destacou o
# estado civil "Married-civ-spouse", com coeficiente 1,6846.
#
# Por outro lado, "Never-married" apresentou um dos maiores coeficientes
# negativos (-1,2112), indicando uma associação negativa com a classe >50K
# no modelo.
#
# Por fim, para uma nova pessoa criada no exemplo, o modelo estimou uma
# probabilidade de aproximadamente 16,9% de pertencer à classe >50K.
