import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
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


#Carregando dataset, escolhendo colunas, convertendo a coluna survived para int e verificando o shape do dataset

df = fetch_openml("titanic", version=1, as_frame=True).frame
df = df[["survived", "pclass", "sex", "age",
         "sibsp", "parch", "fare", "embarked"]]
df["survived"] = df["survived"].astype(int)
df.shape

#Dividindo X e y

X = df.drop(columns="survived")
y = df["survived"]

#Dividindo os dados em treino e teste

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

#Verificando valores ausentes 

X_train.isna().sum().sort_values(ascending=False)
X_test.isna().sum().sort_values(ascending=False)

#Separando colunas numéricas e categóricas

num_cols = ["age", "fare", "sibsp", "parch"]
cat_cols = ["sex", "embarked", "pclass"]

# As colunas não batem"
assert set(num_cols + cat_cols) == set(X_train.columns)

#Pipeline numérico

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

#Pipeline categórico

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

#Combining the pipelines into a ColumnTransformer

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
])


#Testando o shape do dataset após o pré-processamento

preprocessor.fit_transform(X_train).shape

#Criando o pipeline ccompleto com pré-processamento e classificador

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

#Calculando a acurácia com validação cruzada

scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
pipeline.fit(X_train, y_train)
pipeline.score(X_test, y_test)

#pegando os nomes das features e os coeficientes do modelo

names = pipeline.named_steps["preprocessor"].get_feature_names_out()
coefs = pipeline.named_steps["classifier"].coef_[0]

#Colocando os coeficientes em um pandas Series para melhor visualização

pd.Series(coefs, index=names).sort_values(ascending=False)

#Encontra os melhores parâmetros

grid = {
    "preprocessor__num__imputer__strategy": ["median", "mean"],
    "preprocessor__cat__imputer__strategy": ["most_frequent", "constant"],
    "classifier__C": [0.1, 1.0, 10.0],
}

gs = GridSearchCV(pipeline, grid, cv=5, scoring="accuracy", n_jobs=-1)
gs.fit(X_train, y_train)
gs.best_params_, round(gs.best_score_, 4)

log_fare = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  
    ("log", FunctionTransformer(np.log1p, validate=True)),
    ("scaler", StandardScaler())
])

#Transformador personalizado 

class TamanhoFamilia(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
    
        familia = X["sibsp"] + X["parch"]
        return familia.to_frame("tamanho_familia")
    def get_features_names_out(self, input_features=None):
        return np.array(["tamanho_familia"])

#Salvando o modelo 

joblib.dump(pipeline,"titanic_pipeline.joblib")
modelo = joblib.load("titanic_pipeline.joblib")

#Criando novo passageiro 

nova = pd.DataFrame([{"pclass": 3, "sex": "male", "age": None,
                      "sibsp":0,"parch":0, "fare":7.25, "embarked": "S"}])

#Probabilidade do novo passageiro sobreviver 

modelo.predict_probabilidade(nova)[0,1].round(3)


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
print(log_fare)
print(modelo.predict_probabilidade(nova)[0,1].round(3))