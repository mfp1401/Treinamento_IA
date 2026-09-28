# ============================================================
# Aula 5.4 — Feature Engineering | Notebook completo (12 passos)
# Titanic + scikit-learn
# ============================================================

import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

# ---------- Passos 1 a 4 · importar, separar, criar, dividir ----------
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)                                   # 1. importar

y = df["Survived"]                                      # 2. separar X e y
X = df.drop(columns=["Survived", "PassengerId", "Ticket"])

# 3. criar features -> vira o transformer abaixo (entra no Pipeline)

X_train, X_test, y_train, y_test = train_test_split(    # 4. dividir antes de qualquer fit
    X, y, test_size=0.2, stratify=y, random_state=42)

cv = StratifiedKFold(5, shuffle=True, random_state=42)  # validacao cruzada usada nos passos 9-11


# ---------- Passos 5 a 8 · transformar, codificar, escalonar, treinar ----------
class Features(BaseEstimator, TransformerMixin):        # 5. transformacao
    def __init__(self, usar=()): self.usar = usar
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["FamilyType"] = pd.cut(X["FamilySize"],bins=[1, 4, 7, np.inf],labels=["Small", "Medium", "Large"]).astype("object")
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        X["FarePerPerson"] = X["Fare"] / X["FamilySize"]
        X["Deck"] = X["Cabin"].str[0].fillna("Unknown")
        X["AgeGroup"] = pd.cut(X["Age"],bins=[-np.inf, 12, 17, 59, np.inf],labels=["Child", "Teen", "Adult", "Senior"]).astype("object").fillna("Unknown")
        X["Title"] = (X["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip().replace(["Mlle", "Ms"], "Miss").replace("Mme", "Mrs"))
        X["Title"] = X["Title"].where(X["Title"].isin(["Mr", "Miss", "Mrs", "Master"]), "Rare")
        return X.drop(columns=["Name", "Cabin"])
        


def make_model(num_cols, cat_cols):
    num = Pipeline([("imp", SimpleImputer(strategy="median")),          # 7. scaling
                    ("sc",  RobustScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),   # 6. encoding
                    ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10))])
    pre = ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])
    return Pipeline([("feat", Features()), ("pre", pre),
                     ("clf", LogisticRegression(max_iter=1000))])      # 8. treinar


# ---------- Passos 9 a 12 · medir, adicionar, comparar, interpretar ----------
NUM = ["Age", "SibSp", "Parch", "Fare"]                 # 9. baseline
CAT = ["Pclass", "Sex", "Embarked"]

s = cross_val_score(make_model(NUM, CAT), X_train, y_train, cv=cv)
ref = s.mean()
print(f"baseline  {ref:.3f} +/- {s.std():.3f}")

LIMIAR = 0.002                                          # declarado ANTES de ver os numeros
candidatas = [("FamilySize", "num"), ("IsAlone", "num"),
              ("FarePerPerson", "num"), ("Title", "cat"),
              ("Deck", "cat"),("AgeGroup", "cat"),
               ("FamilyType", "cat")]

num, cat = list(NUM), list(CAT)
for nome, tipo in candidatas:                           # 10-11. adicionar e comparar
    n2, c2 = (num + [nome], cat) if tipo == "num" else (num, cat + [nome])
    s = cross_val_score(make_model(n2, c2), X_train, y_train, cv=cv)
    ganho = s.mean() - ref
    print(f"{nome:>14}  {s.mean():.3f} +/- {s.std():.3f}  ganho {ganho:+.3f}")
    if ganho > LIMIAR:
        num, cat, ref = n2, c2, s.mean()                # 12. interpretar e decidir

print("\nfeatures mantidas:", num, cat)

# Teste: usado UMA vez, no final, com o conjunto escolhido
final = make_model(num, cat).fit(X_train, y_train)
print(f"acuracia no teste: {final.score(X_test, y_test):.3f}")

# ============================================================
# INSIGHTS FINAIS
# ============================================================
#
# O modelo baseline apresentou acurácia média de 80,2% na
# validação cruzada, sendo utilizado como referência para
# avaliar as novas features.
#
# Foram testadas sete features de engenharia:
# FamilySize, IsAlone, FarePerPerson, Title, Deck, AgeGroup
# e FamilyType.
#
# Title apresentou o maior ganho individual (+0,017),
# aumentando a média da validação cruzada de 80,2% para 81,9%.
# Essa feature extrai o título do passageiro a partir do seu nome
# e foi mantida no modelo.
#
# Deck também apresentou ganho relevante (+0,007) quando
# adicionado ao modelo que já possuía Title, aumentando a média
# da validação cruzada de 81,9% para 82,6%. Por isso, também
# foi mantida.
#
# FamilySize e FarePerPerson não apresentaram ganho relevante,
# enquanto IsAlone reduziu o desempenho da validação cruzada.
#
# AgeGroup e FamilyType apresentaram pequenas melhorias de
# +0,001, porém não ultrapassaram o limiar de 0,002 definido
# para a seleção das features e, portanto, não foram mantidas.
#
# As features finais selecionadas foram:
# Numéricas: Age, SibSp, Parch, Fare
# Categóricas: Pclass, Sex, Embarked, Title, Deck
#
# Apesar da melhoria observada na validação cruzada com a
# inclusão de Title e Deck, a acurácia no conjunto de teste
# permaneceu em 83,8%, igual ao resultado anterior.
# ============================================================