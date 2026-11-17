# Aula 8.2 — Exercício Guiado: LazyPredict


import sys
import inspect
from importlib.metadata import version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, roc_auc_score
)

from lazypredict.Supervised import LazyClassifier


# =============================================================
# 2. Imports e versões
# =============================================================
print("Python:", sys.version.split()[0])
for pacote in ["numpy", "pandas", "scikit-learn", "lazypredict"]:
    try:
        print(f"{pacote:13} {version(pacote)}")
    except Exception as erro:
        print(f"{pacote:13} Não foi possível consultar: {erro}")


# =============================================================
# 3. Carregar o dataset
# =============================================================
dados = load_breast_cancer()
X = dados.data
y = dados.target

print("Formato de X:", X.shape)
print("Formato de y:", y.shape)
print("Nomes das classes:", dados.target_names)
print("Quantidade de características:", len(dados.feature_names))


# =============================================================
# 4. Explorar os dados
# =============================================================
df = pd.DataFrame(X, columns=dados.feature_names)
df["target"] = y
print(df.head())

# 4.1 Estatísticas descritivas
print(df.describe().T)

# 4.2 Distribuição das classes
contagem = pd.Series(y).value_counts().sort_index()
rotulos = [dados.target_names[i] for i in contagem.index]

plt.figure(figsize=(7, 4))
plt.bar(rotulos, contagem.values)
plt.title("Distribuição das classes")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.grid(axis="y", alpha=0.25)
plt.show()

print(pd.DataFrame({"Classe": rotulos, "Quantidade": contagem.values}))


# =============================================================
# 5. Separar treinamento (60%), validação (20%) e teste (20%)
# ====================================