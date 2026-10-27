import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score,  confusion_matrix)
from sklearn.dummy import DummyClassifier

iris = load_iris()

y = iris.target
x = iris.data

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['especies'] = iris.target_names[iris.target]
df.groupby('especies').mean().round(2)

# Split dos dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.6, stratify=y, random_state=42)

# KNN
modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(x_train, y_train)

# Prever o modelo
y_pred = modelo.predict(x_test)

# Comparar o modelo com o DummyClassifier

base = DummyClassifier(strategy='most_frequent')
base.fit(x_train, y_train)


print(x.shape, y.shape)

print(df.groupby('especies').mean().round(2))

print(x_train.shape, x_test.shape)

print(modelo)

print(y_pred[:8])
print(y_test[:8])

# Avaliar o modelo
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# comparando os modelos
print(base.score(x_test, y_test))
