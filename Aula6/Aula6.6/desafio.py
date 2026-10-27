import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"   

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score   
from sklearn.neighbors import NearestCentroid                

# Dados: 569 amostras, classe 0 = 212, classe 1 = 357
X, y = load_breast_cancer(return_X_y=True)
n = len(y)
print("contagens:", np.bincount(y))


# Parte 1 · K-Fold
def meu_kfold(n_amostras, k=5, shuffle=False, random_state=None):
    if not 2 <= k <= n_amostras:
        raise ValueError("k precisa estar entre 2 e n_amostras")
    indices = np.arange(n_amostras)
    if shuffle:
        indices = np.random.default_rng(random_state).permutation(indices)
    tamanhos = np.full(k, n_amostras // k)
    tamanhos[: n_amostras % k] += 1
    inicio = 0
    for tamanho in tamanhos:
        fim = inicio + tamanho
        val_idx = indices[inicio:fim]
        train_idx = np.concatenate([indices[:inicio], indices[fim:]])
        yield train_idx, val_idx
        inicio = fim


# Parte 2 · Stratified K-Fold
def meu_stratified_kfold(y, k=5, shuffle=False, random_state=None):
    rng = np.random.default_rng(random_state)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx_c = np.flatnonzero(y == c)            
        if shuffle:
            idx_c = rng.permutation(idx_c)
        for f, parte in enumerate(np.array_split(idx_c, k)):
            folds[f].append(parte)                
    for partes in folds:
        val_idx = np.sort(np.concatenate(partes))
        train_idx = np.setdiff1d(np.arange(len(y)), val_idx)
        yield train_idx, val_idx


# Parte 3 · Cross-val-score
def meu_cross_val_score(modelo_factory, X, y, folds, metrica):
    scores = []
    for train_idx, val_idx in folds:
        modelo = modelo_factory()                 
        modelo.fit(X[train_idx], y[train_idx])
        y_pred = modelo.predict(X[val_idx])
        scores.append(metrica(y[val_idx], y_pred))
    return np.array(scores)



class CentroideMaisProximo:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroides_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        dist = np.stack([np.linalg.norm(X - c, axis=1) for c in self.centroides_], axis=1)
        return self.classes_[np.argmin(dist, axis=1)]


def acuracia(y_true, y_pred):
    return np.mean(y_true == y_pred)


# Índices dos folds e distribuição de classes
for fold, (t, v) in enumerate(meu_stratified_kfold(y, 5, True, 42), start=1):
    print(f"Fold {fold} | treino {len(t)} | val {len(v)} | classes na val {np.bincount(y[v])}")

# Scores, média e desvio-padrão 
sc_kfold = meu_cross_val_score(CentroideMaisProximo, X, y, meu_kfold(n, 5), acuracia)
sc_strat = meu_cross_val_score(CentroideMaisProximo, X, y, meu_stratified_kfold(y, 5, True, 42), acuracia)
print("K-Fold sem shuffle:", sc_kfold, f"{sc_kfold.mean():.3f} ± {sc_kfold.std():.3f}")
print("Stratified (42):   ", sc_strat, f"{sc_strat.mean():.3f} ± {sc_strat.std():.3f}")

# Gráfico dos scores
plt.plot(range(1, 6), sc_kfold, "o-", label="K-Fold sem shuffle")
plt.plot(range(1, 6), sc_strat, "o-", label="Stratified, shuffle (42)")
plt.xlabel("fold"); plt.ylabel("accuracy"); plt.legend(); plt.show()

# Comparação com o scikit-learn
print("meu_kfold == KFold:", all(np.array_equal(a[1], b[1])
      for a, b in zip(meu_kfold(n, 5), KFold(5).split(X))))
sc_sk = cross_val_score(NearestCentroid(), X, y, cv=list(meu_stratified_kfold(y, 5, True, 42)))
print("cross_val_score do sklearn com os meus folds:", sc_sk, "| igual:", np.allclose(sc_sk, sc_strat))

# Verificações (critérios 1, 2, 3, 4, 5 e 7)
vals = np.concatenate([v for _, v in meu_kfold(n, 5, True, 42)])
assert np.array_equal(np.sort(vals), np.arange(n))                       
for t, v in meu_stratified_kfold(y, 5, True, 42):
    assert len(np.intersect1d(t, v)) == 0                                
    assert abs(y[v].mean() - y.mean()) < 0.01                            

modelos = []
def fabrica():
    m = CentroideMaisProximo()
    modelos.append(m)
    return m
meu_cross_val_score(fabrica, X, y, meu_kfold(n, 5), acuracia)
assert len(set(map(id, modelos))) == 5                                   

assert acuracia(np.array([1, 0, 1, 1]), np.array([1, 0, 0, 1])) == 0.75  

a = meu_cross_val_score(CentroideMaisProximo, X, y, meu_stratified_kfold(y, 5, True, 7), acuracia)
b = meu_cross_val_score(CentroideMaisProximo, X, y, meu_stratified_kfold(y, 5, True, 7), acuracia)
assert np.array_equal(a, b)                                              
print("Todos os asserts passaram.")