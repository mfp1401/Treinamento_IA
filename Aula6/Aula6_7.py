import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn.model_selection import train_test_split


# =============================================================
# DADOS
# =============================================================

# 01 · Criar o dataset 
rng = np.random.default_rng(42)
X0 = rng.normal(loc=[3, 3], scale=1.3, size=(40, 2))   # classe 0
X1 = rng.normal(loc=[5, 5], scale=1.3, size=(40, 2))   # classe 1

# 02 · Visualizar os pontos
plt.scatter(X0[:, 0], X0[:, 1], label="classe 0")
plt.scatter(X1[:, 0], X1[:, 1], marker="^", label="classe 1")
plt.xlabel("x1"); plt.ylabel("x2"); plt.legend(); plt.title("Etapa 02 · dataset")
plt.show()

# 03 · Identificar X e y
X = np.vstack([X0, X1])                                  
y = np.concatenate([np.zeros(40, dtype=int), np.ones(40, dtype=int)])  
print("03 · X:", X.shape, "| y:", y.shape)

# 04 · Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
print("04 · treino:", X_train.shape, "| teste:", X_test.shape)


# =============================================================
# DISTÂNCIAS
# =============================================================

# 05 · Euclidiana à mão: A = (1, 2), B = (4, 6)
a = np.array([1, 2])
b = np.array([4, 6])
print("\n05 · euclidiana:", np.sqrt(np.sum((a - b) ** 2)))   

# 06 · Manhattan à mão
print("06 · manhattan:", np.sum(np.abs(a - b)))              

# 07 · Função de distâncias (uma amostra contra todo o treino)
def distancias_loop(X_train, x):
    distancias = np.zeros(len(X_train))
    for i in range(len(X_train)):
        diff = X_train[i] - x
        distancias[i] = np.sqrt(np.sum(diff ** 2))
    return distancias

d = distancias_loop(X_train, X_test[0])
print("07 · distâncias da 1ª amostra de teste:", d[:5].round(3), "...")


# =============================================================
# KNN
# =============================================================

# 08 · Seleção dos vizinhos (os k menores)
k = 3
indices = np.argsort(d, kind="stable")[:k]
vizinhos = y_train[indices]
print("\n08 · índices:", indices, "| rótulos dos vizinhos:", vizinhos)

# 09 · Votação
votos = np.bincount(vizinhos)
classe = votos.argmax()
print("09 · votos:", votos, "| classe prevista:", classe, "| real:", y_test[0])

# 10 · KNN com loops
def predict_loop(X_train, y_train, X_test, k=3):
    previsoes = []

    for x in X_test:                                   
        distancias = np.zeros(len(X_train))
        for i in range(len(X_train)):                  
            dif = X_train[i] - x
            distancias[i] = np.sqrt(np.sum(dif ** 2))
        indices = np.argsort(distancias, kind="stable")[:k]
        votos = np.bincount(y_train[indices])
        previsoes.append(votos.argmax())
    return np.array(previsoes)

# 11 · KNN com Broadcasting 
class KNNDoZero:
    def __init__(self, k=3, metrica="euclidiana"):
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError("k deve ser inteiro positivo")
        if metrica not in ("euclidiana", "manhattan"):
            raise ValueError("métrica inválida")
        self.k, self.metrica = k, metrica

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        if self.k > len(self.X_train):
            raise ValueError("k > n_train")
        self.classes, self.y_cod = np.unique(np.asarray(y), return_inverse=True)
        return self

    def _distancias(self, X):
        dif = X[:, None, :] - self.X_train[None, :, :]
        if self.metrica == "euclidiana":
            return np.sqrt(np.sum(dif ** 2, axis=2))
        return np.sum(np.abs(dif), axis=2)

    def predict(self, X):
        d = self._distancias(np.asarray(X, dtype=float))
        indices = np.argsort(d, axis=1, kind="stable")
        indices = indices[:, :self.k]
        viz = self.y_cod[indices]
        venc = [np.bincount(v).argmax() for v in viz]
        return self.classes[venc]

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))

# 12 · Comparar previsões

prev_loop = predict_loop(X_train, y_train, X_test, k=3)
prev_broad = KNNDoZero(k=3).fit(X_train, y_train).predict(X_test)
print("\n12 · loops:      ", prev_loop)
print("12 · broadcasting:", prev_broad)
print("12 · previsões iguais?", np.array_equal(prev_loop, prev_broad))


# =============================================================
# AVALIAÇÃO
# =============================================================

# 13 · Medir o tempo (distâncias com loops vs broadcasting)
def distancias_loops(X_test, X_train):
    n_test, n_train = len(X_test), len(X_train)
    D = np.zeros((n_test, n_train))
    for i in range(n_test):
        for j in range(n_train):
            D[i, j] = np.sqrt(np.sum((X_test[i] - X_train[j]) ** 2))
    return D

def distancias_broadcasting(X_test, X_train):
    dif = X_test[:, None, :] - X_train[None, :, :]
    return np.sqrt(np.sum(dif ** 2, axis=2))

def medir(funcao, *args, repeticoes=7):
    tempos = []
    for _ in range(repeticoes):
        inicio = perf_counter()
        funcao(*args)
        fim = perf_counter()
        tempos.append(fim - inicio)
    return np.median(tempos)

# verificação: mesmas distâncias e mesmos vizinhos
D_a = distancias_loops(X_test, X_train)
D_b = distancias_broadcasting(X_test, X_train)
ia = np.argsort(D_a, axis=1, kind="stable")[:, :k]
ib = np.argsort(D_b, axis=1, kind="stable")[:, :k]
print("\n13 · distâncias iguais?", np.allclose(D_a, D_b), "| mesmos vizinhos?", np.array_equal(ia, ib))

# benchmark com dados maiores (n_test = 200)
X_test_b = rng.normal(size=(200, 2))
tamanhos = [200, 1000, 5000]
t_loops, t_broad = [], []
for n_train in tamanhos:
    X_train_b = rng.normal(size=(n_train, 2))
    tl = medir(distancias_loops, X_test_b, X_train_b, repeticoes=3)
    tb = medir(distancias_broadcasting, X_test_b, X_train_b, repeticoes=3)
    t_loops.append(tl); t_broad.append(tb)
    print(f"13 · n_train={n_train:5d} | loops {tl:.4f} s | broad {tb:.4f} s | speedup {tl / tb:.0f}x")

pos = np.arange(len(tamanhos))
plt.bar(pos - 0.2, t_loops, width=0.4, label="loops")
plt.bar(pos + 0.2, t_broad, width=0.4, label="broadcasting")
plt.xticks(pos, tamanhos); plt.yscale("log")
plt.xlabel("n_train"); plt.ylabel("tempo (s)"); plt.legend()
plt.title("Etapa 13 · tempo das distâncias (n_test = 200)")
plt.show()

# 14 · Escalonar 
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
X_train_esc = (X_train - mu) / sigma
X_test_esc = (X_test - mu) / sigma
print("\n14 · média do treino escalonado:", X_train_esc.mean(axis=0).round(3),
      "| desvio:", X_train_esc.std(axis=0).round(3))

# 15 · Acurácia no teste
for kk in [1, 3, 9]:
    m = KNNDoZero(k=kk).fit(X_train_esc, y_train)
    print(f"15 · k={kk} | treino {m.score(X_train_esc, y_train):.3f} | teste {m.score(X_test_esc, y_test):.3f}")

# 16 · Fronteiras de decisão
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 144),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 136))
grade = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, kk in zip(axes, [1, 3, 9]):
    m = KNNDoZero(k=kk).fit(X_train, y_train)
    Z = m.predict(grade).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlGn_r")
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], label="classe 0")
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], marker="^", label="classe 1")
    ax.set_title(f"K = {kk} · treino: {m.score(X_train, y_train):.1%}")
axes[0].legend()
plt.suptitle("Etapa 16 · fronteiras de decisão")
plt.tight_layout(); plt.show()