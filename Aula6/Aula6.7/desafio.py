import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


# =============================================================
# Votação (usada pelas duas versões)
# Regra de empate explícita e determinística:
#  - distâncias iguais: argsort "stable" mantém o menor índice do treino
#  - votos iguais: bincount + argmax escolhe a MENOR classe
# =============================================================
def votar(rotulos_vizinhos):
    votos = np.bincount(rotulos_vizinhos)
    return votos.argmax()


# =============================================================
# Versão 1 · Loops (padrão: euclidiana)
# =============================================================
class KNNLoop:
    def __init__(self, k=3, metrica="euclidiana"):
        if k < 1:
            raise ValueError("k deve ser inteiro positivo")
        if metrica not in ("euclidiana", "manhattan"):
            raise ValueError("métrica inválida")
        self.k, self.metrica = k, metrica

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def distancias(self, X):
        n_test, n_train = len(X), len(self.X_train)
        D = np.zeros((n_test, n_train))
        for i in range(n_test):                    
            for j in range(n_train):                
                dif = X[i] - self.X_train[j]
                if self.metrica == "euclidiana":
                    D[i, j] = np.sqrt(np.sum(dif ** 2))
                else:
                    D[i, j] = np.sum(np.abs(dif))
        return D

    def predict(self, X):
        D = self.distancias(np.asarray(X, dtype=float))
        previsoes = []
        for d in D:
            indices = np.argsort(d, kind="stable")[:self.k]   # k menores distâncias
            previsoes.append(votar(self.y_train[indices]))
        return np.array(previsoes)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# =============================================================
# Versão 2 · Broadcasting (padrão: Manhattan)
# =============================================================
class KNNBroadcasting:
    def __init__(self, k=3, metrica="manhattan"):
        if k < 1:
            raise ValueError("k deve ser inteiro positivo")
        if metrica not in ("euclidiana", "manhattan"):
            raise ValueError("métrica inválida")
        self.k, self.metrica = k, metrica

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def distancias(self, X):
        dif = X[:, None, :] - self.X_train[None, :, :]   #
        if self.metrica == "euclidiana":
            return np.sqrt(np.sum(dif ** 2, axis=2))
        return np.sum(np.abs(dif), axis=2)

    def predict(self, X):
        D = self.distancias(np.asarray(X, dtype=float))
        indices = np.argsort(D, axis=1, kind="stable")[:, :self.k]
        return np.array([votar(v) for v in self.y_train[indices]])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# =============================================================
# Dados 
# =============================================================
rng = np.random.default_rng(42)
X0 = rng.normal(loc=[3, 3], scale=1.3, size=(40, 2))
X1 = rng.normal(loc=[5, 5], scale=1.3, size=(40, 2))
X = np.vstack([X0, X1])
y = np.concatenate([np.zeros(40, dtype=int), np.ones(40, dtype=int)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# =============================================================
# Verificações: distâncias, vizinhos e empates
# =============================================================
A = np.array([[1.0, 2.0]])
B = np.array([[4.0, 6.0]])
print("Euclidiana A-B:", KNNLoop(1, "euclidiana").fit(B, [0]).distancias(A)[0, 0])        # 5.0
print("Manhattan  A-B:", KNNBroadcasting(1, "manhattan").fit(B, [0]).distancias(A)[0, 0]) # 7.0

# vizinhos = menores distâncias
m = KNNBroadcasting(k=3, metrica="manhattan").fit(X_train, y_train)
D = m.distancias(X_test)
idx = np.argsort(D, axis=1, kind="stable")
for linha, ordem in zip(D, idx):
    assert linha[ordem[:3]].max() <= linha[ordem[3:]].min()
print("Vizinhos = menores distâncias: OK")

# empate de votos: k = 2 com um vizinho de cada classe -> vence a menor classe (0)
X_emp = np.array([[0.0], [2.0]])
y_emp = np.array([1, 0])
print("Empate (votos [1, 1]) ->", KNNLoop(k=2).fit(X_emp, y_emp).predict(np.array([[1.0]])))


# =============================================================
# Entrega 2 · prova de previsões equivalentes
# =============================================================
v1 = KNNLoop(k=3).fit(X_train, y_train)            # versão 1: loops + euclidiana
v2 = KNNBroadcasting(k=3).fit(X_train, y_train)    # versão 2: broadcasting + Manhattan
p1, p2 = v1.predict(X_test), v2.predict(X_test)
print("\nV1 (loops, euclidiana):      ", p1, "| acurácia", v1.score(X_test, y_test))
print("V2 (broadcasting, manhattan):", p2, "| acurácia", v2.score(X_test, y_test))
print("Previsões diferentes entre V1 e V2:", np.sum(p1 != p2))

for metrica in ["euclidiana", "manhattan"]:
    pl = KNNLoop(3, metrica).fit(X_train, y_train).predict(X_test)
    pb = KNNBroadcasting(3, metrica).fit(X_train, y_train).predict(X_test)
    dl = KNNLoop(3, metrica).fit(X_train, y_train).distancias(X_test)
    db = KNNBroadcasting(3, metrica).fit(X_train, y_train).distancias(X_test)
    print(f"Mesma métrica ({metrica}): distâncias iguais {np.allclose(dl, db)} | previsões iguais {np.array_equal(pl, pb)}")


# =============================================================
# Entregas 3 e 4 · tempos com perf_counter e gráfico
# =============================================================
def medir(funcao, *args, repeticoes=3):
    tempos = []
    for _ in range(repeticoes):
        inicio = perf_counter()
        funcao(*args)
        tempos.append(perf_counter() - inicio)
    return np.median(tempos)

X_test_b = rng.normal(size=(200, 2))        
tamanhos = [200, 1000, 5000]
tempos = {"V1 loops · euclidiana": [], "V2 broadcasting · euclidiana": [], "V2 broadcasting · manhattan": []}
for n_train in tamanhos:
    X_tr = rng.normal(size=(n_train, 2))
    y_tr = rng.integers(0, 2, n_train)
    tempos["V1 loops · euclidiana"].append(medir(KNNLoop(3, "euclidiana").fit(X_tr, y_tr).predict, X_test_b))
    tempos["V2 broadcasting · euclidiana"].append(medir(KNNBroadcasting(3, "euclidiana").fit(X_tr, y_tr).predict, X_test_b))
    tempos["V2 broadcasting · manhattan"].append(medir(KNNBroadcasting(3, "manhattan").fit(X_tr, y_tr).predict, X_test_b))

print("\nTempos de predict (n_test = 200), mediana de 3 repetições:")
for i, n_train in enumerate(tamanhos):
    tl = tempos["V1 loops · euclidiana"][i]
    tb = tempos["V2 broadcasting · euclidiana"][i]
    tm = tempos["V2 broadcasting · manhattan"][i]
    print(f"n_train={n_train:5d} | V1 {tl:.4f} s | V2 euclid {tb:.4f} s | V2 manhattan {tm:.4f} s | speedup (mesma métrica) {tl / tb:.0f}x")

pos = np.arange(len(tamanhos))
for j, (nome, t) in enumerate(tempos.items()):
    plt.bar(pos + (j - 1) * 0.27, t, width=0.27, label=nome)
plt.xticks(pos, tamanhos); plt.yscale("log")
plt.xlabel("n_train"); plt.ylabel("tempo (s, escala log)")
plt.title("Tempo de predict · n_test = 200"); plt.legend(fontsize=8)
plt.tight_layout(); plt.show()


# =============================================================
# Entrega 6 · com e sem escalonamento (30 características)
# =============================================================
Xc, yc = load_breast_cancer(return_X_y=True)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.25, random_state=42, stratify=yc)

mu = Xc_tr.mean(axis=0)          
sigma = Xc_tr.std(axis=0)
Xc_tr_esc = (Xc_tr - mu) / sigma
Xc_te_esc = (Xc_te - mu) / sigma

print("\nload_breast_cancer (30 características):")
for metrica in ["euclidiana", "manhattan"]:
    sem = KNNBroadcasting(5, metrica).fit(Xc_tr, yc_tr).score(Xc_te, yc_te)
    com = KNNBroadcasting(5, metrica).fit(Xc_tr_esc, yc_tr).score(Xc_te_esc, yc_te)
    print(f"{metrica:10s} | sem escalonamento {sem:.3f} | com escalonamento {com:.3f}")


# =============================================================
# Entrega 7 · fronteiras com diferentes K (versão 2)
# =============================================================
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 144),
                     np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 136))
grade = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, k in zip(axes, [1, 3, 9]):
    modelo = KNNBroadcasting(k, "manhattan").fit(X_train, y_train)
    Z = modelo.predict(grade).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdYlGn_r")
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], label="classe 0")
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], marker="^", label="classe 1")
    ax.set_title(f"K = {k} · treino {modelo.score(X_train, y_train):.1%} · teste {modelo.score(X_test, y_test):.1%}")
axes[0].legend()
plt.suptitle("Fronteiras de decisão · versão 2 (Manhattan)")
plt.tight_layout(); plt.show()