
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans          # só na comparação (item 12)

np.set_printoptions(suppress=True)


# =============================================================
# 01 · Criar um dataset sintético
# 200 clientes de uma loja online, em 4 perfis (o K-Means NÃO vê o perfil):
#   [visitas ao site por mês, ticket médio por compra (R$)]
# =============================================================
rng = np.random.default_rng(2026)
perfis = {
    "ocasional econômico": ([3, 80],   [1.0, 20]),
    "frequente econômico": ([14, 95],  [2.0, 25]),
    "ocasional premium":   ([4, 420],  [1.2, 60]),
    "frequente premium":   ([12, 380], [2.0, 55]),
}
X = np.vstack([rng.normal(media, desvio, size=(50, 2))
               for media, desvio in perfis.values()])
X[:, 0] = np.clip(X[:, 0], 0.5, None)        # visitas não podem ser negativas
perfil_real = np.repeat(np.arange(4), 50)     # guardado só para conferir no final (item 13)
nomes_col = ["Visitas por mês", "Ticket médio (R$)"]
print("01 · X:", X.shape)

# Padronização: sem ela o ticket (centenas de R$) dominaria a distância
media, desvio = X.mean(axis=0), X.std(axis=0)
X_esc = (X - media) / desvio


# =============================================================
# 02 a 08 · Funções do algoritmo
# =============================================================
def inicializar_centroides(X, K, semente):                   # 02
    """Sorteia K amostras distintas de X como centroides iniciais."""
    indices = np.random.default_rng(semente).choice(len(X), size=K, replace=False)
    return X[indices].copy()


def calcular_distancias(X, centroides):                       # 03
    """Distâncias AO QUADRADO amostra × centroide, com broadcasting: (n, K)."""
    dif = X[:, None, :] - centroides[None, :, :]
    return np.sum(dif ** 2, axis=2)


def atribuir_clusters(distancias):                            # 04
    """Índice do centroide mais próximo de cada amostra: (n,)."""
    return np.argmin(distancias, axis=1)


def atualizar_centroides(X, labels, centroides, distancias):  # 05 e 07
    """Média de cada cluster. Cluster vazio recebe a amostra mais distante."""
    novos = centroides.copy()
    d_atual = distancias[np.arange(len(X)), labels].copy()
    for k in range(len(centroides)):
        pontos = X[labels == k]
        if len(pontos) > 0:
            novos[k] = pontos.mean(axis=0)
        else:                                                 # 07 · cluster vazio
            mais_longe = np.argmax(d_atual)
            novos[k] = X[mais_longe]
            d_atual[mais_longe] = -1.0
    return novos


def inercia(X, labels, centroides):                           # 08
    """Soma das distâncias ao quadrado de cada amostra ao seu centroide."""
    return np.sum((X - centroides[labels]) ** 2)


class KMeansDoZero:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if not 1 <= self.n_clusters <= len(X):
            raise ValueError("n_clusters deve estar entre 1 e o número de amostras")
        C = inicializar_centroides(X, self.n_clusters, self.random_state)
        self.historico_centroides_ = [C.copy()]
        self.historico_inercia_ = []

        for it in range(1, self.max_iter + 1):
            D = calcular_distancias(X, C)
            labels = atribuir_clusters(D)
            self.historico_inercia_.append(D[np.arange(len(X)), labels].sum())
            novos = atualizar_centroides(X, labels, C, D)
            desloc = np.sqrt(((novos - C) ** 2).sum(axis=1)).max()   # 06 · convergência
            C = novos
            self.historico_centroides_.append(C.copy())
            if desloc <= self.tol:
                break

        # estado final: labels_ e inercia_ calculados com os centroides FINAIS
        self.centroides_ = C
        self.labels_ = atribuir_clusters(calcular_distancias(X, C))
        self.inercia_ = inercia(X, self.labels_, C)
        self.historico_inercia_.append(self.inercia_)
        self.n_iter_ = it
        return self

    def predict(self, X):
        return atribuir_clusters(calcular_distancias(np.asarray(X, dtype=float), self.centroides_))

    def fit_predict(self, X):
        return self.fit(X).labels_


# 07 · teste de cluster vazio: o 3º centroide está longe de todos os pontos
X_t = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)
C_t = np.array([[0, 0.5], [10, 10.5], [50, 50]])
D_t = calcular_distancias(X_t, C_t)
C_novo = atualizar_centroides(X_t, atribuir_clusters(D_t), C_t, D_t)
print("07 · cluster vazio tratado, tem NaN?", np.isnan(C_novo).any())


# =============================================================
# 10 · Método do cotovelo (melhor de 10 sementes para cada K)
# =============================================================
valores_K = range(1, 9)
inercias_K = [min(KMeansDoZero(k, random_state=s).fit(X_esc).inercia_ for s in range(10))
              for k in valores_K]
print("\n10 · inércia por K:", np.round(inercias_K, 1))

plt.plot(valores_K, inercias_K, marker="o")
plt.xlabel("K"); plt.ylabel("inércia"); plt.title("Método do cotovelo")
plt.show()

K = 4   # a queda perde força depois de K = 4


# =============================================================
# 11 · Comparar sementes
# =============================================================
print("\n11 · sementes:")
modelos = []
for s in range(10):
    m = KMeansDoZero(K, random_state=s).fit(X_esc)
    modelos.append(m)
    print(f"   semente {s}: inércia = {m.inercia_:7.3f} | iterações = {m.n_iter_:2d} | "
          f"tamanhos = {np.bincount(m.labels_, minlength=K)}")
modelo = min(modelos, key=lambda m: m.inercia_)      # fica com a menor inércia
print("   escolhida: semente", modelo.random_state, "| inércia", round(modelo.inercia_, 3))


# =============================================================
# Critério de aceite: labels_ e inercia_ consistentes com centroides_, sem NaN
# =============================================================
assert not np.isnan(modelo.centroides_).any()
assert np.array_equal(modelo.labels_, modelo.predict(X_esc))
assert np.isclose(modelo.inercia_, inercia(X_esc, modelo.labels_, modelo.centroides_))
D_fim = calcular_distancias(X_esc, modelo.centroides_)
assert np.isclose(modelo.inercia_, D_fim[np.arange(len(X_esc)), modelo.labels_].sum())   # 08 · duas formas
print("\nCritério de aceite: labels_ e inercia_ consistentes, sem NaN ✓")


# =============================================================
# Centroides finais e evolução da inércia
# =============================================================
C_orig = modelo.centroides_ * desvio + media           # volta para as unidades originais
print("\nCentroides finais (visitas/mês, ticket R$):")
print(C_orig.round(1))
print("Histórico da inércia:", np.round(modelo.historico_inercia_, 3))

plt.plot(range(len(modelo.historico_inercia_)), modelo.historico_inercia_, marker="o")
plt.xlabel("iteração"); plt.ylabel("inércia"); plt.title("Evolução da inércia")
plt.show()


# =============================================================
# 09 · Visualizar os grupos (unidades originais)
# =============================================================
cores = np.array(["#3B82F6", "#F59E0B", "#10B981", "#EF4444"])
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=cores[modelo.labels_], s=20)
plt.scatter(C_orig[:, 0], C_orig[:, 1], marker="X", s=250, c=cores, edgecolor="black")
plt.xlabel(nomes_col[0]); plt.ylabel(nomes_col[1])
plt.title(f"Segmentos de clientes · K = {K}")
plt.show()


# =============================================================
# 12 · Comparar com o scikit-learn
# =============================================================
sk = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X_esc)
par = atribuir_clusters(calcular_distancias(modelo.centroides_, sk.cluster_centers_))
print("\n12 · inércia nossa:  ", round(modelo.inercia_, 3))
print("     inércia sklearn:", round(sk.inertia_, 3))
print("     maior diferença entre centroides:",
      round(np.abs(modelo.centroides_ - sk.cluster_centers_[par]).max(), 6))
print("     concordância dos rótulos:", (par[modelo.labels_] == sk.labels_).mean())


# =============================================================
# 13 · Interpretar os resultados
# =============================================================
print("\n13 · perfil de cada grupo (médias nas unidades originais):")
print(f"{'grupo':>5} | {'clientes':>8} | {'visitas/mês':>11} | {'ticket (R$)':>11}")
for k in np.argsort(C_orig[:, 1]):                     # do menor para o maior ticket
    g = X[modelo.labels_ == k]
    print(f"{k:>5} | {len(g):>8} | {g[:, 0].mean():>11.1f} | {g[:, 1].mean():>11.1f}")

# Como o dataset é sintético, dá para conferir: em que grupo caiu cada perfil verdadeiro?
print("\n13 · perfil verdadeiro (linhas) × grupo encontrado (colunas):")
print(f"{'':>21}" + "".join(f"{'g' + str(k):>5}" for k in range(K)))
for i, nome in enumerate(perfis):
    cont = np.bincount(modelo.labels_[perfil_real == i], minlength=K)
    print(f"{nome:>21}" + "".join(f"{c:>5}" for c in cont))
