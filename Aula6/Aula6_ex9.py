import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier   


# =============================================================
# 1 · Receber X e y · dataset Iris (150 flores, 3 espécies)
# Usamos 2 características para poder desenhar as regiões.
# =============================================================
iris = load_iris()
X = iris.data[:, [2, 3]]               # comprimento e largura da pétala (cm)
y = iris.target                        # 0 = setosa, 1 = versicolor, 2 = virginica
nomes = ["Comp. pétala", "Larg. pétala"]
especies = iris.target_names

print("X:", X.shape, "| y:", y.shape, "| flores por classe:", np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
print("Treino:", X_train.shape, "| Teste:", X_test.shape)


# =============================================================
# 2 · Impureza de Gini
# =============================================================
def gini(y):
    if len(y) == 0:
        return 0.0
    _, contagens = np.unique(y, return_counts=True)
    proporcoes = contagens / len(y)
    return 1.0 - np.sum(proporcoes ** 2)


def gini_ponderado(y_esq, y_dir):
    n = len(y_esq) + len(y_dir)
    if n == 0:
        return 0.0
    return len(y_esq) / n * gini(y_esq) + len(y_dir) / n * gini(y_dir)


print("\nGini do treino (3 classes equilibradas):", round(gini(y_train), 4))   # 1 - 3*(1/3)² = 0.667


# =============================================================
# 3 · Buscar o melhor corte
# =============================================================
def melhor_divisao(X, y):
    melhor = (None, None, np.inf)                       # (coluna, limiar, impureza)
    for coluna in range(X.shape[1]):
        valores = np.unique(X[:, coluna])
        limiares = (valores[:-1] + valores[1:]) / 2     # pontos médios
        for limiar in limiares:
            mascara = X[:, coluna] <= limiar
            imp = gini_ponderado(y[mascara], y[~mascara])
            if imp < melhor[2]:                         # empate: fica o primeiro
                melhor = (coluna, limiar, imp)
    return melhor


col, lim, imp = melhor_divisao(X_train, y_train)
print("Melhor corte da raiz:", nomes[col], "<=", lim, "| Gini ponderado:", round(imp, 4))


# =============================================================
# 4 e 5 · Nós internos e folhas
# =============================================================
class No:
    def __init__(self, feature=None, threshold=None, left=None,
                 right=None, classe=None, samples=0, impureza=0.0):
        self.feature = feature          # coluna da pergunta (nó interno)
        self.threshold = threshold      # limiar da pergunta (nó interno)
        self.left = left                # filho Sim (<=)
        self.right = right              # filho Não (>)
        self.classe = classe            # classe majoritária (folha)
        self.samples = samples
        self.impureza = impureza

    def eh_folha(self):
        return self.left is None and self.right is None


# =============================================================
# 6 a 9 · Construção recursiva, critérios de parada,
#         previsões e acurácia
# =============================================================
class ArvoreDecisaoDoZero:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.raiz = None

    def _criar_folha(self, y):                                    # 5 · folha
        classes, contagens = np.unique(y, return_counts=True)
        return No(classe=classes[np.argmax(contagens)],
                  samples=len(y), impureza=gini(y))

    def _construir(self, X, y, depth):                            # 6 · recursão
        no = self._criar_folha(y)
        # 7 · critérios de parada: nó puro, profundidade máxima, poucas amostras
        parar = (no.impureza == 0 or depth >= self.max_depth
                 or len(y) < self.min_samples_split)
        if parar:
            return no
        col, lim, imp = melhor_divisao(X, y)
        if col is None or no.impureza - imp <= 1e-12:             # sem ganho
            return no
        mascara = X[:, col] <= lim                                # 4 · nó interno
        no.feature, no.threshold = col, lim
        no.left = self._construir(X[mascara], y[mascara], depth + 1)
        no.right = self._construir(X[~mascara], y[~mascara], depth + 1)
        return no

    def fit(self, X, y):                                          # 1 · recebe X e y
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.raiz = self._construir(X, y, depth=0)
        return self

    def _prever_amostra(self, amostra, no):
        if no.eh_folha():
            return no.classe
        if amostra[no.feature] <= no.threshold:
            return self._prever_amostra(amostra, no.left)
        return self._prever_amostra(amostra, no.right)

    def predict(self, X):                                         # 8 · previsões
        X = np.asarray(X, dtype=float)
        return np.array([self._prever_amostra(a, self.raiz) for a in X])

    def score(self, X, y):                                        # 9 · acurácia
        return np.mean(self.predict(X) == np.asarray(y))


# =============================================================
# Treinar, prever e avaliar
# =============================================================
modelo = ArvoreDecisaoDoZero(max_depth=3).fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print("\nReal:    ", y_test)
print("Previsto:", y_pred)
print("Acurácia treino:", round(modelo.score(X_train, y_train), 3))
print("Acurácia teste: ", round(modelo.score(X_test, y_test), 3))
print("Matriz de confusão (linhas = real, colunas = previsto):")
print(confusion_matrix(y_test, y_pred))


# =============================================================
# 10 · Imprimir a árvore
# =============================================================
def imprimir_arvore(no, nomes, nivel=0, rotulo="[Raiz]"):
    recuo = "    " * nivel
    if no.eh_folha():
        print(f"{recuo}{rotulo}")
        print(f"{recuo}    Classe: {no.classe} ({especies[no.classe]})")
        print(f"{recuo}    Amostras: {no.samples} | Gini: {no.impureza:.3f}")
        return
    print(f"{recuo}{rotulo} {nomes[no.feature]} <= {no.threshold:g}?")
    imprimir_arvore(no.left, nomes, nivel + 1, "[Sim]")
    imprimir_arvore(no.right, nomes, nivel + 1, "[Não]")


print()
imprimir_arvore(modelo.raiz, nomes)


# =============================================================
# Gráfico das regiões de decisão
# =============================================================
cores = ["#F4BA64", "#35D49A", "#5B8DEF"]
xx, yy = np.meshgrid(np.linspace(0.5, 7.5, 300), np.linspace(0, 2.8, 300))
grade = np.c_[xx.ravel(), yy.ravel()]
Z = modelo.predict(grade).reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5], colors=cores, alpha=0.25)
for c in range(3):
    plt.scatter(X_train[y_train == c, 0], X_train[y_train == c, 1],
                c=cores[c], edgecolors="k", label=especies[c])
plt.xlabel("Comprimento da pétala (cm)"); plt.ylabel("Largura da pétala (cm)")
plt.title("Regiões de decisão · max_depth=3"); plt.legend()
plt.show()


# =============================================================
# 11 · Comparar com o scikit-learn (mesmas condições)
# =============================================================
sk = DecisionTreeClassifier(criterion="gini", max_depth=3,
                            min_samples_split=2, random_state=0)
sk.fit(X_train, y_train)
y_pred_sk = sk.predict(X_test)

print("\nAcurácia nossa:  ", round(accuracy_score(y_test, y_pred), 3))
print("Acurácia sklearn:", round(accuracy_score(y_test, y_pred_sk), 3))
print("Mesmas predições:", np.array_equal(y_pred, y_pred_sk))
print("sklearn:", sk.get_depth(), "níveis |", sk.get_n_leaves(), "folhas")


# =============================================================
# Análise: efeito da profundidade (overfitting)
# =============================================================
print("\nmax_depth | treino | teste")
for d in [1, 2, 3, 4, 5, None]:
    m = ArvoreDecisaoDoZero(max_depth=d if d else 100).fit(X_train, y_train)
    print(f"{str(d):>9} | {m.score(X_train, y_train):.3f}  | {m.score(X_test, y_test):.3f}")