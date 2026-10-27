

# =============================================================
# ETAPA 01 · Importar as bibliotecas
# =============================================================
import numpy as np                       # arrays e contas vetorizadas
import matplotlib.pyplot as plt          # gráficos

from sklearn.model_selection import train_test_split    # separar treino/teste
from sklearn.metrics import accuracy_score, confusion_matrix  # métricas

print("Bibliotecas carregadas")


# =============================================================
# ETAPA 02 · Criar o dataset do zero
# =============================================================
# Cada linha é um cliente: [idade (anos), visitas à loja no mês]
X = np.array([
    [19, 1], [25, 6], [22, 6], [28, 8], [24, 2], [31, 3],
    [27, 3], [34, 9], [35, 2], [36, 5], [41, 3], [39, 8],
    [47, 1], [42, 6], [55, 4], [45, 10], [58, 8], [29, 4],
    [61, 6], [48, 7], [64, 9], [33, 11], [38, 4], [52, 9]])

# 0 = não comprou, 1 = comprou (um rótulo por linha de X)
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
              0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

print("X.shape:", X.shape, "| y.shape:", y.shape)
print("Clientes por classe:", np.bincount(y))


# =============================================================
# ETAPA 03 · Visualizar os dados
# =============================================================
nao = y == 0                     # máscara: quem não comprou
sim = y == 1                     # máscara: quem comprou

plt.scatter(X[nao, 0], X[nao, 1], c="#F4BA64", label="0 · não comprou")
plt.scatter(X[sim, 0], X[sim, 1], c="#35D49A", label="1 · comprou")
plt.xlabel("Idade (anos)")
plt.ylabel("Visitas à loja no mês")
plt.title("Clientes")
plt.legend()
plt.show()


# =============================================================
# ETAPA 04 · Separar treino e teste
# =============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,      # 25% dos clientes vão para o teste
    random_state=42,     # mesma divisão em toda execução
    stratify=y)          # mesma proporção de classes nos dois lados

print("Treino:", X_train.shape, "| Teste:", X_test.shape)
print("Classes no treino:", np.bincount(y_train))
print("Classes no teste: ", np.bincount(y_test))


# =============================================================
# ETAPA 05 · Calcular o Gini manualmente
# =============================================================
y_exemplo = np.array([0, 0, 1, 1, 1])

n0 = np.sum(y_exemplo == 0)          # 2 amostras da classe 0
n1 = np.sum(y_exemplo == 1)          # 3 amostras da classe 1
p0 = n0 / len(y_exemplo)             # 2/5 = 0.4
p1 = n1 / len(y_exemplo)             # 3/5 = 0.6
soma = p0 ** 2 + p1 ** 2             # 0.16 + 0.36 = 0.52

print("Gini =", round(1 - soma, 4))


# =============================================================
# ETAPA 06 · Implementar a função gini(y)
# =============================================================
def gini(y):
    if len(y) == 0:                                      # grupo vazio
        return 0.0
    classes, contagens = np.unique(y, return_counts=True)  # n_i
    proporcoes = contagens / len(y)                      # p_i = n_i / n
    return 1.0 - np.sum(proporcoes ** 2)                 # 1 - Σ p_i²

print(round(gini(y_exemplo), 4))         # exemplo da etapa 5
print(gini(np.array([0, 0, 0, 0])))      # grupo puro
print(gini(np.array([0, 1, 0, 1])))      # mistura máxima
print(gini(y_train))                     # treino inteiro


# =============================================================
# ETAPA 07 · Testar uma divisão: Idade <= 30?
# =============================================================
coluna = 0                    # 0 = Idade, 1 = Visitas
limiar = 30
mascara = X_train[:, coluna] <= limiar      # True = responde Sim

X_esq, X_dir = X_train[mascara], X_train[~mascara]
y_esq, y_dir = y_train[mascara], y_train[~mascara]

print("y_esq:", y_esq)
print("y_dir:", y_dir)
print("Gini esq:", gini(y_esq), "| Gini dir:", gini(y_dir))
n = len(y_train)
pond = len(y_esq) / n * gini(y_esq) + len(y_dir) / n * gini(y_dir)
print("Gini ponderado:", round(pond, 4))


# =============================================================
# ETAPA 08 · Implementar gini_ponderado()
# =============================================================
def gini_ponderado(y_esq, y_dir):
    n = len(y_esq) + len(y_dir)              # total de amostras
    if n == 0:
        return 0.0
    peso_esq = len(y_esq) / n                # n_esq / n
    peso_dir = len(y_dir) / n                # n_dir / n
    return peso_esq * gini(y_esq) + peso_dir * gini(y_dir)

print(round(gini_ponderado(y_esq, y_dir), 4))                              # etapa 7
print(round(gini_ponderado(np.array([0, 0]), np.array([1, 1, 1, 0])), 4))
print(gini_ponderado(np.array([]), np.array([0, 1])))                      # lado vazio


# =============================================================
# ETAPA 09 · Encontrar o melhor corte: melhor_divisao()
# =============================================================
def melhor_divisao(X, y):
    melhor = (None, None, np.inf)                       # (coluna, limiar, impureza)
    for coluna in range(X.shape[1]):                    # 1. cada característica
        valores = np.unique(X[:, coluna])               # 2-3. distintos e ordenados
        limiares = (valores[:-1] + valores[1:]) / 2     # 4. pontos médios
        for limiar in limiares:                         # 5. testar cada um
            mascara = X[:, coluna] <= limiar            # 6. separar
            imp = gini_ponderado(y[mascara], y[~mascara])   # 7. avaliar
            if imp < melhor[2]:                         # 8. comparar
                melhor = (coluna, limiar, imp)          # 9-10. guardar
    return melhor                                       # 11. retornar

col, lim, imp = melhor_divisao(X_train, y_train)
nomes = ["Idade", "Visitas"]

print(nomes[col], "<=", lim, "| Gini ponderado:", round(imp, 4))
print("Ganho:", round(gini(y_train) - imp, 4))


# =============================================================
# ETAPA 10 · Criar a estrutura dos nós: classe No
# =============================================================
class No:
    def __init__(self, feature=None, threshold=None, left=None,
                 right=None, classe=None, samples=0, impureza=0.0):
        self.feature = feature          # coluna da pergunta
        self.threshold = threshold      # limiar da pergunta
        self.left = left                # filho Sim (<=)
        self.right = right              # filho Não (>)
        self.classe = classe            # classe majoritária
        self.samples = samples          # amostras no nó
        self.impureza = impureza        # Gini do nó

    def eh_folha(self):
        return self.left is None and self.right is None

folha = No(classe=0, samples=4, impureza=0.0)
print(folha.classe, folha.samples, folha.eh_folha())


# =============================================================
# ETAPA 11 · Recursividade: a função chama a si mesma
# =============================================================
def contagem(n):
    if n == 0:                  # caso base: para aqui
        print("fim")
        return
    print(n)
    contagem(n - 1)             # chamada recursiva com problema menor

contagem(3)


# =============================================================
# ETAPA 12 · A classe ArvoreDecisaoDoZero
# =============================================================
class ArvoreDecisaoDoZero:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth                  # profundidade máxima
        self.min_samples_split = min_samples_split  # mínimo para dividir
        self.raiz = None                            # árvore ainda vazia

    def _criar_folha(self, y):
        classes, contagens = np.unique(y, return_counts=True)
        return No(classe=classes[np.argmax(contagens)],   # majoritária
                  samples=len(y), impureza=gini(y))

    def _construir(self, X, y, depth):
        no = self._criar_folha(y)                         # começa como folha
        parar = (no.impureza == 0 or depth >= self.max_depth
                 or len(y) < self.min_samples_split)
        if parar:
            return no                                     # caso base
        col, lim, imp = melhor_divisao(X, y)
        if col is None or no.impureza - imp <= 1e-12:
            return no                                     # sem divisão válida
        mascara = X[:, col] <= lim
        no.feature, no.threshold = col, lim
        no.left = self._construir(X[mascara], y[mascara], depth + 1)
        no.right = self._construir(X[~mascara], y[~mascara], depth + 1)
        return no

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.raiz = self._construir(X, y, depth=0)        # começa pela raiz
        return self

    def _prever_amostra(self, amostra, no):
        if no.eh_folha():
            return no.classe
        if amostra[no.feature] <= no.threshold:
            return self._prever_amostra(amostra, no.left)
        return self._prever_amostra(amostra, no.right)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._prever_amostra(a, self.raiz) for a in X])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))


# =============================================================
# ETAPA 13 · Treinar o nosso modelo
# =============================================================
modelo = ArvoreDecisaoDoZero(max_depth=3)      # objeto com a árvore vazia
print("Antes do fit:", modelo.raiz)

modelo.fit(X_train, y_train)                   # constrói a árvore
raiz = modelo.raiz

print("Treinamento concluído")
print("Raiz:", nomes[raiz.feature], "<=", raiz.threshold,
      "| amostras:", raiz.samples, "| Gini:", raiz.impureza)


# =============================================================
# ETAPA 14 · Realizar previsões no conjunto de teste
# =============================================================
y_pred = modelo.predict(X_test)

print("Real:    ", y_test)
print("Previsto:", y_pred)
print("Acertos: ", y_test == y_pred)


# O percurso de um cliente pela árvore
def explicar(amostra, no, nomes):
    while not no.eh_folha():
        valor = amostra[no.feature]
        sim = valor <= no.threshold
        print(nomes[no.feature], valor, "<=", no.threshold, "?",
              "Sim" if sim else "Não")
        no = no.left if sim else no.right
    print("Folha -> classe", no.classe)

explicar(X_test[1], modelo.raiz, nomes)        # cliente [52, 9]


# =============================================================
# ETAPA 15 · Avaliar o modelo: acurácia e matriz de confusão
# =============================================================
acuracia = accuracy_score(y_test, y_pred)
print("Acurácia:", round(acuracia, 3))
print("Nossa score():", round(modelo.score(X_test, y_test), 3))
print("Treino:", round(modelo.score(X_train, y_train), 3))

cm = confusion_matrix(y_test, y_pred)
print(cm)


# =============================================================
# ETAPA 16 · Imprimir a árvore treinada no terminal
# =============================================================
def imprimir_arvore(no, nomes, nivel=0, rotulo="[Raiz]"):
    recuo = "    " * nivel                       # 4 espaços por nível
    if no.eh_folha():
        print(f"{recuo}{rotulo}")
        print(f"{recuo}    Classe: {no.classe}")
        print(f"{recuo}    Amostras: {no.samples}")
        return
    print(f"{recuo}{rotulo} {nomes[no.feature]} <= {no.threshold:g}?")
    imprimir_arvore(no.left, nomes, nivel + 1, "[Sim]")
    imprimir_arvore(no.right, nomes, nivel + 1, "[Não]")

imprimir_arvore(modelo.raiz, nomes)


# =============================================================
# ETAPA 17 · Visualizar as regiões de decisão
# =============================================================
xx, yy = np.meshgrid(np.linspace(15, 68, 200), np.linspace(0, 12, 200))
grade = np.c_[xx.ravel(), yy.ravel()]                 # 40 000 pontos
Z = modelo.predict(grade).reshape(xx.shape)           # classe de cada ponto

plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
             colors=["#F4BA64", "#35D49A"], alpha=0.25)
for c, cor in [(0, "#F4BA64"), (1, "#35D49A")]:
    plt.scatter(X_train[y_train == c, 0], X_train[y_train == c, 1],
                c=cor, label=str(c))
plt.xlabel("Idade (anos)"); plt.ylabel("Visitas à loja no mês")
plt.title("Regiões de decisão · max_depth=3"); plt.legend()
plt.show()


# =============================================================
# ETAPA 18 · Comparar com o DecisionTreeClassifier
# =============================================================
from sklearn.tree import DecisionTreeClassifier

sk = DecisionTreeClassifier(criterion="gini", max_depth=3,
                            min_samples_split=2, random_state=0)
sk.fit(X_train, y_train)                       # mesmo treino
y_pred_sk = sk.predict(X_test)                 # mesmo teste

print("Acurácia nossa:  ", round(accuracy_score(y_test, y_pred), 3))
print("Acurácia sklearn:", round(accuracy_score(y_test, y_pred_sk), 3))
print("Mesmas predições:", np.array_equal(y_pred, y_pred_sk))
print("sklearn:", sk.get_depth(), "níveis |", sk.get_n_leaves(), "folhas")