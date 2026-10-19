import numpy as np
import matplotlib.pyplot as plt


class RegressaoLogistica:
    def __init__(self, alpha=0.1, epocas=1000):
        self.alpha = alpha
        self.epocas = epocas
        self.beta = None
        self.historico = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)  # evita overflow no exp
        return 1 / (1 + np.exp(-z))

    def _com_intercepto(self, X):
        uns = np.ones((X.shape[0], 1))
        return np.hstack([uns, X])

    def _custo(self, p, y, eps=1e-12):
        p = np.clip(p, eps, 1 - eps)  # evita log(0)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def fit(self, X, y):
        Xb = self._com_intercepto(X)
        m = len(y)
        self.beta = np.zeros(Xb.shape[1])
        self.historico = []

        for epoca in range(self.epocas):
            p = self._sigmoid(Xb @ self.beta)
            self.historico.append(self._custo(p, y))
            grad = Xb.T @ (p - y) / m
            self.beta = self.beta - self.alpha * grad
        return self

    def predict_proba(self, X):
        return self._sigmoid(self._com_intercepto(X) @ self.beta)

    def predict(self, X, limiar=0.5):
        return (self.predict_proba(X) >= limiar).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# 1. Dados da aula
rng = np.random.default_rng(42)
m_por_classe = 50
X0 = rng.normal(loc=[-1, -1], scale=1.2, size=(m_por_classe, 2))
X1 = rng.normal(loc=[1, 1], scale=1.2, size=(m_por_classe, 2))
X_dados = np.vstack([X0, X1])
y = np.concatenate([np.zeros(m_por_classe), np.ones(m_por_classe)])

idx = rng.permutation(len(y))
corte = int(0.8 * len(y))
treino, teste = idx[:corte], idx[corte:]
X_treino, y_treino = X_dados[treino], y[treino]   # sem a coluna de 1s: a classe adiciona
X_teste, y_teste = X_dados[teste], y[teste]

modelo = RegressaoLogistica(alpha=0.1, epocas=1000).fit(X_treino, y_treino)

# 2. Visualizar os dados
plt.scatter(X_dados[y == 0, 0], X_dados[y == 0, 1], label="classe 0")
plt.scatter(X_dados[y == 1, 0], X_dados[y == 1, 1], label="classe 1")
plt.xlabel("x1"); plt.ylabel("x2"); plt.legend(); plt.show()

# 3. Prever
p_teste = modelo.predict_proba(X_teste)
y_pred = modelo.predict(X_teste)
print("beta:", modelo.beta.round(3))
print(p_teste[:5].round(3), y_pred[:5], y_teste[:5].astype(int))

# 4. Acurácia
print(f"acurácia treino {modelo.score(X_treino, y_treino):.3f} | teste {modelo.score(X_teste, y_teste):.3f}")

# Critérios de aceite
print("len(historico) == epocas:", len(modelo.historico) == modelo.epocas)
print("primeiro custo:", round(modelo.historico[0], 4), "| ln 2 =", round(np.log(2), 4))
print("último custo < primeiro:", modelo.historico[-1] < modelo.historico[0])
print("predict_proba em (0, 1):", np.all((p_teste > 0) & (p_teste < 1)))
print("predict só 0 e 1:", set(np.unique(y_pred)) <= {0, 1})

# 5. Curva de custo
plt.plot(modelo.historico)
plt.xlabel("época"); plt.ylabel("J(β)"); plt.show()

# 6. Fronteira aprendida (reta onde p = 0,5, ou seja, z = 0)
b0, b1, b2 = modelo.beta
x1 = np.linspace(X_dados[:, 0].min(), X_dados[:, 0].max(), 100)
x2 = -(b0 + b1 * x1) / b2

plt.scatter(X_dados[y == 0, 0], X_dados[y == 0, 1], label="classe 0")
plt.scatter(X_dados[y == 1, 0], X_dados[y == 1, 1], label="classe 1")
plt.plot(x1, x2, "k--", label="fronteira p = 0,5")
plt.xlabel("x1"); plt.ylabel("x2"); plt.legend(); plt.show()

# Critérios de aceite:
# 1) len(historico) = 1000 = epocas (um custo guardado por época).
# 2) Primeiro custo = ln 2: com beta zerado, p = 0,5 para todos e o custo é -ln(0,5).
# 3) Custo caiu de 0,69 para ~0,19 (não zera porque as classes se misturam).
# 4) Sigmoide fica sempre em (0, 1); predict usa limiar 0,5 e devolve só 0 e 1.
# 5) Acurácia no teste = 0,900 (>= 0,85), igual à do treino.
# 6) Fronteira: reta onde z = 0, x2 = -(b0 + b1*x1) / b2, calculada com o beta aprendido.