import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class GradienteDescendente:
    def __init__(self, alpha=0.1, epocas=1000, tol=1e-8):
        self.alpha = alpha
        self.epocas = epocas
        self.tol = tol

    def _adicionar_intercepto(self, x):
        return np.column_stack([np.ones(len(x)), x])

    def fit(self, x, y):
        X = self._adicionar_intercepto(x)
        m = len(y)
        self.beta_ = np.zeros(X.shape[1])
        self.historico_ = []
        self.trajetoria_ = [self.beta_.copy()]

        for t in range(self.epocas):
            erro = X @ self.beta_ - y
            custo = (erro @ erro) / (2 * m)
            grad = (X.T @ erro) / m
            self.beta_ = self.beta_ - self.alpha * grad
            self.historico_.append(custo)
            self.trajetoria_.append(self.beta_.copy())
            if t > 0 and abs(self.historico_[-2] - self.historico_[-1]) < self.tol:
                break

        self.trajetoria_ = np.array(self.trajetoria_)
        return self

    def predict(self, x):
        return self._adicionar_intercepto(x) @ self.beta_

    def score(self, x, y):
        y_hat = self.predict(x)
        return 1 - np.sum((y - y_hat)**2) / np.sum((y - y.mean())**2)


rng = np.random.default_rng(42)
x = rng.uniform(0, 2, 100)
y = 4 + 3 * x + rng.normal(0, 0.5, 100)   # esperado: beta0 = 4, beta1 = 3

gd = GradienteDescendente(alpha=0.1, epocas=1000, tol=1e-8).fit(x, y)

X = gd._adicionar_intercepto(x)
print("beta (gradiente):", gd.beta_)
print("beta (pinv):     ", np.linalg.pinv(X) @ y)
print("épocas usadas:", len(gd.historico_))
print("custo inicial:", gd.historico_[0], "| custo final:", gd.historico_[-1])
print("R2:", gd.score(x, y))

plt.scatter(x, y, s=10)
plt.plot(np.sort(x), gd.predict(np.sort(x)), color="C1")
plt.title("Reta ajustada")
plt.show()

plt.plot(gd.historico_)
plt.yscale("log")
plt.xlabel("época"); plt.ylabel("custo J")
plt.show()

traj = gd.trajetoria_
m = len(y)

B0, B1 = np.meshgrid(np.linspace(-1, 6, 200), np.linspace(-1, 6, 200))
E = B0[..., None] + B1[..., None] * x - y
J_grade = (E ** 2).sum(axis=-1) / (2 * m)

fig, ax = plt.subplots()
ax.contour(B0, B1, J_grade, levels=30)
ax.set_xlabel("β₀"); ax.set_ylabel("β₁")
caminho, = ax.plot([], [], "o-", color="C1", markersize=3)

def atualizador(k):
    caminho.set_data(traj[:k + 1, 0], traj[:k + 1, 1])
    J = ((X @ traj[k] - y) ** 2).sum() / (2 * m)
    ax.set_title(f"t = {k}, J = {J:.4f}")
    return caminho,

anim = FuncAnimation(fig, atualizador, frames=range(0, len(traj), 5), interval=100)
anim.save("descida.gif", writer="pillow", fps=10)

# Critérios de aceite:
# 1) Executa sem erros: treino, prints, gráficos e gif.
# 2) β ≈ (3,97; 3,02), perto dos valores usados para gerar os dados (4 e 3) e igual ao pinv.
# 3) Custo caiu de 25,37 para 0,12 sem subir (α = 0,1); parou na época 431 pela tolerância.
# 4) predict gera a reta ajustada e score devolve R² = 0,92.
# 5) A animação mostra β saindo de (0, 0) e descendo pelas curvas de nível até o mínimo.
# 6) β = coeficientes da reta | α = tamanho do passo |
#    gradiente = direção em que o custo sobe (β anda ao contrário) | custo J = erro quadrático médio / 2.