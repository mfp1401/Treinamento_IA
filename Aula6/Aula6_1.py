
import numpy as np
import timeit

rng = np.random.default_rng(42)
n, d = 10_000, 5

# 1. Criando os dados
X = rng.normal(size=(n, d))
w = rng.normal(size=d)
y = X @ w + rng.normal(size=n, scale=0.1)

print("X:", X.shape, "| w:", w.shape, "| y:", y.shape)
print("Exemplo de previsão:", X[0] @ w)
print("Valor esperado:", y[0])

# 2. Comparando valores
total = 0.0
total_v = X[0] @ w
print("\nValores aproximadamente iguais?", np.isclose(total, total_v))

y_hat = np.zeros(n)
y_hat_v = X @ w
print("Vetores iguais?", np.allclose(y_hat, y_hat_v))

# 3. Medindo o tempo
tempo = timeit.timeit(lambda: X @ w, number=1000)
print(f"\nTempo de 1000 multiplicações: {tempo:.4f} s")

# 4. Calculando distâncias entre vetores
A, B = X[:3], X[3:7]

diff = A[:, None, :] - B[None, :, :]
D = np.sqrt(np.sum(diff ** 2, axis=2))

# Conferência com loops
D_v = np.zeros((len(A), len(B)))
for i in range(len(A)):
    for j in range(len(B)):
        D_v[i, j] = np.sqrt(np.sum((A[i] - B[j]) ** 2))

print("\nFormato das distâncias:", D.shape)
print("Cálculo vetorizado correto?", np.allclose(D, D_v))

# 5. Calculando o erro e o gradiente
w0 = np.zeros(d)

erro = X @ w0 - y
mse_v = np.mean(erro ** 2)
grad_v = 2 / n * X.T @ erro

print(f"\nMSE inicial: {mse_v:.4f}")
print("Gradiente:", grad_v)

# 6. Função de MSE e gradiente
def mse_grad(w):
    erro = X @ w - y
    return np.mean(erro ** 2), 2 / n * X.T @ erro

# 7. Conferindo o gradiente numericamente
def grad_numerico(f, w, h=1e-6):
    g = np.zeros_like(w)
    for j in range(len(w)):
        e = np.zeros_like(w)
        e[j] = h
        g[j] = (f(w + e) - f(w - e)) / (2 * h)
    return g

grad_num = grad_numerico(lambda w: mse_grad(w)[0], w0)

print("\nGradientes iguais?", np.allclose(grad_v, grad_num))
