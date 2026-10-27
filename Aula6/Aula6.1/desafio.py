
"""
algebra.py — Desafio da aula 6.1
Álgebra linear e cálculo para ML
"""

import numpy as np


# Normas L1 e L2
def norma(x, ord=2):
    if ord == 1:
        return np.sum(np.abs(x))
    if ord == 2:
        return np.sqrt(np.sum(x ** 2))
    raise ValueError("ord deve ser 1 ou 2")


# Distâncias euclidianas
def distancias(A, B):
    diff = A[:, None, :] - B[None, :, :]
    D = np.sqrt(np.sum(diff ** 2, axis=2))
    return D


# Erro quadrático médio
def mse(X, y, w):
    erro = X @ w - y
    return np.mean(erro ** 2)


# Gradiente do MSE
def grad_mse(X, y, w):
    n = len(y)
    erro = X @ w - y
    return 2 / n * X.T @ erro


# Gradiente numérico por diferença central
def grad_numerico(f, w, h=1e-6):
    g = np.zeros_like(w)

    for j in range(len(w)):
        e = np.zeros_like(w)
        e[j] = h
        g[j] = (f(w + e) - f(w - e)) / (2 * h)

    return g


# ==================================================
# TESTES DAS FUNÇÕES
# ==================================================

if __name__ == "__main__":

    print("=== 1. NORMAS ===")
    x = np.array([3.0, 4.0])

    print("Vetor:", x)
    print("Norma L1:", norma(x, ord=1))  # 7
    print("Norma L2:", norma(x, ord=2))  # 5

    print("\n=== 2. DISTÂNCIAS ===")
    A = np.array([[0.0, 0.0], [3.0, 4.0]])
    B = np.array([[0.0, 4.0], [3.0, 0.0]])

    D = distancias(A, B)

    print("Matriz de distâncias:")
    print(D)

    print("\n=== 3. MSE E GRADIENTE ===")
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([5.0, 11.0, 17.0])
    w = np.zeros(2)

    erro = mse(X, y, w)
    grad = grad_mse(X, y, w)

    print(f"MSE: {erro:.4f}")
    print("Gradiente analítico:", grad)

    print("\n=== 4. VALIDANDO O GRADIENTE ===")
    grad_num = grad_numerico(lambda w: mse(X, y, w), w)

    print("Gradiente numérico:", grad_num)
    print("Resultados compatíveis?", np.allclose(grad, grad_num))

    print("\n=== TESTES CONCLUÍDOS ===")
