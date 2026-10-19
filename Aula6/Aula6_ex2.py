import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression


class MinimosQuadrados:
    def __init__(self, metodo="pinv"):
        if metodo not in ["inv", "solve", "pinv"]:
            raise ValueError("metodo invalido")
        self.metodo = metodo

    def _adicionar_intercepto(self, X):
        return np.column_stack([np.ones(X.shape[0]), X])

    def fit(self, X, y):
        Xb = self._adicionar_intercepto(X)
        A = Xb.T @ Xb
        b = Xb.T @ y

        if self.metodo == "inv":
            self.coef_ = np.linalg.inv(A) @ b
        elif self.metodo == "solve":
            self.coef_ = np.linalg.solve(A, b)
        else:
            self.coef_ = np.linalg.pinv(Xb) @ y
        return self

    def predict(self, X):
        return self._adicionar_intercepto(X) @ self.coef_

    def score(self, X, y):
        y_hat = self.predict(X)
        return 1 - np.sum((y - y_hat)**2) / np.sum((y - y.mean())**2)


dados = fetch_california_housing()
X, y = dados.data, dados.target

lr = LinearRegression().fit(X, y)

for metodo in ["inv", "solve", "pinv"]:
    m = MinimosQuadrados(metodo).fit(X, y)
    print(metodo,
          np.allclose(m.coef_, np.r_[lr.intercept_, lr.coef_]),
          np.allclose(m.predict(X), lr.predict(X)),
          np.isclose(m.score(X, y), lr.score(X, y)))


X_dup = np.column_stack([X, X[:, 0]])   # MedInc repetida
m = MinimosQuadrados("pinv")

print(np.linalg.cond(m._adicionar_intercepto(X)))
print(np.linalg.cond(m._adicionar_intercepto(X_dup)))

for metodo in ["inv", "solve", "pinv"]:
    try:
        m = MinimosQuadrados(metodo).fit(X_dup, y)
        print(metodo, "funcionou, R2 =", m.score(X_dup, y))
    except np.linalg.LinAlgError:
        print(metodo, "deu erro: matriz singular")


# Critérios de aceite:
# 1) inv, solve e pinv resolvem a mesma equação normal, por isso os três allclose dão True.
# 2) Com X_dup, XᵀX fica singular: inv e solve dão erro e pinv funciona (solução de menor norma).
# 3) cond impresso nos dois casos: ~2,4e5 sem duplicata e ~5,9e17 com a coluna repetida.
# 4) A classe só usa operações matriciais do NumPy; o único for está nos testes, fora da classe.