import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression


dados = fetch_california_housing()
X,y = dados.data, dados.target


n =  X.shape[0]
Xb = np.column_stack([np.ones((n)),X])

A= Xb.T @ Xb
b= Xb.T @ y

beta_inv = np.linalg.inv(A) @ b
beta_solve = np.linalg.solve(A,b)
beta_pinv = np.linalg.pinv(Xb) @ y

y_hat = Xb @ beta_pinv

r2 = 1 - np.sum((y - y_hat)**2)/np.sum((y - y.mean())**2)

lr = LinearRegression()
lr.fit(X,y)
beta_sk = np.r_[lr.intercept_,lr.coef_]




print(X.shape,y.shape)
print(dados.feature_names)
print(Xb.shape)
print(Xb[0,:3])
print(np.allclose(beta_solve,beta_inv))
print(np.abs(beta_inv - beta_pinv).max())
print(r2)
print(np.allclose(beta_pinv,beta_sk))
print(lr.score(X,y))
