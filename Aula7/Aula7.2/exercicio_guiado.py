
# =============================================================
# ETAPA 01 · Imports
# =============================================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


# =============================================================
# ETAPA 02 · Criar o dataset
# =============================================================
X, y = make_circles(
    n_samples=300,
    factor=0.45,
    noise=0.10,
    random_state=42
)
print(X.shape)
print(y.shape)
print(np.unique(y, return_counts=True))


# =============================================================
# ETAPA 03 · Visualizar o dataset
# =============================================================
plt.figure(figsize=(6, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1],
            c="#B2B9C1", marker="s",
            edgecolor="k", label="Classe 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1],
            c="#FF6B35", marker="o",
            edgecolor="k", label="Classe 1")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend(loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, -0.1), frameon=False)
plt.axis("equal")
plt.show()


# =============================================================
# ETAPA 04 · Separar treino e teste
# =============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
print(X_train.shape, X_test.shape)


# =============================================================
# ETAPA 05 · Construir pipeline SVM Linear
# =============================================================
svm_linear = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="linear", C=1.0))
])


# =============================================================
# ETAPA 06 · Treinar SVM Linear
# =============================================================
print(svm_linear.fit(X_train, y_train))


# =============================================================
# ETAPA 07 · Prever com SVM Linear
# =============================================================
pred_linear = svm_linear.predict(X_test)
print("Reais:    ", y_test[:15])
print("Previstos:", pred_linear[:15])


# =============================================================
# ETAPA 08 · Acurácia do SVM Linear
# =============================================================
acc_linear = accuracy_score(y_test, pred_linear)
print(f"Acurácia (linear): {acc_linear:.3f}")


# =============================================================
# ETAPA 09 · Matriz de confusão do SVM Linear
# =============================================================
cm_linear = confusion_matrix(y_test, pred_linear)
print(cm_linear)


# =============================================================
# ETAPA 10 · Função de visualização da fronteira
# =============================================================
def plot_decision_boundary(model, X, y, ax, title,
                           x_lim, y_lim, resolution=300):
    # 1. grade em coordenadas ORIGINAIS
    xx, yy = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], resolution),
        np.linspace(y_lim[0], y_lim[1], resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 2. o pipeline escalona a grade sozinho
    Z = model.predict(grid).reshape(xx.shape)
    D = model.decision_function(grid).reshape(xx.shape)

    # 3. regiões, fronteira (0) e margens (±1)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
                colors=["#E7E5E1", "#F8DED1"])
    ax.contour(xx, yy, D, levels=[-1, 0, 1],
               colors="k", linestyles=["--", "-", "--"])

    # 4. amostras
    ax.scatter(X[y == 0, 0], X[y == 0, 1],
               c="#B2B9C1", marker="s",
               edgecolor="k", s=30, label="Classe 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1],
               c="#FF6B35", marker="o",
               edgecolor="k", s=30, label="Classe 1")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend(loc="upper center", ncol=3,
              frameon=False, bbox_to_anchor=(0.5, -0.12))

x_lim = (X[:, 0].min() - 0.3, X[:, 0].max() + 0.3)
y_lim = (X[:, 1].min() - 0.3, X[:, 1].max() + 0.3)
fig, ax = plt.subplots(figsize=(6, 6))
plot_decision_boundary(svm_linear, X_train, y_train,
                       ax, "SVM Linear (C=1.0)", x_lim, y_lim)
plt.show()


# =============================================================
# ETAPA 11 · Construir pipeline SVM RBF
# =============================================================
svm_rbf = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", C=1.0, gamma="scale"))
])


# =============================================================
# ETAPA 12 · Treinar SVM RBF
# =============================================================
svm_rbf.fit(X_train, y_train)

X_train_scaled = svm_rbf.named_steps["scaler"].transform(X_train)
print("gamma efetivo:",
      1 / (X_train_scaled.shape[1] * X_train_scaled.var()))


# =============================================================
# ETAPA 13 · Avaliar SVM RBF
# =============================================================
pred_rbf = svm_rbf.predict(X_test)
acc_rbf = accuracy_score(y_test, pred_rbf)
cm_rbf = confusion_matrix(y_test, pred_rbf)
print(f"Acurácia (RBF): {acc_rbf:.3f}")
print(cm_rbf)


# =============================================================
# ETAPA 14 · Comparar as fronteiras lado a lado
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
plot_decision_boundary(svm_linear, X_train, y_train, axes[0],
                       f"SVM Linear | teste {acc_linear:.3f}", x_lim, y_lim)
plot_decision_boundary(svm_rbf, X_train, y_train, axes[1],
                       f"SVM RBF | teste {acc_rbf:.3f}", x_lim, y_lim)
plt.tight_layout()
plt.show()


# =============================================================
# ETAPA 15 · Vetores de suporte nas coordenadas corretas
# =============================================================
svc_rbf = svm_rbf.named_steps["svc"]
idx_sv = svc_rbf.support_
print("Total de vetores de suporte:", len(idx_sv))
print("Por classe:", svc_rbf.n_support_)
# índices valem para X_train (mesma ordem)
sv_original = X_train[idx_sv]
# support_vectors_ está ESCALONADO
scaler = svm_rbf.named_steps["scaler"]
sv_reconstruido = scaler.inverse_transform(
    svc_rbf.support_vectors_)
print(np.allclose(sv_original, sv_reconstruido))
fig, ax = plt.subplots(figsize=(6, 6))
plot_decision_boundary(svm_rbf, X_train, y_train, ax,
                       "SVM RBF | vetores de suporte", x_lim, y_lim)
ax.scatter(sv_original[:, 0], sv_original[:, 1],
           s=160, facecolors="none",
           edgecolors="#0C0D10", linewidths=1.5,
           label="Vetores de suporte")
ax.legend(loc="upper center", ncol=3,
          frameon=False, bbox_to_anchor=(0.5, -0.12))
plt.show()


# =============================================================
# ETAPA 16 · Experimento com C (gamma=1.0 fixo)
# =============================================================
valores_C = [0.01, 1, 100]
modelos_C = {}
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, C in zip(axes, valores_C):
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=C,
                    gamma=1.0)),
    ])
    modelo.fit(X_train, y_train)
    modelos_C[C] = modelo
    acc_tr = accuracy_score(
        y_train, modelo.predict(X_train))
    acc_te = accuracy_score(
        y_test, modelo.predict(X_test))
    plot_decision_boundary(
        modelo, X_train, y_train, ax,
        f"C={C} | treino {acc_tr:.3f}"
        f" | teste {acc_te:.3f}",
        x_lim, y_lim)
plt.tight_layout()
plt.show()


# =============================================================
# ETAPA 17 · Experimento com gamma (C=1.0 fixo)
# =============================================================
valores_gamma = [0.1, 1, 10]
modelos_gamma = {}
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, g in zip(axes, valores_gamma):
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0,
                    gamma=g)),
    ])
    modelo.fit(X_train, y_train)
    modelos_gamma[g] = modelo
    acc_tr = accuracy_score(
        y_train, modelo.predict(X_train))
    acc_te = accuracy_score(
        y_test, modelo.predict(X_test))
    plot_decision_boundary(
        modelo, X_train, y_train, ax,
        f"gamma={g} | treino {acc_tr:.3f}"
        f" | teste {acc_te:.3f}",
        x_lim, y_lim)
plt.tight_layout()
plt.show()


# =============================================================
# ETAPA 18 · Tabela comparativa
# =============================================================
def resumo(nome, modelo):
    svc = modelo.named_steps["svc"]
    g = "-" if svc.kernel == "linear" else svc.gamma
    return [nome, svc.kernel, svc.C, g,
            accuracy_score(y_train, modelo.predict(X_train)),
            accuracy_score(y_test, modelo.predict(X_test)),
            int(svc.n_support_.sum())]

linhas = [resumo("Linear", svm_linear),
          resumo("RBF scale", svm_rbf)]
linhas += [resumo(f"RBF C={c}", m)
           for c, m in modelos_C.items()]
linhas += [resumo(f"RBF gamma={g}", m)
           for g, m in modelos_gamma.items()]

print(f"{'Modelo':<15}{'Kernel':<8}{'C':>7}"
      f"{'gamma':>8}{'Treino':>9}{'Teste':>8}{'VS':>6}")
for nome, k, c, g, tr, te, nv in linhas:
    print(f"{nome:<15}{k:<8}{c:>7}{str(g):>8}"
          f"{tr:>9.3f}{te:>8.3f}{nv:>6}")


# =============================================================
# ETAPA 19 · Conclusão
# =============================================================
# - Linear: uma reta não separa disco e anel.
# - RBF: a similaridade por distância permite obter uma fronteira fechada.
# - C: regula o custo das violações e o equilíbrio entre margem e ajuste.
# - Gamma: controla o alcance da influência de cada vetor de suporte.
# - Escalonamento: deve ficar dentro do Pipeline.
