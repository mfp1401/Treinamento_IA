
# =============================================================
# ETAPA 01 · Imports
# =============================================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance

# auxiliares: custo computacional e tabela final
import time
import pandas as pd


# =============================================================
# ETAPA 02 · Criar o dataset
# =============================================================
X, y = make_moons(
    n_samples=600,
    noise=0.28,
    random_state=42
)
print("Formato de X:", X.shape)
print("Formato de y:", y.shape)
print("Classes:", np.unique(y))
print("Por classe:", np.bincount(y))


# =============================================================
# ETAPA 03 · Visualizar o dataset
# =============================================================
plt.figure(figsize=(7, 5))
plt.scatter(X[y == 0, 0], X[y == 0, 1], s=18, c="#352364",
            edgecolors="#15121D", linewidths=0.4, label="Classe 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=18, c="#D7FF56",
            edgecolors="#15121D", linewidths=0.4, label="Classe 1")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("make_moons: 600 amostras, noise=0.28")
plt.legend()
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
print(np.bincount(y_train), np.bincount(y_test))


# =============================================================
# ETAPA 05 · Treinar uma árvore
# =============================================================
arvore = DecisionTreeClassifier(random_state=42)
t0 = time.perf_counter()
arvore.fit(X_train, y_train)
tempo_arvore = time.perf_counter() - t0
print("Profundidade:", arvore.get_depth())
print("Folhas:", arvore.get_n_leaves())


# =============================================================
# ETAPA 06 · Prever com a árvore
# =============================================================
pred_arvore = arvore.predict(X_test)
print("Previsto:", pred_arvore[:10])
print("Real:    ", y_test[:10])


# =============================================================
# ETAPA 07 · Avaliar a árvore
# =============================================================
acc_treino_arvore = accuracy_score(y_train, arvore.predict(X_train))
acc_teste_arvore = accuracy_score(y_test, pred_arvore)
cm_arvore = confusion_matrix(y_test, pred_arvore)
print("Acc. treino:", acc_treino_arvore)
print("Acc. teste:", acc_teste_arvore)
print(cm_arvore)


# =============================================================
# ETAPA 08 · Função de região de decisão
# =============================================================
COR_CLASSE = ["#352364", "#D7FF56"]
COR_REGIAO = ["#C9BEF5", "#F1FFC2"]
LIMITES = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5,
           X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

def plotar_fronteira(modelo, X_plot, y_plot, ax, titulo, resolucao=300):
    x_min, x_max, y_min, y_max = LIMITES
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolucao),
                         np.linspace(y_min, y_max, resolucao))
    grade = np.c_[xx.ravel(), yy.ravel()]          # (90000, 2)
    Z = modelo.predict(grade).reshape(xx.shape)    # (300, 300)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=COR_REGIAO)
    for classe in (0, 1):
        m = y_plot == classe
        ax.scatter(X_plot[m, 0], X_plot[m, 1], s=18, c=COR_CLASSE[classe],
                   edgecolors="#15121D", linewidths=0.4, label=f"Classe {classe}")
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_title(titulo); ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.legend(loc="upper right")

fig, ax = plt.subplots(figsize=(7, 5.5))
plotar_fronteira(arvore, X_test, y_test, ax, "Árvore de Decisão")
plt.show()


# =============================================================
# ETAPA 09 · Criar a floresta
# =============================================================
floresta = RandomForestClassifier(
    n_estimators=500,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)


# =============================================================
# ETAPA 10 · Treinar as 500 árvores
# =============================================================
t0 = time.perf_counter()
floresta.fit(X_train, y_train)
tempo_floresta = time.perf_counter() - t0
print("Tempo de treino (s):", round(tempo_floresta, 3))


# =============================================================
# ETAPA 11 · Prever com a floresta
# =============================================================
pred_floresta = floresta.predict(X_test)
print("Discordâncias:", (pred_floresta != pred_arvore).sum())


# =============================================================
# ETAPA 12 · Avaliar a floresta
# =============================================================
acc_treino_floresta = accuracy_score(y_train, floresta.predict(X_train))
acc_teste_floresta = accuracy_score(y_test, pred_floresta)
cm_floresta = confusion_matrix(y_test, pred_floresta)
print("Acc. treino:", acc_treino_floresta)
print("Acc. teste:", acc_teste_floresta)
print(cm_floresta)


# =============================================================
# ETAPA 13 · Fronteiras lado a lado
# =============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)
plotar_fronteira(arvore, X_test, y_test, axes[0], "Árvore de Decisão (1 árvore)")
plotar_fronteira(floresta, X_test, y_test, axes[1], "Random Forest (500 árvores)")
plt.tight_layout()
plt.show()


# =============================================================
# ETAPA 14 · OOB e acurácia de teste
# =============================================================
print("OOB Score:", floresta.oob_score_)
print("Acc. teste:", acc_teste_floresta)
print("Diferença:", floresta.oob_score_ - acc_teste_floresta)


# =============================================================
# ETAPA 15 · Importância por impureza (MDI)
# =============================================================
nomes = ["x1", "x2"]
imp_mdi = floresta.feature_importances_
print(dict(zip(nomes, imp_mdi)), imp_mdi.sum())


# =============================================================
# ETAPA 16 · Importância por permutação (no teste)
# =============================================================
perm = permutation_importance(
    floresta,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)
print(perm.importances_mean, perm.importances_std)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(nomes, imp_mdi, color="#7255E8")
axes[0].set_title("Importância MDI (treino)")
axes[1].bar(nomes, perm.importances_mean, yerr=perm.importances_std, color="#352364")
axes[1].set_title("Importância por permutação (teste)")
plt.tight_layout()
plt.show()


# =============================================================
# ETAPA 17 · Quantidade de árvores
# =============================================================
quantidades = [1, 5, 10, 25, 50, 100, 200, 500]
resultados_n = []
for n in quantidades:
    usa_oob = n >= 25   # cobertura OOB suficiente
    rf_n = RandomForestClassifier(n_estimators=n, oob_score=usa_oob,
                                  random_state=42, n_jobs=-1)
    rf_n.fit(X_train, y_train)
    resultados_n.append({
        "n_estimators": n,
        "acc_treino": accuracy_score(y_train, rf_n.predict(X_train)),
        "acc_teste": accuracy_score(y_test, rf_n.predict(X_test)),
        "oob": rf_n.oob_score_ if usa_oob else np.nan,
    })
tabela_n = pd.DataFrame(resultados_n)
print(tabela_n.to_string(index=False))

plt.figure(figsize=(8, 4.5))
plt.plot(tabela_n["n_estimators"], tabela_n["acc_treino"], "o-", label="Treino")
plt.plot(tabela_n["n_estimators"], tabela_n["acc_teste"], "s-", label="Teste")
plt.plot(tabela_n["n_estimators"], tabela_n["oob"], "^--", label="OOB")
plt.xscale("log")
plt.xlabel("Número de árvores (escala log)")
plt.ylabel("Acurácia")
plt.title("Acurácia × número de árvores")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# =============================================================
# ETAPA 18 · Profundidade das árvores
# =============================================================
profundidades = [1, 2, 3, 4, 6, 8, 10, None]
resultados_d = []
for d in profundidades:
    a = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    f = RandomForestClassifier(n_estimators=500, max_depth=d, random_state=42,
                               n_jobs=-1).fit(X_train, y_train)
    for nome, m in [("Árvore", a), ("Floresta", f)]:
        resultados_d.append({"modelo": nome,
                             "max_depth": str(d),
                             "acc_treino": m.score(X_train, y_train),
                             "acc_teste": m.score(X_test, y_test)})
tabela_d = pd.DataFrame(resultados_d)
print(tabela_d.to_string(index=False))

plt.figure(figsize=(9, 4.5))
for nome, estilo in [("Árvore", "--"), ("Floresta", "-")]:
    t = tabela_d[tabela_d["modelo"] == nome]
    plt.plot(t["max_depth"], t["acc_treino"], "o" + estilo, label=f"{nome} · treino")
    plt.plot(t["max_depth"], t["acc_teste"], "s" + estilo, label=f"{nome} · teste")
plt.xlabel("max_depth")
plt.ylabel("Acurácia")
plt.title("Treino × teste por profundidade")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
for ax, d in zip(axes, [2, 4, None]):
    f = RandomForestClassifier(n_estimators=500, max_depth=d, random_state=42,
                               n_jobs=-1).fit(X_train, y_train)
    plotar_fronteira(f, X_test, y_test, ax, f"Floresta · max_depth={d}")
plt.tight_layout()
plt.show()


# =============================================================
# ETAPA 19 · Tabela final
# =============================================================
arvore_rasa = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
floresta_rasa = RandomForestClassifier(n_estimators=500, max_depth=4, oob_score=True,
                                       random_state=42, n_jobs=-1).fit(X_train, y_train)

linhas = []
for nome, m, arv, prof in [("Árvore de Decisão", arvore, 1, "None"),
                           ("Árvore rasa", arvore_rasa, 1, "4"),
                           ("Random Forest", floresta, 500, "None"),
                           ("Random Forest rasa", floresta_rasa, 500, "4")]:
    linhas.append({"modelo": nome,
                   "árvores": arv,
                   "max_depth": prof,
                   "treino": round(m.score(X_train, y_train), 3),
                   "teste": round(m.score(X_test, y_test), 3),
                   "matriz (teste)": confusion_matrix(y_test, m.predict(X_test)).tolist(),
                   "OOB": round(m.oob_score_, 3) if arv > 1 else "n/a"})
tabela_final = pd.DataFrame(linhas)
print(tabela_final.to_string())

# custo de previsão
t0 = time.perf_counter(); arvore.predict(X_test); prev_arvore = time.perf_counter() - t0
t0 = time.perf_counter(); floresta.predict(X_test); prev_floresta = time.perf_counter() - t0
print(f"Treino:   árvore {tempo_arvore:.4f} s | floresta {tempo_floresta:.4f} s")
print(f"Previsão: árvore {prev_arvore:.4f} s | floresta {prev_floresta:.4f} s")


# =============================================================
# ETAPA 20 · Interpretação
# =============================================================
print(f"""
1. Floresta × árvore profunda no teste: {acc_teste_floresta:.3f} × {acc_teste_arvore:.3f}
   ({(acc_teste_floresta - acc_teste_arvore) * 150:.0f} pontos de teste; 1 ponto = 0,67 p.p.)
2. A árvore profunda acerta 100% do treino e cai no teste: decorou o ruído.
3. A árvore rasa (max_depth=4) chegou a {arvore_rasa.score(X_test, y_test):.3f} no teste.
4. A fronteira da árvore é em degraus, com ilhas; a da floresta é mais suave.
5. OOB = {floresta.oob_score_:.3f}, teste = {acc_teste_floresta:.3f}: o OOB estima o teste sem gastá-lo.
6. MDI e permutação concordam sobre qual característica pesa mais.
7. Custo: a floresta leva bem mais tempo para treinar e prever do que uma árvore.
""")
