
import numpy as np
import pandas as pd
import time

from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_regression
)

from sklearn.pipeline import Pipeline


# ============================================================
# 2. CARREGAMENTO DO DATASET
# ============================================================

X, y = fetch_california_housing(
    return_X_y=True,
    as_frame=True
)

print("=" * 60)
print("DATASET CALIFORNIA HOUSING")
print("=" * 60)

print("Formato original de X:", X.shape)
print("Formato de y:", y.shape)

print("\nFeatures originais:")
print(X.columns.tolist())


# ============================================================
# 3. CRIAÇÃO DAS FEATURES DE RUÍDO
# ============================================================
#
# As 8 primeiras features são reais.
# Aqui adicionamos 8 features aleatórias para testar
# se o método de seleção consegue identificar e remover
# atributos sem informação útil.
#
# Total:
# 8 features reais + 8 features de ruído = 16 features
# ============================================================

rng = np.random.default_rng(42)

for i in range(8):
    X[f"ruido_{i}"] = rng.normal(size=len(X))


print("\nFeatures após adicionar ruído:")
print(X.columns.tolist())

print("\nNúmero total de features:", X.shape[1])


# ============================================================
# 4. DIVISÃO ENTRE TREINO E TESTE
# ============================================================
#
# O teste fica "trancado" durante todo o processo de seleção.
# Ele só será utilizado no final.
# ============================================================

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n" + "=" * 60)
print("DIVISÃO DOS DADOS")
print("=" * 60)

print("Treino:", X_tr.shape)
print("Teste:", X_te.shape)


# ============================================================
# 5. VALIDAÇÃO CRUZADA
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 6. MODELO
# ============================================================
#
# O mesmo algoritmo será utilizado nos dois modelos:
#
# - antes da seleção
# - depois da seleção
#
# Isso garante uma comparação justa.
# ============================================================

modelo = HistGradientBoostingRegressor(
    random_state=42
)


# ============================================================
# 7. FUNÇÃO PARA AVALIAR UM MODELO
# ============================================================

def avaliar(estimator):
    inicio = time.perf_counter()

    scores = cross_val_score(
        estimator,
        X_tr,
        y_tr,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )

    tempo = time.perf_counter() - inicio

    return (
        scores.mean(),
        scores.std(),
        tempo
    )


# ============================================================
# 8. BASELINE — MODELO COM TODAS AS FEATURES
# ============================================================
#
# Aqui utilizamos todas as 16 features:
#
# 8 reais
# +
# 8 de ruído
#
# Esse resultado será nossa referência.
# ============================================================

baseline = avaliar(modelo)

r2_baseline = baseline[0]
desvio_baseline = baseline[1]
tempo_baseline = baseline[2]

print("\n" + "=" * 60)
print("BASELINE — TODAS AS FEATURES")
print("=" * 60)

print(f"Número de features: {X_tr.shape[1]}")
print(f"R² médio: {r2_baseline:.4f}")
print(f"Desvio padrão: {desvio_baseline:.4f}")
print(f"Tempo: {tempo_baseline:.1f} segundos")


# ============================================================
# 9. DEFINIÇÃO DA FAIXA ACEITÁVEL
# ============================================================
#
#
# R² aceitável >= R² do baseline - 1 desvio padrão
#
# Assim conseguimos procurar uma solução com menos features
# sem aceitar uma perda muito grande de desempenho.
# ============================================================

limite = r2_baseline - desvio_baseline

print("\nFaixa aceitável:")
print(f"R² >= {limite:.4f}")


# ============================================================
# 10. SELEÇÃO DE FEATURES
# ============================================================
#
# Método:
#
# SelectKBest + Mutual Information
#
# Vamos testar diferentes quantidades de features.
# ============================================================

resultados_selecao = {}

print("\n" + "=" * 60)
print("SELEÇÃO DE FEATURES — MUTUAL INFORMATION")
print("=" * 60)

for k in [2, 4, 6, 8, 10, 12, 14, 16]:

    pipeline = Pipeline([
        (
            "selecao",
            SelectKBest(
                mutual_info_regression,
                k=k
            )
        ),
        (
            "modelo",
            HistGradientBoostingRegressor(
                random_state=42
            )
        )
    ])

    resultado = avaliar(pipeline)

    resultados_selecao[k] = resultado

    print(
        f"k={k:2d} | "
        f"R²={resultado[0]:.4f} ± {resultado[1]:.4f} | "
        f"tempo={resultado[2]:.1f}s"
    )


# ============================================================
# 11. ESCOLHA DO MENOR NÚMERO DE FEATURES ACEITÁVEL
# ============================================================
#
# Procuramos o menor k cujo R² esteja dentro da faixa
# aceitável definida anteriormente.
# ============================================================

k_aceitaveis = [
    k
    for k, resultado in resultados_selecao.items()
    if resultado[0] >= limite
]

if not k_aceitaveis:
    raise ValueError(
        "Nenhum valor de k ficou dentro da faixa aceitável."
    )

k_final = min(k_aceitaveis)

r2_selecionado = resultados_selecao[k_final][0]
desvio_selecionado = resultados_selecao[k_final][1]
tempo_selecionado = resultados_selecao[k_final][2]

print("\n" + "=" * 60)
print("RESULTADO DA SELEÇÃO")
print("=" * 60)

print(f"Baseline: {X_tr.shape[1]} features")
print(f"Features selecionadas: {k_final}")
print(f"R² baseline: {r2_baseline:.4f}")
print(f"R² selecionado: {r2_selecionado:.4f}")
print(f"Limite aceitável: {limite:.4f}")


# ============================================================
# 12. TREINAR O SELETOR PARA DESCOBRIR AS FEATURES
# ============================================================

seletor = SelectKBest(
    mutual_info_regression,
    k=k_final
)

seletor.fit(X_tr, y_tr)


# ============================================================
# 13. FEATURES MANTIDAS E REMOVIDAS
# ============================================================

features_mantidas = X_tr.columns[
    seletor.get_support()
].tolist()

features_removidas = X_tr.columns[
    ~seletor.get_support()
].tolist()


print("\n" + "=" * 60)
print("FEATURES MANTIDAS")
print("=" * 60)

for feature in features_mantidas:
    print("-", feature)


print("\n" + "=" * 60)
print("FEATURES REMOVIDAS")
print("=" * 60)

for feature in features_removidas:
    print("-", feature)


# ============================================================
# 14. RANKING DAS FEATURES
# ============================================================
#
# A informação mútua fornece um score para cada feature.
#
# Quanto maior o score:
# maior a relação/dependência entre a feature e o alvo.
# ============================================================

ranking = pd.DataFrame({
    "Feature": X_tr.columns,
    "Score_MI": seletor.scores_
})

ranking = ranking.sort_values(
    "Score_MI",
    ascending=False
)

print("\n" + "=" * 60)
print("RANKING DAS FEATURES — MUTUAL INFORMATION")
print("=" * 60)

print(
    ranking.to_string(
        index=False,
        formatters={
            "Score_MI": "{:.4f}".format
        }
    )
)


# ============================================================
# 15. IDENTIFICAR O QUE ACONTECEU COM AS FEATURES DE RUÍDO
# ============================================================

ruidos_mantidos = [
    feature
    for feature in features_mantidas
    if feature.startswith("ruido")
]

ruidos_removidos = [
    feature
    for feature in features_removidas
    if feature.startswith("ruido")
]

print("\n" + "=" * 60)
print("ANÁLISE DAS FEATURES DE RUÍDO")
print("=" * 60)

print("Ruídos mantidos:", ruidos_mantidos)
print("Ruídos removidos:", ruidos_removidos)


# ============================================================
# 16. CONSTRUIR O PIPELINE FINAL
# ============================================================
#
# O Pipeline realiza:
#
# 1. seleção das melhores features
# 2. treinamento do modelo
#
# A seleção continua dentro do Pipeline para evitar
# vazamento de informação durante a validação.
# ============================================================

pipeline_final = Pipeline([
    (
        "selecao",
        SelectKBest(
            mutual_info_regression,
            k=k_final
        )
    ),
    (
        "modelo",
        HistGradientBoostingRegressor(
            random_state=42
        )
    )
])


# ============================================================
# 17. TREINAMENTO FINAL
# ============================================================

pipeline_final.fit(
    X_tr,
    y_tr
)


# ============================================================
# 18. AVALIAÇÃO NO TESTE
# ============================================================
#
# O conjunto de teste é utilizado somente agora.
# ============================================================

r2_teste_final = pipeline_final.score(
    X_te,
    y_te
)


# Também avaliamos o baseline no teste para comparação.
modelo_baseline_final = HistGradientBoostingRegressor(
    random_state=42
)

modelo_baseline_final.fit(
    X_tr,
    y_tr
)

r2_teste_baseline = modelo_baseline_final.score(
    X_te,
    y_te
)


# ============================================================
# 19. COMPARAÇÃO FINAL
# ============================================================

print("\n" + "=" * 60)
print("COMPARAÇÃO FINAL")
print("=" * 60)

print("\nMODELO ANTES DA SELEÇÃO")
print(f"Features: {X_tr.shape[1]}")
print(f"R² CV: {r2_baseline:.4f} ± {desvio_baseline:.4f}")
print(f"R² teste: {r2_teste_baseline:.4f}")

print("\nMODELO DEPOIS DA SELEÇÃO")
print(f"Features: {k_final}")
print(f"R² CV: {r2_selecionado:.4f} ± {desvio_selecionado:.4f}")
print(f"R² teste: {r2_teste_final:.4f}")


# ============================================================
# 20. DIFERENÇA DE DESEMPENHO
# ============================================================

ganho_cv = r2_selecionado - r2_baseline
ganho_teste = r2_teste_final - r2_teste_baseline

reducao_features = X_tr.shape[1] - k_final

print("\n" + "=" * 60)
print("INSIGHTS")
print("=" * 60)

print(
    f"Redução no número de features: "
    f"{X_tr.shape[1]} → {k_final}"
)

print(
    f"Features removidas: "
    f"{reducao_features}"
)

print(
    f"Variação do R² na validação cruzada: "
    f"{ganho_cv:+.4f}"
)

print(
    f"Variação do R² no teste: "
    f"{ganho_teste:+.4f}"
)


# ============================================================
# 21. CONCLUSÃO AUTOMÁTICA
# ============================================================

print("\n" + "=" * 60)
print("CONCLUSÃO")
print("=" * 60)

if ganho_teste >= 0:
    print(
        "A seleção de features foi vantajosa ou, no mínimo, "
        "não prejudicou o desempenho no conjunto de teste."
    )
else:
    print(
        "A seleção de features reduziu o desempenho no conjunto "
        "de teste."
    )

print(
    f"O modelo passou de {X_tr.shape[1]} para {k_final} features, "
    f"mantendo um R² de teste de {r2_teste_final:.4f}."
)

print(
    "As features selecionadas foram escolhidas com base no "
    "método Mutual Information."
)