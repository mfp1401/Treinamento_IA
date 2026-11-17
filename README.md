# Treinamento IA

Repositório com os estudos, exercícios e desafios do treinamento de Inteligência Artificial do Inatel.
O conteúdo segue a ordem das aulas: dos fundamentos de Python à implementação de algoritmos de Machine Learning do zero, com NumPy, passando pelos algoritmos modernos (SVM, Random Forest e Boosting), pela otimização de hiperparâmetros e pelo AutoML.

---

## 📚 Aulas

| Aula | Tema | Conteúdo |
|---|---|---|
| [Aula 2](Aula2/README.md) | Introdução ao Python | variáveis, condicionais, listas, dicionários, `for` e `while` |
| [Aula 3](Aula3/README.md) | NumPy, Pandas e Matplotlib | arrays, DataFrames, gráficos e análise exploratória |
| [Aula 5](Aula5/README.md) | Machine Learning com scikit-learn | KNN, pipelines, feature engineering e seleção de features |
| [Aula 6](Aula6/README.md) | ML na mão, com NumPy | álgebra linear, regressão, gradiente, logística, métricas, validação cruzada, KNN, Naive Bayes, árvore de decisão, K-Means, PCA e redes neurais |
| [Aula 7](Aula7/README.md) | Algoritmos modernos e ajuste fino | regularização (Ridge, Lasso e ElasticNet), SVM com kernel, Random Forest, Boosting (XGBoost, LightGBM, CatBoost) e otimização de hiperparâmetros (Grid, Random e Optuna) |
| [Aula 8](Aula8/README.md) | AutoML | comparação automática de modelos com LazyPredict |
| [Laboratórios](Laboratorios/README.md) | Projetos de ponta a ponta | Lab 1: classificação de pinguins |

---

## 📂 Estrutura do projeto

```text
Treinamento_IA/
├── Aula2/                      # fundamentos de Python
├── Aula3/                      # NumPy, Pandas e Matplotlib
├── Aula5/
│   ├── Aula5.1/                # KNN e baseline
│   ├── Aula5.3/                # pipelines (Titanic e Adult Income)
│   ├── Aula5.4/                # feature engineering
│   ├── Aula5.5/                # seleção de features
│   └── revisao_aula5.py
├── Aula6/
│   ├── Aula6.1/ ... Aula6.12/  # cada pasta: exercício guiado + desafio + README.md
├── Aula7/
│   ├── Aula7.1/                # regularização
│   ├── Aula7.2/                # SVM e o truque do kernel
│   ├── Aula7.3/                # Random Forest
│   ├── Aula7.4/                # Boosting: XGBoost, LightGBM e CatBoost
│   └── Aula7.5/                # otimização de hiperparâmetros: Grid, Random e Optuna
├── Aula8/
│   ├── Aula8.1/                # introdução ao AutoML
│   └── Aula8.2/                # LazyPredict: comparação automática de modelos
├── Laboratorios/
├── requirements.txt
└── README.md
```

Em cada aula, `exercicio_guiado` é o código construído em sala e `desafio` é o desafio de casa. Até a Aula 7.3 os arquivos são `.py`. A partir da 7.4, alguns são notebooks (`.ipynb`), já executados, com as saídas e os gráficos.

---

## ▶️ Como executar

```bash
# 1. criar e ativar o ambiente virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. instalar as bibliotecas
pip install -r requirements.txt

# 3. rodar um exercício
python "Aula6/Aula6.9/desafio.py"

# 4. abrir um notebook (ou abrir o .ipynb direto no VS Code)
jupyter notebook "Aula7/Aula7.5/desafio.ipynb"
```

---

## 🛠️ Tecnologias

Python · NumPy · Pandas · Matplotlib · scikit-learn · SciPy · Joblib · XGBoost · LightGBM · CatBoost · Optuna · LazyPredict