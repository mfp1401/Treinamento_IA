# Treinamento IA

Repositório com os estudos, exercícios e desafios do treinamento de Inteligência Artificial do Inatel.
O conteúdo segue a ordem das aulas: dos fundamentos de Python à implementação de algoritmos de Machine Learning do zero, com NumPy.

---

## 📚 Aulas

| Aula | Tema | Conteúdo |
|---|---|---|
| [Aula 2](Aula2/README.md) | Introdução ao Python | variáveis, condicionais, listas, dicionários, `for` e `while` |
| [Aula 3](Aula3/README.md) | NumPy, Pandas e Matplotlib | arrays, DataFrames, gráficos e análise exploratória |
| [Aula 5](Aula5/README.md) | Machine Learning com scikit-learn | KNN, pipelines, feature engineering e seleção de features |
| [Aula 6](Aula6/README.md) | ML na mão, com NumPy | álgebra linear, regressão, gradiente, logística, métricas, validação cruzada, KNN, Naive Bayes e árvore de decisão |
| [Aula 7](Aula7/README.md) | Modelos avançados | regularização (Ridge, Lasso e ElasticNet) |
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
│   ├── Aula6.1/ ... Aula6.9/   # cada pasta: exercicio_guiado.py + desafio.py + README.md
├── Aula7/
│   └── Aula7.1/                # regularização
├── Laboratorios/
├── requirements.txt
└── README.md
```

Em cada aula, `exercicio_guiado.py` é o código construído em sala, e `desafio.py` é o desafio de casa.

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
```

---

## 🛠️ Tecnologias

Python · NumPy · Pandas · Matplotlib · scikit-learn · SciPy · Joblib
