import numpy as np
from sklearn.metrics import classification_report   


def meu_classification_report(y_real, y_pred, nomes):
    # 1. matriz de confusão 
    classes = np.unique(np.concatenate([y_real, y_pred]))
    k = len(classes)
    M = np.zeros((k, k), dtype=int)
    for real, prev in zip(y_real, y_pred):
        M[real, prev] += 1

    # 2. TP, FP, FN e support 
    tp = np.diag(M)
    fp = M.sum(axis=0) - tp
    fn = M.sum(axis=1) - tp
    support = M.sum(axis=1)

    # 3. métricas
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)
    acc = tp.sum() / M.sum()

    # 4. relatório
    print(M)
    print(f"{'':>14}{'precision':>11}{'recall':>9}{'f1-score':>10}{'support':>9}")
    for i in range(k):
        print(f"{nomes[i]:>14}{prec[i]:>11.2f}{rec[i]:>9.2f}{f1[i]:>10.2f}{support[i]:>9}")
    print(f"{'accuracy':>14}{'':>20}{acc:>10.2f}{support.sum():>9}")
    print(f"{'macro avg':>14}{prec.mean():>11.2f}{rec.mean():>9.2f}{f1.mean():>10.2f}{support.sum():>9}")
    print(f"{'weighted avg':>14}"
          f"{np.average(prec, weights=support):>11.2f}"
          f"{np.average(rec, weights=support):>9.2f}"
          f"{np.average(f1, weights=support):>10.2f}{support.sum():>9}")


# Extensão: curva ROC e AUC 
def curva_roc(y_real, scores):
    ordem = np.argsort(-scores, kind="mergesort")
    y, s = y_real[ordem], scores[ordem]
    fim = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tp = np.cumsum(y)[fim]
    fp = (fim + 1) - tp
    P, N = y.sum(), len(y) - y.sum()
    tpr = np.r_[0, tp] / P
    fpr = np.r_[0, fp] / N
    return fpr, tpr


def auc(fpr, tpr):
    larg = fpr[1:] - fpr[:-1]
    alt = (tpr[1:] + tpr[:-1]) / 2
    return np.sum(larg * alt)


# Binário: spam
y_real = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
y_pred = np.array([1, 0, 1, 1, 1, 0, 0, 0, 1, 1])
meu_classification_report(y_real, y_pred, ["normal", "spam"])
print(classification_report(y_real, y_pred, target_names=["normal", "spam"]))

# Três classes: pinguins
y_real3 = np.array([0]*10 + [1]*5 + [2]*10)
y_pred3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                    0, 1, 1, 1, 2,
                    1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
meu_classification_report(y_real3, y_pred3, ["Adelie", "Chinstrap", "Gentoo"])
print(classification_report(y_real3, y_pred3, target_names=["Adelie", "Chinstrap", "Gentoo"]))

# ROC e AUC
scores = np.array([0.95, 0.10, 0.70, 0.55, 0.85, 0.38, 0.45, 0.25, 0.62, 0.78])
fpr, tpr = curva_roc(y_real, scores)
print("FPR:", fpr)
print("TPR:", tpr)
print("AUC:", auc(fpr, tpr))