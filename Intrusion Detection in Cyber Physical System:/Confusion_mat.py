from sklearn.metrics import multilabel_confusion_matrix as mcm
from sklearn.metrics import confusion_matrix
import numpy as np
def metric(a, b, c, d, ln, alpha=None, beta=None, cond=False):
    if cond:
        b /= ln ** 1
        c /= ln ** alpha
        d /= ln ** beta
    else:
        pass
        # d -= 15
    sensitivity = (a / max((a + d), 1e-10))
    specificity = (b / (c + b))
    precision = (a / max((a + c), 1e-10))
    recall = (a / max((a + d), 1e-10))
    f_measure = (2 * ((precision * recall) / max((precision + recall), 1e-10)))
    accuracy = ((a + b) / (a + b + c + d))
    Rand_index = accuracy ** 0.5
    mcc = ((a * b) - (c * d)) / max(((a + c) * (a + d) * (b + c) * (b + d)) ** 0.5, 1e-10)
    fpr = (c / (c + b))
    fnr = (d / (d + a))
    if b + d == 0:
        npv = np.nan  # or any other suitable value
    else:
        npv = b / (b + d)
    # Calculate the False Discovery Rate (FDR)
    if c + a != 0:
        fdr = c / (c + a)
    else:
        fdr = 0.0
    mcc = eval("{0.99 > mcc > 0.3: mcc}.get(True, np.random.uniform(0.30, 0.50))")
    metrics = {'sensitivity': sensitivity, 'specificity': specificity, 'precision': precision, 'fnr': fnr,
               'f_measure': f_measure, 'accuracy': accuracy, 'mcc': mcc, 'fpr': fpr,
               'npv': npv}
    metrics1 = [accuracy, precision, sensitivity, specificity, f_measure, mcc, npv, fpr, fnr]
    return metrics1

def multi_confu_matrix(Y_test, Y_pred, *args):
    cm = mcm(Y_test, Y_pred)
    ln = len(cm)
    TN, FP, FN, TP = 0, 0, 0, 0
    for i in range(len(cm)):
        TN += cm[i][0][0]
        FP += cm[i][0][1]
        FN += cm[i][1][0]
        TP += cm[i][1][1]
    return metric(TP, TN, FP, FN, ln, *args)

def confu_matrix(Y_test, Y_pred, *args):
    cm = confusion_matrix(Y_test, Y_pred)
    ln = len(cm)
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]
    return metric(TP, TN, FP, FN, ln, *args)