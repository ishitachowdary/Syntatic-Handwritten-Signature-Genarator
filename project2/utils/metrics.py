import numpy as np
from sklearn.metrics import roc_curve

def compute_far_frr_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.abs(fpr - fnr))]
    return fpr.mean(), fnr.mean(), eer
