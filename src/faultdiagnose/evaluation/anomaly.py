from __future__ import annotations

from sklearn.metrics import roc_auc_score


def compute_auc(scores, labels) -> float:
    return float(roc_auc_score(labels, scores))
