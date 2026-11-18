"""Candidate and trip-level model evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


def candidate_metrics(y_true, y_pred, y_prob=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
        except ValueError:
            metrics["log_loss"] = float("nan")
    return metrics


def positive_class_probability(model, X) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    if 1 in classes:
        return probabilities[:, int(np.where(classes == 1)[0][0])]
    if probabilities.shape[1] == 1:
        return np.zeros(len(probabilities))
    raise ValueError("The model probability interface has no positive (arc match) class")
