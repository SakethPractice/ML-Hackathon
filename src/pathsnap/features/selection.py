"""In-fold feature-selection helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.inspection import permutation_importance


def variance_selected(X: pd.DataFrame, threshold: float = 0.0) -> list[str]:
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X.fillna(X.median(numeric_only=True)))
    return X.columns[selector.get_support()].tolist()


def correlation_selected(X: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = {column for column in upper.columns if any(upper[column] > threshold)}
    return [column for column in X.columns if column not in drop]


def kbest_selected(X: pd.DataFrame, y: pd.Series, k: int = 10) -> list[str]:
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X.fillna(X.median(numeric_only=True)), y)
    return X.columns[selector.get_support()].tolist()


def permutation_selected(model, X: pd.DataFrame, y: pd.Series, repeats: int = 5,
                         random_seed: int = 42) -> pd.DataFrame:
    result = permutation_importance(model, X, y, scoring="f1_macro", n_repeats=repeats,
                                    random_state=random_seed, n_jobs=-1)
    out = pd.DataFrame({"feature": X.columns,
                        "mean_importance": result.importances_mean,
                        "std_importance": result.importances_std})
    return out.sort_values("mean_importance", ascending=False).reset_index(drop=True)
