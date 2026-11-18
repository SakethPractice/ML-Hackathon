"""Run training-only feature importance and feature-selection analysis."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

from scripts.run_feature_analysis import build_training_data
from pathsnap.features.selection import correlation_selected, kbest_selected, permutation_selected, variance_selected


RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SEED = 42


def main() -> None:
    data, X, y, _ = build_training_data()
    groups = data["trip_id"]
    splitter = GroupKFold(n_splits=5)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                          n_jobs=-1, random_state=SEED)),
    ])
    rf.fit(X_train, y_train)
    rf_importances = rf.named_steps["model"].feature_importances_

    rows = [{"feature": name, "random_forest": float(rf_importances[i])}
            for i, name in enumerate(X.columns)]
    importance = pd.DataFrame(rows)

    if LGBMClassifier is not None:
        lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              class_weight="balanced", n_jobs=-1, random_state=SEED, verbosity=-1)
        lgbm.fit(X_train.fillna(X_train.median()), y_train)
        importance["lightgbm_gain"] = importance["feature"].map(
            dict(zip(X.columns, lgbm.booster_.feature_importance(importance_type="gain")))
        )
        importance["lightgbm_split"] = importance["feature"].map(
            dict(zip(X.columns, lgbm.booster_.feature_importance(importance_type="split")))
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        ordered = importance.sort_values("lightgbm_gain", ascending=True)
        ax.barh(ordered["feature"], ordered["lightgbm_gain"])
        ax.set_title("LightGBM gain importance")
        fig.tight_layout()
        fig.savefig(FIGURES / "lightgbm_gain_importance.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 7))
        ordered = importance.sort_values("lightgbm_split", ascending=True)
        ax.barh(ordered["feature"], ordered["lightgbm_split"])
        ax.set_title("LightGBM split importance")
        fig.tight_layout()
        fig.savefig(FIGURES / "lightgbm_split_importance.png", dpi=150)
        plt.close(fig)

    perm = permutation_selected(rf, X_test, y_test, repeats=5, random_seed=SEED)
    perm.to_csv(RESULTS / "permutation_importance.csv", index=False)
    perm_rank = perm.set_index("feature")["mean_importance"]

    fig, ax = plt.subplots(figsize=(10, 7))
    ordered = perm.sort_values("mean_importance", ascending=True)
    ax.barh(ordered["feature"], ordered["mean_importance"], xerr=ordered["std_importance"])
    ax.set_title("Held-out permutation importance (Macro F1)")
    fig.tight_layout()
    fig.savefig(FIGURES / "permutation_importance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    ordered = importance.sort_values("random_forest", ascending=True)
    ax.barh(ordered["feature"], ordered["random_forest"])
    ax.set_title("Random Forest impurity importance")
    fig.tight_layout()
    fig.savefig(FIGURES / "random_forest_importance.png", dpi=150)
    plt.close(fig)

    variance = variance_selected(X_train)
    correlated = correlation_selected(X_train, threshold=0.95)
    kbest = kbest_selected(X_train, y_train, k=10)
    rf_top = importance.sort_values("random_forest", ascending=False)["feature"].head(10).tolist()
    if "lightgbm_gain" in importance:
        lgbm_top = importance.sort_values("lightgbm_gain", ascending=False)["feature"].head(10).tolist()
    else:
        lgbm_top = []
    perm_top = perm.head(10)["feature"].tolist()
    chosen = list(dict.fromkeys(rf_top[:8] + lgbm_top[:8] + perm_top[:8]))

    results = pd.DataFrame([{
        "method": "variance_threshold", "selected_features": ";".join(variance)
    }, {"method": "correlation_filter_0.95", "selected_features": ";".join(correlated)},
       {"method": "select_k_best_10", "selected_features": ";".join(kbest)},
       {"method": "random_forest_top10", "selected_features": ";".join(rf_top)},
       {"method": "lightgbm_gain_top10", "selected_features": ";".join(lgbm_top)},
       {"method": "permutation_top10", "selected_features": ";".join(perm_top)},
       {"method": "training_selected_subset", "selected_features": ";".join(chosen)}])
    results.to_csv(RESULTS / "feature_selection_results.csv", index=False)
    (RESULTS / "selected_features.json").write_text(
        json.dumps({"selected_features": chosen, "method": "union of training-fold RF, LightGBM and permutation top rankings",
                    "seed": SEED, "note": "Choose E8 from cross-validation; this file is a reproducible candidate subset."},
                   indent=2), encoding="utf-8")
    importance.to_csv(RESULTS / "feature_importance.csv", index=False)
    print(json.dumps({"selected_features": chosen, "rf_top10": rf_top, "lgbm_top10": lgbm_top,
                      "permutation_top10": perm_top}, indent=2))


if __name__ == "__main__":
    main()
