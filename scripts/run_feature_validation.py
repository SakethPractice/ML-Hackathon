"""Run GroupKFold validation and feature ablations for all five official models."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from time import perf_counter
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import sklearn
import csv
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GroupKFold

from scripts.run_feature_analysis import build_training_data
from pathsnap.features.validation import ablation_configs, model_definitions, observation_top1


RESULTS = ROOT / "results"
SEED = 42
N_FOLD = 5
FOLD_FILE = RESULTS / "feature_validation_results.csv"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    data, all_X, y, _ = build_training_data()
    selected_path = RESULTS / "selected_features.json"
    selected = all_X.columns.tolist()
    if selected_path.exists():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))["selected_features"]

    configs = ablation_configs(selected)
    configs["E8_selected"] = [name for name in selected if name in all_X.columns]

    groups = data["trip_id"].to_numpy()
    splitter = GroupKFold(n_splits=N_FOLD)
    rows = []
    completed = set()
    if FOLD_FILE.exists():
        existing = pd.read_csv(FOLD_FILE)
        rows = existing.to_dict("records")
        completed = set(
            (r["model"], r["feature_configuration"], int(r["fold"]))
            for r in existing.to_dict("records")
        )
    fold_summary = []

    for config_name, feature_names in configs.items():
        X = all_X[feature_names]
        print(f"Starting {config_name} with {len(feature_names)} features", flush=True)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            for model_name, model in model_definitions(SEED).items():
                key = (model_name, config_name, fold)
                if key in completed:
                    continue
                print(f"  fitting {config_name} fold {fold}: {model_name}", flush=True)
                start = perf_counter()
                model.fit(X_train, y_train)
                train_time = perf_counter() - start

                start = perf_counter()
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)
                inference_time = perf_counter() - start
                classes = np.asarray(model.classes_)
                if 1 not in classes:
                    logloss = float("nan")
                else:
                    logloss = float(log_loss(y_test, y_prob, labels=classes.tolist()))
                if 1 not in y_test.unique():
                    logloss = float("nan")
                top1 = observation_top1(
                    model, X_test, data.iloc[test_idx]["observation_id"].to_numpy(),
                    data.iloc[test_idx]["arc_id"].to_numpy(), y_test.to_numpy()
                )
                row = {
                    "model": model_name,
                    "feature_configuration": config_name,
                    "selected_features": ";".join(feature_names),
                    "n_features": len(feature_names),
                    "fold": fold,
                    "cv_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                    "cv_top1_accuracy": top1,
                    "cv_log_loss": logloss,
                    "training_time_s": train_time,
                    "inference_time_s": inference_time,
                    "seed": SEED,
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(FOLD_FILE, index=False)

            fold_summary.append({
                "configuration": config_name,
                "fold": fold,
                "train_trips": int(len(set(groups[train_idx]))),
                "test_trips": int(len(set(groups[test_idx]))),
            })

    fold_results = pd.DataFrame(rows)
    fold_results.to_csv(RESULTS / "feature_validation_results.csv", index=False)
    summary = fold_results.groupby(["model", "feature_configuration"]).agg(
        cv_macro_f1_mean=("cv_macro_f1", "mean"),
        cv_macro_f1_std=("cv_macro_f1", "std"),
        cv_top1_accuracy_mean=("cv_top1_accuracy", "mean"),
        cv_top1_accuracy_std=("cv_top1_accuracy", "std"),
        cv_log_loss_mean=("cv_log_loss", "mean"),
        cv_log_loss_std=("cv_log_loss", "std"),
        training_time_s_mean=("training_time_s", "mean"),
        inference_time_s_mean=("inference_time_s", "mean"),
        n_features=("n_features", "first"),
    ).reset_index()
    summary.to_csv(RESULTS / "feature_ablation_results.csv", index=False)

    environment = {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "seed": SEED,
        "partition": "train",
        "trip_count": int(data["trip_id"].nunique()),
        "cv": f"GroupKFold(n_splits={N_FOLD}, groups=trip_id)",
        "lightgbm_available": True,
    }
    try:
        import lightgbm
        environment["lightgbm"] = lightgbm.__version__
    except ImportError:
        environment["lightgbm_available"] = False
    (RESULTS / "experiment_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(environment, indent=2))


if __name__ == "__main__":
    main()
