"""Run the approved five-model PathSnap experiment pipeline.

The runner fits only on the training partition.  Five-fold GroupKFold keeps
all candidate rows from a trip together; the student-test partition is used
only for the final development comparison.  It writes fold metrics, a compact
model comparison, probability outputs for decoder experiments, and reloadable
fitted pipelines.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pathsnap.features import get_feature_columns
from pathsnap.models.evaluate import positive_class_probability
from pathsnap.models.train import make_model_definitions
from scripts.run_feature_analysis import build_labeled_data


SEED = 42
N_SPLITS = 5
KNN_MAX_FIT_ROWS = 5_000
RESULTS = ROOT / "results"
PREDICTIONS = RESULTS / "predictions"
ARTIFACTS = ROOT / "artifacts" / "models"


def _observation_top1(frame: pd.DataFrame, probabilities: np.ndarray) -> float:
    """Top-1 accuracy grouped by the trip-local observation identity."""
    ranked = frame[["trip_id", "observation_id", "arc_id", "is_reference"]].copy()
    ranked["probability"] = probabilities
    top = ranked.loc[ranked.groupby(["trip_id", "observation_id"], sort=False)["probability"].idxmax()]
    return float(top["is_reference"].mean())


def _metrics(frame: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "candidate_accuracy": float(accuracy_score(y_true, y_pred)),
        "candidate_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "candidate_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "candidate_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "candidate_log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "observation_top1_accuracy": _observation_top1(frame, probability),
    }


def _prediction_table(frame: pd.DataFrame, y_true: pd.Series, probability: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    output = frame[["trip_id", "observation_id", "arc_id"]].copy()
    output["is_reference"] = y_true.to_numpy(dtype=np.int8)
    output["match_probability"] = probability
    output["candidate_prediction"] = predicted.astype(np.int8)
    output["probability_rank"] = output.groupby(["trip_id", "observation_id"])["match_probability"].rank(
        method="first", ascending=False
    ).astype(np.int16)
    return output.sort_values(["trip_id", "observation_id", "probability_rank"], kind="mergesort")


def _fit_indices(model_name: str, y: pd.Series, seed: int) -> np.ndarray:
    """Return deterministic fit rows, limiting only the costly KNN baseline.

    KNN scoring remains against every validation/student candidate.  Its
    fitting subset preserves the candidate-level class share and is recorded
    in every output so this computational trade-off is never hidden.
    """
    indices = np.arange(len(y))
    if model_name != "knn" or len(indices) <= KNN_MAX_FIT_ROWS:
        return indices
    rng = np.random.default_rng(seed)
    positive = indices[y.to_numpy() == 1]
    negative = indices[y.to_numpy() == 0]
    n_positive = round(KNN_MAX_FIT_ROWS * len(positive) / len(indices))
    n_negative = KNN_MAX_FIT_ROWS - n_positive
    return np.sort(np.concatenate((
        rng.choice(positive, size=n_positive, replace=False),
        rng.choice(negative, size=n_negative, replace=False),
    )))


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("Loading labelled training candidates…", flush=True)
    train_data, X_train, y_train = build_labeled_data("train")
    print("Loading labelled student-test candidates…", flush=True)
    student_data, X_student, y_student = build_labeled_data("student_test")
    features = get_feature_columns()
    X_train, X_student = X_train[features], X_student[features]
    groups = train_data["trip_id"].astype(str).to_numpy()
    models = make_model_definitions(random_seed=SEED)

    fold_rows: list[dict] = []
    splitter = GroupKFold(n_splits=N_SPLITS)
    for model_name, prototype in models.items():
        print(f"Cross-validating {model_name}…", flush=True)
        for fold, (fit_idx, valid_idx) in enumerate(splitter.split(X_train, y_train, groups), start=1):
            model = clone(prototype)
            fit_rows = _fit_indices(model_name, y_train.iloc[fit_idx], SEED + fold)
            started = perf_counter()
            model.fit(X_train.iloc[fit_idx].iloc[fit_rows], y_train.iloc[fit_idx].iloc[fit_rows])
            train_seconds = perf_counter() - started
            started = perf_counter()
            probability = positive_class_probability(model, X_train.iloc[valid_idx])
            predicted = model.predict(X_train.iloc[valid_idx])
            inference_seconds = perf_counter() - started
            row = {
                "model": model_name,
                "partition": "train_cv",
                "fold": fold,
                "train_trip_count": int(pd.Series(groups[fit_idx]).nunique()),
                "validation_trip_count": int(pd.Series(groups[valid_idx]).nunique()),
                "candidate_rows": int(len(valid_idx)),
                "fit_candidate_rows": int(len(fit_rows)),
                "fit_sampling": "stratified_5k" if model_name == "knn" else "all_rows",
                "training_time_s": train_seconds,
                "inference_time_s": inference_seconds,
                "seed": SEED,
            }
            row.update(_metrics(train_data.iloc[valid_idx], y_train.iloc[valid_idx], predicted, probability))
            fold_rows.append(row)

    fold_results = pd.DataFrame(fold_rows)
    fold_results.to_csv(RESULTS / "fold_results.csv", index=False)

    summary_rows: list[dict] = []
    for model_name, prototype in models.items():
        print(f"Fitting {model_name} on all training trips…", flush=True)
        model = clone(prototype)
        fit_rows = _fit_indices(model_name, y_train, SEED)
        started = perf_counter()
        model.fit(X_train.iloc[fit_rows], y_train.iloc[fit_rows])
        fit_seconds = perf_counter() - started
        started = perf_counter()
        probability = positive_class_probability(model, X_student)
        predicted = model.predict(X_student)
        student_inference_seconds = perf_counter() - started
        student_metrics = _metrics(student_data, y_student, predicted, probability)
        _prediction_table(student_data, y_student, probability, predicted).to_csv(
            PREDICTIONS / f"student_test_{model_name}.csv.gz", index=False, compression="gzip"
        )
        artifact_path = ARTIFACTS / f"{model_name}.joblib"
        joblib.dump(model, artifact_path)
        cv = fold_results.loc[fold_results["model"] == model_name]
        row = {
            "model": model_name,
            "feature_count": len(features),
            "cv_folds": N_SPLITS,
            "cv_candidate_macro_f1_mean": float(cv["candidate_macro_f1"].mean()),
            "cv_candidate_macro_f1_std": float(cv["candidate_macro_f1"].std(ddof=0)),
            "cv_candidate_log_loss_mean": float(cv["candidate_log_loss"].mean()),
            "cv_observation_top1_accuracy_mean": float(cv["observation_top1_accuracy"].mean()),
            "cv_observation_top1_accuracy_std": float(cv["observation_top1_accuracy"].std(ddof=0)),
            "full_training_time_s": fit_seconds,
            "full_fit_candidate_rows": int(len(fit_rows)),
            "fit_sampling": "stratified_5k" if model_name == "knn" else "all_rows",
            "student_test_inference_time_s": student_inference_seconds,
            "student_test_candidate_rows": int(len(student_data)),
            "student_test_trip_count": int(student_data["trip_id"].nunique()),
            "artifact_path": str(artifact_path.relative_to(ROOT)).replace("\\", "/"),
        }
        row.update({f"student_test_{key}": value for key, value in student_metrics.items()})
        summary_rows.append(row)

    experiments = pd.DataFrame(summary_rows).sort_values(
        ["cv_observation_top1_accuracy_mean", "cv_candidate_macro_f1_mean", "cv_candidate_log_loss_mean"],
        ascending=[False, False, True], kind="mergesort"
    ).reset_index(drop=True)
    experiments.to_csv(RESULTS / "experiment_results.csv", index=False)
    selected = experiments.iloc[0]
    metadata = {
        "experiment_id": "five_model_baseline_v1",
        "selection_rule": "highest GroupKFold CV observation Top-1 accuracy; ties use CV Macro F1 then lower CV log loss",
        "selected_model": selected["model"],
        "feature_columns": features,
        "class_convention": {"0": "non-reference candidate", "1": "reference candidate"},
        "training_partition": "train",
        "student_evaluation_partition": "student_test",
        "training_trip_count": int(train_data["trip_id"].nunique()),
        "student_test_trip_count": int(student_data["trip_id"].nunique()),
        "cv": f"GroupKFold(n_splits={N_SPLITS}, groups=trip_id)",
        "random_seed": SEED,
        "knn_fit_sampling": "deterministic class-stratified sample of at most 5,000 training candidate rows; all validation and student-test candidates are scored",
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }
    try:
        import lightgbm
        metadata["lightgbm"] = lightgbm.__version__
    except ImportError:  # protected by make_model_definitions; useful if code changes later
        pass
    (ARTIFACTS / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (RESULTS / "model_selection.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(experiments.to_string(index=False))
    print(f"Selected model: {metadata['selected_model']}")


if __name__ == "__main__":
    main()
