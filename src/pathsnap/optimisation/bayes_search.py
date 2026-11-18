"""Bayesian hyperparameter optimisation (Optuna TPE) with trip-grouped cross-validation.

* Only the rows you pass in are used (pass the 80 training trips only).
* Every CV fold keeps all rows of a trip together; the *whole* estimator/pipeline is cloned
  and fitted inside each training fold, so imputers/scalers never see validation rows.
* Objective (maximised): mean over folds of
    - ``candidate_table_top1``: candidate-table Top-1 (alias ``top1_accuracy``) (needs ``observation_ids``), or
    - ``macro_f1``     : candidate-level binary Macro F1.
  Candidate-level Macro F1, Top-1 proxy and binary log loss are recorded for every fold of
  every trial regardless of the objective.
* ``model_factory(params) -> unfitted estimator``; ``model_factory({})`` MUST return the
  untuned baseline. Member 2's selected pipeline can be supplied through this factory.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..evaluation.route_metrics import candidate_level_metrics, labelled_top1_accuracy

logger = logging.getLogger(__name__)
ModelFactory = Callable[[Mapping[str, Any]], Any]
OBJECTIVES = ("candidate_table_top1", "top1_accuracy", "macro_f1")

DEFAULT_SEARCH_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "lightgbm": {
        "n_estimators": {"type": "int", "low": 100, "high": 800},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "num_leaves": {"type": "int", "low": 15, "high": 127},
        "max_depth": {"type": "int", "low": 3, "high": 12},
        "min_child_samples": {"type": "int", "low": 5, "high": 100},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
        "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    },
    "random_forest": {
        "n_estimators": {"type": "int", "low": 100, "high": 600},
        "max_depth": {"type": "int", "low": 4, "high": 30},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 20},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "max_features": {"type": "float", "low": 0.2, "high": 1.0},
        "class_weight": {"type": "categorical", "choices": [None, "balanced"]},
    },
    "logistic_regression": {
        "C": {"type": "float", "low": 1e-4, "high": 100.0, "log": True},
        "class_weight": {"type": "categorical", "choices": [None, "balanced"]},
    },
    "knn": {
        "n_neighbors": {"type": "int", "low": 3, "high": 101},
        "weights": {"type": "categorical", "choices": ["uniform", "distance"]},
        "p": {"type": "categorical", "choices": [1, 2]},
    },
    "mlp": {
        "hidden_layer_sizes": {"type": "categorical", "choices": ["64", "128", "64-32", "128-64"]},
        "alpha": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        "learning_rate_init": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [64, 128, 256]},
    },
}


@dataclass
class BayesianSearchConfig:
    model_name: str = "lightgbm"
    n_trials: int = 50
    n_splits: int = 5
    seed: int = 42
    objective: str = "candidate_table_top1"
    cv_strategy: str = "stratified_group_kfold"   # or group_kfold
    threshold: float = 0.5
    timeout_seconds: Optional[float] = None
    storage: Optional[str] = None                  # e.g. sqlite:///optuna.db
    study_name: str = "pathsnap_bayes"
    search_space: dict[str, dict[str, Any]] = field(default_factory=dict)  # overrides defaults

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        if self.objective == "top1_accuracy":
            self.objective = "candidate_table_top1"  # legacy config alias
        if self.cv_strategy not in ("stratified_group_kfold", "group_kfold"):
            raise ValueError("cv_strategy must be stratified_group_kfold or group_kfold")
        if self.n_trials < 1 or self.n_splits < 2:
            raise ValueError("n_trials must be >= 1 and n_splits >= 2")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BayesianSearchConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        valid = set(cls.__dataclass_fields__)
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"Unknown bayesian search config keys: {sorted(unknown)}")
        return cls(**data)

    def space(self) -> dict[str, dict[str, Any]]:
        if self.search_space:
            return self.search_space
        if self.model_name not in DEFAULT_SEARCH_SPACES:
            raise ValueError(f"No default search space for '{self.model_name}'; supply search_space in config")
        return DEFAULT_SEARCH_SPACES[self.model_name]


@dataclass
class OptimisationResult:
    best_params: dict[str, Any]
    best_value: float
    objective: str
    tuned_model: Any
    trials: pd.DataFrame
    fold_results: pd.DataFrame
    comparison: pd.DataFrame
    limitations: list[str]
    artifact_paths: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------- factories
def default_model_factory(model_name: str, seed: int = 42, n_jobs: int = 1) -> ModelFactory:
    """Fallback factory for the five project model families (Member 2's pipeline preferred)."""
    def build(params: Mapping[str, Any] | None = None):
        p = dict(params or {})
        if model_name == "lightgbm":
            from lightgbm import LGBMClassifier
            return Pipeline([("clf", LGBMClassifier(random_state=seed, n_jobs=n_jobs, verbose=-1, **p))])
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if model_name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **p)
        elif model_name == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            steps.append(("scaler", StandardScaler()))
            clf = LogisticRegression(max_iter=2000, random_state=seed, **p)
        elif model_name == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            steps.append(("scaler", StandardScaler()))
            clf = KNeighborsClassifier(n_jobs=n_jobs, **p)
        elif model_name == "mlp":
            from sklearn.neural_network import MLPClassifier
            steps.append(("scaler", StandardScaler()))
            if "hidden_layer_sizes" in p:
                p["hidden_layer_sizes"] = tuple(int(x) for x in str(p["hidden_layer_sizes"]).split("-"))
            clf = MLPClassifier(random_state=seed, max_iter=300, early_stopping=True, **p)
        else:
            raise ValueError(f"Unknown model_name '{model_name}'")
        steps.append(("clf", clf))
        return Pipeline(steps)
    return build


def _suggest(trial: optuna.Trial, space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, s in space.items():
        kind = s["type"]
        if kind == "int":
            kw = {"log": bool(s.get("log", False))}
            if "step" in s:
                kw = {"step": int(s["step"])}
            params[name] = trial.suggest_int(name, int(s["low"]), int(s["high"]), **kw)
        elif kind == "float":
            params[name] = trial.suggest_float(name, float(s["low"]), float(s["high"]), log=bool(s.get("log", False)))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(s["choices"]))
        else:
            raise ValueError(f"Unknown search-space type '{kind}' for '{name}'")
    return params


# ------------------------------------------------------------------ helpers
def _rows(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else np.asarray(X)[idx]


def predict_match_scores(model, X) -> np.ndarray:
    """P(match) for the positive class (class label 1)."""
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        classes = list(getattr(model, "classes_", [0, 1]))
        return proba[:, classes.index(1)] if 1 in classes else proba[:, -1]
    if hasattr(model, "decision_function"):
        return 1.0 / (1.0 + np.exp(-np.asarray(model.decision_function(X))))
    raise TypeError("model must implement predict_proba or decision_function")


def make_group_splits(y, groups, n_splits: int, seed: int, strategy: str):
    groups = np.asarray(groups)
    if len(np.unique(groups)) < n_splits:
        raise ValueError("fewer trips than CV folds")
    if strategy == "group_kfold":
        splits = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))
    else:
        splits = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                      .split(np.zeros(len(y)), y, groups))
    for tr, va in splits:   # hard guarantee: no trip on both sides of a fold
        if np.intersect1d(groups[tr], groups[va]).size:
            raise RuntimeError("trip leakage between train and validation fold")
    return splits


def evaluate_fitted_model(model, X, y, groups, observation_ids=None, threshold: float = 0.5) -> dict[str, float]:
    """Candidate-level + Top-1-proxy metrics and inference time of an already fitted model."""
    t0 = time.perf_counter()
    p = predict_match_scores(model, X)
    infer = time.perf_counter() - t0
    y = np.asarray(y).astype(int)
    m = candidate_level_metrics(y, (p >= threshold).astype(int), p)
    m["inference_seconds"] = infer
    m["inference_seconds_per_1000_rows"] = infer / max(len(y), 1) * 1000
    if observation_ids is not None:
        acc, n_pos, n_none = labelled_top1_accuracy(groups, observation_ids, y, p)
        m.update(candidate_table_top1=acc, n_obs_with_positive=n_pos, n_obs_without_positive=n_none)
    return m


def _cv_folds(factory: ModelFactory, params: Mapping[str, Any], X, y, groups, obs_ids, splits, cfg) -> list[dict]:
    out = []
    for k, (tr, va) in enumerate(splits):
        if len(np.unique(y[tr])) < 2:
            raise ValueError(f"fold {k}: training fold contains a single class")
        model = clone(factory(params))
        t0 = time.perf_counter()
        model.fit(_rows(X, tr), y[tr])          # all learned preprocessing is inside the fitted pipeline
        fit_s = time.perf_counter() - t0
        p = predict_match_scores(model, _rows(X, va))
        m = candidate_level_metrics(y[va], (p >= cfg.threshold).astype(int), p)
        row = {"fold": k, "n_train": int(len(tr)), "n_val": int(len(va)),
               "n_val_trips": int(len(np.unique(groups[va]))), "fit_seconds": fit_s,
               "macro_f1": m["macro_f1"], "log_loss": m["log_loss"], "accuracy": m["accuracy"],
               "precision": m["precision"], "recall": m["recall"], "candidate_table_top1": float("nan")}
        if obs_ids is not None:
            acc, n_pos, _ = labelled_top1_accuracy(groups[va], obs_ids[va], y[va], p)
            row.update(candidate_table_top1=acc, n_obs_with_positive=n_pos)
        out.append(row)
    return out


def _fold_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals = np.array([r[key] for r in rows], dtype=float)
    return float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float("nan")


def save_optimisation_plots(trials: pd.DataFrame, output_dir: Path, objective: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    done = trials[trials["state"] == "COMPLETE"].sort_values("number")
    if done.empty:
        return files
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(done["number"], done["value"], s=14, label="trial")
    ax.plot(done["number"], done["value"].cummax(), color="C1", label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel(f"CV {objective}")
    ax.legend()
    ax.set_title("Optuna optimisation history")
    fig.tight_layout()
    p = output_dir / "optimisation_history.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    files.append(str(p))
    pcols = [c for c in done.columns if c.startswith("params_") and pd.api.types.is_numeric_dtype(done[c])]
    if pcols:
        ncol = min(3, len(pcols))
        nrow = int(np.ceil(len(pcols) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
        for ax, c in zip(axes.ravel(), pcols):
            ax.scatter(done[c], done["value"], s=12)
            ax.set_xlabel(c.replace("params_", ""))
            ax.set_ylabel(objective)
            if (done[c] > 0).all() and done[c].max() / max(done[c].min(), 1e-300) > 100:
                ax.set_xscale("log")
        for ax in axes.ravel()[len(pcols):]:
            ax.axis("off")
        fig.tight_layout()
        p = output_dir / "optimisation_param_scatter.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        files.append(str(p))
    return files


# ------------------------------------------------------------- main routine
def optimise_model(model_factory: ModelFactory, X, y, groups, observation_ids=None,
                   config: BayesianSearchConfig | None = None,
                   output_dir: str | Path | None = None) -> OptimisationResult:
    """Run Optuna TPE over trip-grouped CV, refit the best model on all supplied rows.

    X, y, groups (trip ids), observation_ids (observation index per row, optional) must be
    row-aligned and contain ONLY permitted training trips.
    """
    cfg = config or BayesianSearchConfig()
    y_arr = np.asarray(y).astype(int)
    groups_arr = np.asarray(groups).astype(str)
    n = len(y_arr)
    if len(groups_arr) != n or (hasattr(X, "shape") and X.shape[0] != n):
        raise ValueError("X, y and groups must have the same number of rows")
    obs_arr = None if observation_ids is None else np.asarray(observation_ids)
    if obs_arr is not None and len(obs_arr) != n:
        raise ValueError("observation_ids must be row-aligned with y")
    if not set(np.unique(y_arr)) <= {0, 1}:
        raise ValueError("labels must be binary 0/1")

    limitations: list[str] = []
    objective = cfg.objective
    if objective == "candidate_table_top1" and obs_arr is None:
        objective = "macro_f1"
        limitations.append("observation_ids not supplied: optimised candidate-level Macro F1 instead of Top-1.")
    if obs_arr is not None:
        limitations.append("Top-1 here is a CANDIDATE-TABLE proxy: observations without any positive candidate "
                           "are excluded from its denominator (full Top-1 with candidate-miss handling is "
                           "computed by evaluate_decoder using the alignment table).")
    space = cfg.space()
    splits = make_group_splits(y_arr, groups_arr, cfg.n_splits, cfg.seed, cfg.cv_strategy)

    out = Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fold_log: list[dict] = []

    def objective_fn(trial: optuna.Trial) -> float:
        params = _suggest(trial, space)
        rows = _cv_folds(model_factory, params, X, y_arr, groups_arr, obs_arr, splits, cfg)
        for r in rows:
            fold_log.append({"trial": trial.number, "is_baseline": False, **r})
        trial.set_user_attr("cv_macro_f1", _fold_mean(rows, "macro_f1"))
        trial.set_user_attr("cv_candidate_table_top1", _fold_mean(rows, "candidate_table_top1"))
        trial.set_user_attr("cv_log_loss", _fold_mean(rows, "log_loss"))
        trial.set_user_attr("cv_fit_seconds", float(sum(r["fit_seconds"] for r in rows)))
        trial.set_user_attr("seed", cfg.seed)
        value = _fold_mean(rows, objective)
        if not np.isfinite(value):
            raise ValueError("objective is not finite for this trial")
        return value

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed),
                                study_name=cfg.study_name, storage=cfg.storage)
    t0 = time.perf_counter()
    study.optimize(objective_fn, n_trials=cfg.n_trials, timeout=cfg.timeout_seconds, catch=(ValueError,))
    search_seconds = time.perf_counter() - t0
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise RuntimeError("no Optuna trial completed successfully")

    best = study.best_trial
    best_params = dict(best.params)
    baseline_rows = _cv_folds(model_factory, {}, X, y_arr, groups_arr, obs_arr, splits, cfg)
    for r in baseline_rows:
        fold_log.append({"trial": -1, "is_baseline": True, **r})

    t1 = time.perf_counter()
    tuned = clone(model_factory(best_params))
    tuned.fit(X, y_arr)
    refit_seconds = time.perf_counter() - t1

    trials = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs",
                                           "datetime_start", "datetime_complete", "duration"))
    trials["state"] = trials["state"].astype(str).str.replace("TrialState.", "", regex=False)
    fold_df = pd.DataFrame(fold_log)
    best_rows = fold_df[fold_df["trial"] == best.number].to_dict("records")
    comparison = pd.DataFrame([
        {"model": "baseline_untuned", **{f"cv_{k}": _fold_mean(baseline_rows, k)
                                          for k in ("macro_f1", "candidate_table_top1", "log_loss", "fit_seconds")}},
        {"model": "tuned_best_trial", **{f"cv_{k}": _fold_mean(best_rows, k)
                                         for k in ("macro_f1", "candidate_table_top1", "log_loss", "fit_seconds")}},
    ])

    paths: dict[str, str] = {}
    if out:
        trials.to_csv(out / "optuna_trials.csv", index=False)
        fold_df.to_csv(out / "fold_results.csv", index=False)
        comparison.to_csv(out / "tuned_vs_baseline_cv.csv", index=False)
        meta = {"best_params": best_params, "best_value": best.value, "objective": objective,
                "requested_objective": cfg.objective, "best_trial_number": best.number,
                "n_trials_requested": cfg.n_trials, "n_trials_completed": int(len(trials)),
                "sampler": "optuna.TPESampler", "seed": cfg.seed, "n_splits": cfg.n_splits,
                "cv_strategy": cfg.cv_strategy, "model_name": cfg.model_name,
                "n_rows": int(n), "n_trips": int(len(np.unique(groups_arr))),
                "search_seconds": search_seconds, "refit_seconds": refit_seconds,
                "search_space": {k: dict(v) for k, v in space.items()}, "limitations": limitations}
        (out / "best_params.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        joblib.dump({"model": tuned, "best_params": best_params, "objective": objective,
                     "feature_columns": list(getattr(X, "columns", [])), "seed": cfg.seed},
                    out / "tuned_model.joblib")
        paths = {k: str(out / v) for k, v in {
            "trials_csv": "optuna_trials.csv", "best_params_json": "best_params.json",
            "fold_results_csv": "fold_results.csv", "comparison_csv": "tuned_vs_baseline_cv.csv",
            "model": "tuned_model.joblib"}.items()}
        for i, f in enumerate(save_optimisation_plots(trials, out, objective)):
            paths[f"plot_{i}"] = f
    return OptimisationResult(best_params, float(best.value), objective, tuned, trials, fold_df,
                              comparison, limitations, paths)
