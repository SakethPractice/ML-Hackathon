"""Candidate-level model training for PathSnap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


DEFAULT_FEATURE_COLUMNS = [
    "distance_m",
    "rank",
    "t_along",
    "radius_used_m",
    "dt_prev_s",
    "step_prev_m",
    "speed_prev_mps",
    "heading_deg",
    "window_disp_m",
    "is_stationary",
    "arc_length_m",
    "arc_bearing_deg",
    "from_degree",
    "to_degree",
    "from_in_degree",
    "to_out_degree",
]


@dataclass
class ModelResult:
    name: str
    model: object
    fold_metrics: list[dict]
    training_time_s: float
    random_seed: int


def build_candidate_features(prepared_trip) -> pd.DataFrame:
    """Build one inference-time feature row per (obs_id, arc_id)."""
    candidates = prepared_trip.candidates.table.copy()
    context = prepared_trip.context.copy()
    graph = prepared_trip.graph

    candidates = candidates.merge(context, left_on="obs_id", right_index=True, how="left", validate="many_to_one")
    arc_ids = candidates["arc_id"].to_numpy(dtype=np.int64)
    out_deg, in_deg = graph.node_degree()

    candidates["arc_length_m"] = graph.arc_length_m[arc_ids]
    candidates["arc_bearing_deg"] = graph.arc_bearing_deg[arc_ids]
    candidates["from_degree"] = in_deg[graph.arc_from[arc_ids]] + out_deg[graph.arc_from[arc_ids]]
    candidates["to_degree"] = in_deg[graph.arc_to[arc_ids]] + out_deg[graph.arc_to[arc_ids]]
    candidates["from_in_degree"] = in_deg[graph.arc_from[arc_ids]]
    candidates["to_out_degree"] = out_deg[graph.arc_to[arc_ids]]
    candidates["trip_id"] = prepared_trip.trip.trip_id
    return candidates


def build_training_table(prepared_trips: Iterable) -> pd.DataFrame:
    """Attach defensible reference labels to candidate features."""
    frames = []
    for prepared in prepared_trips:
        if prepared.labels is None:
            raise ValueError(f"Trip {prepared.trip.trip_id} has no reference labels")
        features = build_candidate_features(prepared)
        labels = prepared.labels[
            ["ref_arc_id", "label_valid", "ref_has_duplicate"]
        ].reset_index(names="obs_id")
        features = features.merge(labels, on="obs_id", how="left", validate="many_to_one")

        features["y"] = (
            features["label_valid"].fillna(False)
            & ~features["ref_has_duplicate"].fillna(False)
            & (features["arc_id"] == features["ref_arc_id"])
        ).astype(np.int8)

        valid_pair = features["label_valid"].fillna(False) & ~features["ref_has_duplicate"].fillna(False)
        frames.append(features.loc[valid_pair].copy())

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def make_model_definitions(random_seed: int = 42) -> dict[str, object]:
    """Return five distinct baseline models."""
    linear = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=1.0, max_iter=2000, random_state=random_seed)),
    ])
    knn = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=7, weights="distance")),
    ])
    tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(max_depth=12, random_state=random_seed)),
    ])
    forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=random_seed)),
    ])
    extra = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=random_seed)),
    ])
    return {
        "logistic_regression": linear,
        "knn": knn,
        "decision_tree": tree,
        "random_forest": forest,
        "extra_trees": extra,
    }


def save_model(model: object, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
