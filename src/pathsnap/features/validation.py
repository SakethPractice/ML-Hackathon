"""Cross-validation and feature ablation utilities."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as np_pd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


SPATIAL = ["distance_to_segment_m", "segment_length_m", "segment_bearing_deg"]
MOVEMENT = ["time_delta_s", "gps_displacement_m", "gps_heading_deg", "implied_speed_mps"]
DIRECTION = ["candidate_rank", "heading_difference_deg"]
ROAD = ["from_node_degree", "to_node_degree", "from_node_in_degree",
        "from_node_out_degree", "to_node_in_degree", "to_node_out_degree"]
ALL_FEATURES = SPATIAL + MOVEMENT + DIRECTION + ROAD


def ablation_configs(selected: list[str] | None = None) -> dict[str, list[str]]:
    all_features = ALL_FEATURES if selected is None else selected
    return {
        "E1_spatial": [x for x in SPATIAL if x in all_features],
        "E2_spatial_movement": [x for x in SPATIAL + MOVEMENT if x in all_features],
        "E3_spatial_movement_direction": [x for x in SPATIAL + MOVEMENT + DIRECTION if x in all_features],
        "E4_all_15": all_features,
        "E5_without_road": [x for x in all_features if x not in ROAD],
        "E6_without_movement": [x for x in all_features if x not in MOVEMENT],
        "E7_without_direction": [x for x in all_features if x not in DIRECTION],
    }


def model_definitions(seed: int = 42) -> dict[str, Pipeline]:
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=1.0, class_weight="balanced", max_iter=500,
                                         solver="lbfgs", random_state=seed)),
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(
                n_neighbors=7, weights="distance", metric="minkowski",
                algorithm="kd_tree", leaf_size=100, n_jobs=-1,
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=None,
                                             class_weight="balanced", n_jobs=-1, random_state=seed)),
        ]),
        "MLP": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                    solver="adam", batch_size=1024, early_stopping=True, max_iter=150,
                                    random_state=seed)),
        ]),
    }
    if LGBMClassifier is not None:
        models["LightGBM"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                     class_weight="balanced", n_jobs=-1, random_state=seed,
                                     verbosity=-1)),
        ])
    return models


def observation_top1(model, X: pd.DataFrame, observation_ids: np.ndarray,
                     arc_ids: np.ndarray, y: np.ndarray) -> float:
    probabilities = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    positive = int(np.where(classes == 1)[0][0])
    scores = probabilities[:, positive]
    frame = pd.DataFrame({"observation_id": observation_ids, "arc_id": arc_ids,
                          "score": scores, "y": y})
    top = frame.loc[frame.groupby("observation_id")["score"].idxmax()]
    reference = frame.groupby("observation_id")["y"].apply(lambda values: int(values.max()))
    return float((top["y"].to_numpy() == reference.loc[top["observation_id"]].to_numpy()).mean())
