"""Tests for candidate model interfaces."""

import numpy as np
import pandas as pd

from pathsnap.models.compare import comparison_table
from pathsnap.models.evaluate import candidate_metrics, positive_class_probability
from pathsnap.models.prediction import rank_candidates
from pathsnap.models.train import make_model_definitions


def test_model_definitions_cover_five_families():
    models = make_model_definitions(random_seed=7)
    assert set(models) == {
        "logistic_regression",
        "knn",
        "decision_tree",
        "random_forest",
        "extra_trees",
    }


def test_candidate_metrics_and_probability_interface():
    X = pd.DataFrame({"x": [-2, -1, 0, 1, 2, 3], "y": [0, 0, 1, 1, 2, 2]})
    y = np.array([0, 0, 0, 1, 1, 1])
    model = make_model_definitions()["logistic_regression"].fit(X, y)
    pred = model.predict(X)
    prob = positive_class_probability(model, X)
    metrics = candidate_metrics(y, pred, prob)
    assert set(metrics) >= {"accuracy", "balanced_accuracy", "macro_f1", "log_loss"}


def test_rank_candidates_is_within_observation():
    frame = pd.DataFrame({"obs_id": [0, 0, 1, 1], "arc_id": [10, 11, 20, 21]})
    ranked = rank_candidates(frame, [0.1, 0.9, 0.8, 0.2])
    assert ranked.groupby("obs_id")["probability_rank"].apply(set).to_dict() == {
        0: {1, 2},
        1: {1, 2},
    }


def test_comparison_table():
    table = comparison_table({
        "model_a": {"accuracy": 0.8, "macro_f1": 0.7},
        "model_b": {"accuracy": 0.9, "macro_f1": 0.6},
    })
    assert table.iloc[0]["model"] == "model_a"
