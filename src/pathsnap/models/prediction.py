"""Candidate probability prediction utilities."""

from __future__ import annotations

import numpy as np

from .evaluate import positive_class_probability


def predict_candidate_probabilities(model, feature_frame, feature_columns) -> np.ndarray:
    X = feature_frame[feature_columns]
    return positive_class_probability(model, X)


def rank_candidates(feature_frame, probabilities) -> object:
    """Attach and rank probabilities within each observation."""
    result = feature_frame.copy()
    result["probability"] = np.asarray(probabilities)
    result["probability_rank"] = result.groupby("obs_id")["probability"].rank(
        method="first", ascending=False
    ).astype(int)
    return result.sort_values(["obs_id", "probability_rank"]).reset_index(drop=True)