"""Tests for feature-selection and ablation helpers."""

import numpy as np
import pandas as pd

from pathsnap.features.selection import correlation_selected, kbest_selected, variance_selected
from pathsnap.features.validation import ALL_FEATURES, ablation_configs


def test_selection_helpers_preserve_feature_names():
    X = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [1.0, 2.0, 3.0, 4.0],
        "constant": [1.0, 1.0, 1.0, 1.0],
    })
    y = pd.Series([0, 0, 1, 1])
    assert variance_selected(X) == ["a", "b"]
    assert correlation_selected(X, threshold=0.95) == ["a", "constant"]
    assert set(kbest_selected(X[["a", "constant"]], y, k=1)).issubset({"a", "constant"})


def test_ablation_configurations_cover_the_requested_groups():
    configs = ablation_configs()
    assert set(configs) >= {
        "E1_spatial", "E2_spatial_movement",
        "E3_spatial_movement_direction", "E4_all_15",
        "E5_without_road", "E6_without_movement", "E7_without_direction",
    }
    assert len(configs["E4_all_15"]) == 15
    assert set(configs["E4_all_15"]) == set(ALL_FEATURES)
    assert not set(configs["E5_without_road"]) & {
        "from_node_degree", "to_node_degree",
        "from_node_in_degree", "from_node_out_degree",
        "to_node_in_degree", "to_node_out_degree",
    }
