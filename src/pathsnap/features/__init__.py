"""
PathSnap feature engineering package.

Contains functions for:
- GPS movement feature extraction
- Road segment feature extraction
- Candidate ranking
- Road graph connectivity features
"""

from .extract import (
    extract_features,
    get_feature_columns,
    get_feature_matrix,
    calculate_gps_features,
    calculate_graph_features,
    haversine_distance,
    calculate_bearing,
    calculate_angle_difference,
    validate_inputs
)
from .analysis import descriptive_statistics, correlation_tables
from .selection import variance_selected, correlation_selected, kbest_selected, permutation_selected
from .validation import ablation_configs, model_definitions, observation_top1

__all__ = [
    "extract_features",
    "get_feature_columns",
    "get_feature_matrix",
    "calculate_gps_features",
    "calculate_graph_features",
    "haversine_distance",
    "calculate_bearing",
    "calculate_angle_difference",
    "validate_inputs"
]