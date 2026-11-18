
"""
PathSnap - Feature Engineering

Generates one feature row per GPS observation-candidate road pair.

Inputs:
    gps_observations:
        observation_id, longitude, latitude, timestamp

    candidate_segments:
        observation_id, arc_id, distance_to_segment_m,
        segment_length_m, segment_bearing_deg

    road_graph:
        Optional NetworkX directed road graph.

Output:
    pandas DataFrame containing candidate-level ML features.

Important:
    - Does not use reference route labels.
    - Does not use evaluator labels.
    - Does not calculate target arc IDs as features.
    - Distances must be in metres.
    - Bearings must be in degrees clockwise from north.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6_371_000.0


# ---------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------

def validate_inputs(gps_observations, candidate_segments):

    required_gps = {
        "observation_id",
        "longitude",
        "latitude",
        "timestamp"
    }

    required_candidates = {
        "observation_id",
        "arc_id",
        "distance_to_segment_m",
        "segment_length_m",
        "segment_bearing_deg"
    }

    missing_gps = required_gps - set(gps_observations.columns)
    missing_candidates = required_candidates - set(candidate_segments.columns)

    if missing_gps:
        raise ValueError(
            f"Missing GPS columns: {sorted(missing_gps)}"
        )

    if missing_candidates:
        raise ValueError(
            f"Missing candidate columns: {sorted(missing_candidates)}"
        )

    if gps_observations.empty:
        raise ValueError("GPS observations cannot be empty.")

    if candidate_segments.empty:
        raise ValueError("Candidate segments cannot be empty.")

    if gps_observations["observation_id"].duplicated().any():
        raise ValueError("GPS observation IDs must be unique.")

    gps = gps_observations.copy()
    candidates = candidate_segments.copy()

    numeric_gps = ["longitude", "latitude", "timestamp"]

    for column in numeric_gps:
        gps[column] = pd.to_numeric(
            gps[column], errors="coerce"
        )

    numeric_candidates = [
        "distance_to_segment_m",
        "segment_length_m",
        "segment_bearing_deg"
    ]

    for column in numeric_candidates:
        candidates[column] = pd.to_numeric(
            candidates[column], errors="coerce"
        )

    if gps[numeric_gps].isna().any().any():
        raise ValueError(
            "GPS coordinates and timestamps must be valid numbers."
        )

    if candidates[numeric_candidates].isna().any().any():
        raise ValueError(
            "Candidate geometry features must contain valid numbers."
        )

    if not gps["longitude"].between(-180, 180).all():
        raise ValueError("Invalid longitude values.")

    if not gps["latitude"].between(-90, 90).all():
        raise ValueError("Invalid latitude values.")

    if (candidates["distance_to_segment_m"] < 0).any():
        raise ValueError("Candidate distances cannot be negative.")

    if (candidates["segment_length_m"] < 0).any():
        raise ValueError("Segment lengths cannot be negative.")

    if not candidates["segment_bearing_deg"].between(
        0, 360
    ).all():
        raise ValueError("Segment bearings must be between 0 and 360.")

    if candidates.duplicated(
        subset=["observation_id", "arc_id"]
    ).any():
        raise ValueError(
            "Duplicate observation-candidate pairs detected."
        )

    unknown_ids = set(candidates["observation_id"]) - set(
        gps["observation_id"]
    )

    if unknown_ids:
        raise ValueError(
            f"Candidates reference unknown observations: "
            f"{list(unknown_ids)[:5]}"
        )

    return gps, candidates


# ---------------------------------------------------------
# 2. Geographic calculations
# ---------------------------------------------------------

def haversine_distance(
    lon1, lat1, lon2, lat2
):
    """
    Calculate geodesic distance between two GPS points.

    Returns distance in metres.
    """

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0.0, 1.0)

    return (
        2 * EARTH_RADIUS_M
        * np.arcsin(np.sqrt(a))
    )


def calculate_bearing(
    lon1, lat1, lon2, lat2
):
    """
    Calculate initial bearing between two GPS points.

    Returns bearing in degrees clockwise from north.
    """

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    dlon = lon2 - lon1

    x = np.sin(dlon) * np.cos(lat2)

    y = (
        np.cos(lat1) * np.sin(lat2)
        - np.sin(lat1)
        * np.cos(lat2)
        * np.cos(dlon)
    )

    bearing = np.degrees(np.arctan2(x, y))

    return (bearing + 360) % 360


def calculate_angle_difference(angle1, angle2):
    """
    Calculate smallest absolute angular difference.

    Result is between 0 and 180 degrees.
    """

    difference = np.abs(angle1 - angle2)

    return np.minimum(
        difference,
        360 - difference
    )


# ---------------------------------------------------------
# 3. GPS movement features
# ---------------------------------------------------------

def calculate_gps_features(gps_observations):
    """
    Calculate movement features between consecutive GPS points.

    Features:
        gps_displacement_m
        time_delta_s
        gps_heading_deg
        implied_speed_mps
    """

    gps = gps_observations.copy()

    gps = gps.sort_values(
        "observation_id"
    ).reset_index(drop=True)

    gps["previous_longitude"] = gps["longitude"].shift(1)
    gps["previous_latitude"] = gps["latitude"].shift(1)

    gps["previous_timestamp"] = gps["timestamp"].shift(1)

    gps["time_delta_s"] = (
        gps["timestamp"] - gps["previous_timestamp"]
    )

    gps["gps_displacement_m"] = haversine_distance(
        gps["previous_longitude"],
        gps["previous_latitude"],
        gps["longitude"],
        gps["latitude"]
    )

    gps["gps_heading_deg"] = calculate_bearing(
        gps["previous_longitude"],
        gps["previous_latitude"],
        gps["longitude"],
        gps["latitude"]
    )

    # First observation has no previous point.
    first_observation = gps["previous_timestamp"].isna()

    gps.loc[
        first_observation,
        [
            "time_delta_s",
            "gps_displacement_m",
            "gps_heading_deg"
        ]
    ] = np.nan

    # Speed is undefined for non-positive time intervals.
    valid_time = gps["time_delta_s"] > 0

    gps["implied_speed_mps"] = np.where(
        valid_time,
        gps["gps_displacement_m"] / gps["time_delta_s"],
        np.nan
    )

    return gps[
        [
            "observation_id",
            "time_delta_s",
            "gps_displacement_m",
            "gps_heading_deg",
            "implied_speed_mps"
        ]
    ]


# ---------------------------------------------------------
# 4. Road graph features
# ---------------------------------------------------------

def calculate_graph_features(candidates, road_graph=None):
    """
    Calculate connectivity-related features.

    Requires candidate rows to optionally contain:
        from_node_id
        to_node_id

    If a directed NetworkX graph is supplied, calculate
    in-degree and out-degree for candidate endpoints.
    """

    result = candidates.copy()

    result["from_node_degree"] = np.nan
    result["to_node_degree"] = np.nan

    result["from_node_in_degree"] = np.nan
    result["from_node_out_degree"] = np.nan

    result["to_node_in_degree"] = np.nan
    result["to_node_out_degree"] = np.nan

    if road_graph is None:
        return result

    if not hasattr(road_graph, "degree"):
        raise TypeError(
            "road_graph must be a NetworkX graph."
        )

    if not {
        "from_node_id",
        "to_node_id"
    }.issubset(result.columns):
        raise ValueError(
            "Candidate rows must contain from_node_id and "
            "to_node_id when road_graph is supplied."
        )

    def get_degree(node, degree_type):
        if node not in road_graph:
            return np.nan

        if road_graph.is_directed():
            if degree_type == "in":
                return road_graph.in_degree(node)
            if degree_type == "out":
                return road_graph.out_degree(node)

        return road_graph.degree(node)

    for index, row in result.iterrows():

        source = row["from_node_id"]
        target = row["to_node_id"]

        source_degree = get_degree(source, "in")
        target_degree = get_degree(target, "in")

        result.at[index, "from_node_degree"] = (
            road_graph.degree(source)
            if source in road_graph else np.nan
        )

        result.at[index, "to_node_degree"] = (
            road_graph.degree(target)
            if target in road_graph else np.nan
        )

        result.at[index, "from_node_in_degree"] = (
            source_degree if road_graph.is_directed()
            else np.nan
        )

        result.at[index, "from_node_out_degree"] = (
            get_degree(source, "out")
            if road_graph.is_directed() else np.nan
        )

        result.at[index, "to_node_in_degree"] = (
            target_degree if road_graph.is_directed()
            else np.nan
        )

        result.at[index, "to_node_out_degree"] = (
            get_degree(target, "out")
            if road_graph.is_directed() else np.nan
        )

    return result


# ---------------------------------------------------------
# 5. Main feature extraction function
# ---------------------------------------------------------

def extract_features(
    gps_observations,
    candidate_segments,
    road_graph=None
):
    """
    Generate candidate-level features for PathSnap.

    Parameters
    ----------
    gps_observations : pandas.DataFrame
        GPS observations with:
        observation_id, longitude, latitude, timestamp.

    candidate_segments : pandas.DataFrame
        Candidate road segments with:
        observation_id, arc_id, distance_to_segment_m,
        segment_length_m, segment_bearing_deg.

        Optional:
        from_node_id, to_node_id.

    road_graph : NetworkX graph, optional
        Directed road graph for connectivity features.

    Returns
    -------
    pandas.DataFrame
        One feature row per observation-candidate pair.
    """

    # Validate input data.
    gps, candidates = validate_inputs(
        gps_observations,
        candidate_segments
    )

    # Calculate GPS movement features.
    gps_features = calculate_gps_features(gps)

    # Join GPS movement features to candidate pairs.
    features = candidates.merge(
        gps_features,
        on="observation_id",
        how="left",
        validate="many_to_one"
    )

    # Calculate candidate distance ranking.
    features["candidate_rank"] = (
        features.groupby("observation_id")[
            "distance_to_segment_m"
        ]
        .rank(method="dense", ascending=True)
    )

    # Calculate heading difference.
    features["heading_difference_deg"] = (
        calculate_angle_difference(
            features["gps_heading_deg"].to_numpy(),
            features["segment_bearing_deg"].to_numpy()
        )
    )

    # No heading information for the first observation.
    features.loc[
        features["gps_heading_deg"].isna(),
        "heading_difference_deg"
    ] = np.nan

    # Calculate road graph features.
    features = calculate_graph_features(
        features,
        road_graph
    )

    # Preserve original row ordering.
    features = features.sort_values(
        ["observation_id", "candidate_rank", "arc_id"]
    ).reset_index(drop=True)

    # Check output integrity.
    if len(features) != len(candidates):
        raise RuntimeError(
            "Feature extraction changed the candidate row count."
        )

    if features.duplicated(
        subset=["observation_id", "arc_id"]
    ).any():
        raise RuntimeError(
            "Duplicate candidate rows found in feature output."
        )

    return features


# ---------------------------------------------------------
# 6. Get ML feature columns
# ---------------------------------------------------------

def get_feature_columns():
    """
    Return the approved feature columns for ML training.

    Identifiers and target labels are deliberately excluded.
    """

    return [
        "distance_to_segment_m",
        "segment_length_m",
        "segment_bearing_deg",
        "time_delta_s",
        "gps_displacement_m",
        "gps_heading_deg",
        "implied_speed_mps",
        "candidate_rank",
        "heading_difference_deg",
        "from_node_degree",
        "to_node_degree",
        "from_node_in_degree",
        "from_node_out_degree",
        "to_node_in_degree",
        "to_node_out_degree"
    ]


def get_feature_matrix(features):
    """
    Extract only ML input columns.

    Returns:
        X: Feature matrix without observation IDs or arc IDs.
    """

    columns = get_feature_columns()

    missing = set(columns) - set(features.columns)

    if missing:
        raise ValueError(
            f"Missing feature columns: {sorted(missing)}"
        )

    return features[columns].copy()
