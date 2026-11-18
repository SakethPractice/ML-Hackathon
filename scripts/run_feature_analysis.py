"""Run feature engineering and quality analysis on processed training data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

from pathsnap.data.loader import load_trip
from pathsnap.data.processed import load_processed
from pathsnap.features import extract_features, get_feature_matrix
from pathsnap.features.analysis import descriptive_statistics, correlation_tables, save_feature_plots
from pathsnap.features.validation import ALL_FEATURES


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SEED = 42


def build_labeled_data(partition: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build labelled candidate rows for a permitted development partition.

    Labels are merged only after inference-time features are extracted.  This
    keeps the feature matrix free of route/reference information.
    """
    base = ROOT / "data" / "processed" / partition
    if not base.is_dir():
        raise FileNotFoundError(f"Processed partition not found: {base}")
    frames = []
    for trip_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        trip_id = trip_dir.name
        tables = load_processed(partition, trip_id, ROOT / "data" / "processed", with_labels=True)
        trip = load_trip(partition, trip_id, data_root=ROOT / "data" / "raw")
        graph = trip_graph = trip
        from pathsnap.graph.road_graph import RoadGraph
        graph = RoadGraph.from_trip(trip_graph).to_networkx()

        features = extract_features(
            tables["gps_observations"], tables["candidate_segments"], road_graph=graph
        )
        labels = tables["candidate_labels"].merge(
            tables["labels"][["observation_id", "ref_has_duplicate"]],
            on="observation_id", how="left", validate="many_to_one"
        )
        labels = labels[(labels["label_valid"].fillna(False) & ~labels["ref_has_duplicate"].fillna(False))]
        features = features.merge(
            labels[["observation_id", "arc_id", "is_reference"]],
            on=["observation_id", "arc_id"], how="inner", validate="one_to_one"
        )
        frames.append(features)

    data = pd.concat(frames, ignore_index=True)
    X = get_feature_matrix(data)
    y = data["is_reference"].astype("int8")
    return data, X, y


def build_training_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Backward-compatible training-data entry point for feature scripts."""
    data, X, y = build_labeled_data("train")
    return data, X, y, data[["trip_id", "observation_id", "arc_id"]]


def main() -> None:
    data, X, y, ids = build_training_data()
    RESULTS.mkdir(parents=True, exist_ok=True)
    stats = descriptive_statistics(X)
    stats.to_csv(RESULTS / "feature_statistics.csv", index=False)
    pearson, spearman, pairs = correlation_tables(X)
    pairs.to_csv(RESULTS / "feature_correlations.csv", index=False)
    pearson.to_csv(RESULTS / "feature_pearson_correlations.csv")
    spearman.to_csv(RESULTS / "feature_spearman_correlations.csv")
    save_feature_plots(X, y, FIGURES)

    valid_trips = int(data["trip_id"].nunique())
    summary = {
        "partition": "train",
        "trip_count": valid_trips,
        "observation_count": int(data.groupby("trip_id")["observation_id"].nunique().sum()),
        "candidate_count": int(len(data)),
        "class_0_count": int((y == 0).sum()),
        "class_1_count": int((y == 1).sum()),
        "missing_values": int(X.isna().sum().sum()),
        "feature_count": int(X.shape[1]),
        "feature_names": ALL_FEATURES,
        "cv_folds": 5,
        "group_column": "trip_id",
    }
    (RESULTS / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote feature analysis outputs under {RESULTS}")


if __name__ == "__main__":
    main()
