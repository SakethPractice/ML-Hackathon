"""Export label-free candidate probabilities for PathSnap's HMM decoder.

Reuses an EXISTING trained model pipeline. No training, reference route, or
reference-label files are read here. Run from the repository root:

    python scripts/export_complete_decoder_predictions.py --model logistic_regression

Requires artifacts/models/logistic_regression.joblib (provided by model trainer)
and the raw/processed student_test trip data. The model artifact must be trusted.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.data.loader import load_trip
from pathsnap.data.processed import load_processed
from pathsnap.features import extract_features, get_feature_columns
from pathsnap.graph.road_graph import RoadGraph
from pathsnap.models.evaluate import positive_class_probability

KEY_COLUMNS = ["observation_id", "arc_id"]


def _assert_same_candidate_keys(expected: pd.DataFrame, actual: pd.DataFrame, trip_id: str) -> None:
    """Require exact keyed coverage, independent of row order."""
    a = pd.MultiIndex.from_frame(expected[KEY_COLUMNS])
    b = pd.MultiIndex.from_frame(actual[KEY_COLUMNS])
    if a.has_duplicates or b.has_duplicates:
        raise ValueError(f"Duplicate observation/arc keys in trip {trip_id}")
    if len(a) != len(b) or len(a.difference(b)) or len(b.difference(a)):
        raise ValueError(
            f"Feature rows do not cover every candidate for trip {trip_id}: "
            f"candidates={len(a)}, features={len(b)}, "
            f"missing={len(a.difference(b))}, extra={len(b.difference(a))}"
        )


def export_complete_predictions(
    model: object,
    *,
    partition: str,
    raw_root: Path,
    processed_root: Path,
    output: Path,
) -> pd.DataFrame:
    """Score ALL candidate rows. Never consult label/reference files."""
    base = processed_root / partition
    if not base.is_dir():
        raise FileNotFoundError(f"Processed partition not found: {base}")

    feature_columns = get_feature_columns()
    all_outputs: list[pd.DataFrame] = []

    for trip_folder in sorted(p for p in base.iterdir() if p.is_dir()):
        trip_id = trip_folder.name
        tables = load_processed(
            partition, trip_id, root=processed_root, with_labels=False
        )
        gps = tables["gps_observations"]
        raw_candidates = tables["candidate_segments"]

        # Graph inputs only: the .route reference is NEVER opened.
        trip = load_trip(partition, trip_id, data_root=raw_root, with_route=False)
        graph = RoadGraph.from_trip(trip).to_networkx()

        feature_rows = extract_features(gps, raw_candidates, road_graph=graph)
        _assert_same_candidate_keys(raw_candidates, feature_rows, trip_id)

        # The fitted sklearn Pipeline includes its original imputing/scaling.
        # Never fit on student-test data or substitute new preprocessing here.
        probability = np.asarray(
            positive_class_probability(model, feature_rows[feature_columns]), dtype=float
        )
        if probability.shape != (len(feature_rows),):
            raise ValueError(f"Wrong prediction count for trip {trip_id}")
        if not np.isfinite(probability).all() or not np.logical_and(
            probability >= 0, probability <= 1
        ).all():
            raise ValueError(f"Invalid predicted probabilities in trip {trip_id}")

        # Intentionally no is_reference, label_valid, ref_arc_id, etc.
        scored = feature_rows[KEY_COLUMNS].copy()
        scored.insert(0, "trip_id", str(trip_id))
        scored["match_probability"] = probability
        all_outputs.append(scored)
        print(f"{trip_id}: scored {len(scored)} of {len(raw_candidates)} candidates")

    if not all_outputs:
        raise ValueError(f"No trips found in {base}")

    result = pd.concat(all_outputs, ignore_index=True)
    keys = ["trip_id", *KEY_COLUMNS]
    if result.duplicated(keys).any():
        raise ValueError("Duplicate prediction keys in final output")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, compression="gzip")
    print(f"Saved {len(result)} full-coverage candidate predictions to {output}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="logistic_regression")
    parser.add_argument("--partition", choices=["student_test", "train"], default="student_test")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path(
        os.environ.get("PATHSNAP_DATA_ROOT", str(ROOT / "data" / "raw"))
    ))
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model_path = args.model_path or ROOT / "artifacts" / "models" / f"{args.model}.joblib"
    if not model_path.is_file() or model_path.stat().st_size == 0:
        parser.error(
            f"Trained model missing or empty: {model_path}. Ask M2 for the fitted "
            "joblib pipeline. There is no need to rerun five-model CV."
        )
    # joblib/pickle is unsafe with untrusted files: only load M2's known artifact.
    model = joblib.load(model_path)
    output = args.output or (
        ROOT / "results" / "predictions" / f"{args.partition}_{args.model}_full.csv.gz"
    )
    export_complete_predictions(
        model,
        partition=args.partition,
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        output=output,
    )


if __name__ == "__main__":
    main()
