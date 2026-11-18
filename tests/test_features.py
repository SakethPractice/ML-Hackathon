
import pandas as pd

from pathsnap.features.extract import (
    extract_features,
    get_feature_matrix
)


def test_feature_extraction():

    gps = pd.DataFrame({
        "observation_id": [0, 1, 2],
        "longitude": [77.5945, 77.5948, 77.5951],
        "latitude": [12.9716, 12.9718, 12.9720],
        "timestamp": [1000, 1005, 1010]
    })

    candidates = pd.DataFrame({
        "observation_id": [0, 0, 1, 1, 2],
        "arc_id": [12, 15, 12, 18, 20],
        "distance_to_segment_m": [8.2, 15.4, 5.1, 12.0, 7.5],
        "segment_length_m": [120, 90, 120, 80, 100],
        "segment_bearing_deg": [45, 90, 45, 180, 45]
    })

    result = extract_features(gps, candidates)

    X = get_feature_matrix(result)

    # Every candidate must produce one feature row.
    assert len(result) == len(candidates)

    # No duplicate observation-candidate pairs.
    assert not result.duplicated(
        ["observation_id", "arc_id"]
    ).any()

    # Feature matrix must not contain identifiers.
    assert "arc_id" not in X.columns
    assert "observation_id" not in X.columns

    # Candidate ranking should start at 1.
    assert result.groupby(
        "observation_id"
    )["candidate_rank"].min().eq(1).all()

    print("Feature extraction test passed!")
    print(result.head())
    print("\nFeature matrix shape:", X.shape)


if __name__ == "__main__":
    test_feature_extraction()
