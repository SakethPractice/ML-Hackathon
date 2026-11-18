"""Trained candidate-model and HMM inference adapter for the API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from pathsnap.candidates.pipeline import load_data_config, prepare_trip
from pathsnap.data.processed import candidate_table, gps_table
from pathsnap.decoder.config import DecoderConfig
from pathsnap.decoder.hmm import decode_trip
from pathsnap.decoder.pipeline_adapter import decoder_inputs
from pathsnap.decoder.types import TripDecodeResult
from pathsnap.features import extract_features, get_feature_columns, get_feature_matrix
from pathsnap.models.prediction import predict_candidate_probabilities


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIRECTORY = PROJECT_ROOT / "artifacts" / "models"
MODEL_METADATA = MODEL_DIRECTORY / "experiment_metadata.json"
DATA_CONFIG = PROJECT_ROOT / "configs" / "data.yaml"
DECODER_CONFIG = PROJECT_ROOT / "configs" / "decoder.yaml"


class TrainedModelPredictor:
    """Run the selected trained classifier and HMM for one raw trip."""

    def __init__(
        self,
        model: Any,
        feature_columns: list[str],
        data_root: str | Path,
        partition: str = "evaluator_test",
        decoder_config: DecoderConfig | None = None,
    ) -> None:
        expected = get_feature_columns()
        if feature_columns != expected:
            raise ValueError(
                "The model artifact feature columns do not match the current "
                f"inference contract. Expected {expected}, got {feature_columns}."
            )
        self.model = model
        self.feature_columns = feature_columns
        self.data_root = Path(data_root)
        self.partition = partition
        self.data_config = load_data_config(DATA_CONFIG)
        self.decoder_config = decoder_config or DecoderConfig.from_yaml(DECODER_CONFIG)

    @classmethod
    def from_artifacts(
        cls,
        data_root: str | Path,
        partition: str = "evaluator_test",
    ) -> "TrainedModelPredictor":
        if not MODEL_METADATA.is_file():
            raise FileNotFoundError(f"Trained model metadata not found: {MODEL_METADATA}")
        metadata = json.loads(MODEL_METADATA.read_text(encoding="utf-8"))
        model_name = metadata.get("selected_model")
        allowed_models = {"logistic_regression", "knn", "random_forest", "lightgbm", "mlp"}
        if model_name not in allowed_models:
            raise ValueError(f"Unsupported selected model in metadata: {model_name!r}")

        model_path = MODEL_DIRECTORY / f"{model_name}.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Trained model artifact not found: {model_path}")

        model = joblib.load(model_path)
        feature_columns = metadata.get("feature_columns")
        if not isinstance(feature_columns, list) or not all(
            isinstance(name, str) for name in feature_columns
        ):
            raise ValueError(f"Invalid feature_columns in {MODEL_METADATA}")
        return cls(model, feature_columns, data_root, partition)

    def __call__(self, trip_id: str) -> TripDecodeResult:
        trip = prepare_trip(
            self.partition,
            trip_id,
            cfg=self.data_config,
            with_labels=False,
            data_root=self.data_root,
        )

        features = extract_features(
            gps_table(trip),
            candidate_table(trip),
            road_graph=trip.graph.to_networkx(),
        )
        X = get_feature_matrix(features)[self.feature_columns]
        probabilities = predict_candidate_probabilities(
            self.model, X, self.feature_columns
        )

        scores = pd.DataFrame({
            "obs_id": features["observation_id"].to_numpy(),
            "arc_id": features["arc_id"].to_numpy(),
            "match_probability": probabilities,
        })
        observations, candidate_predictions, network = decoder_inputs(trip, scores)
        return decode_trip(
            trip_id,
            observations,
            candidate_predictions,
            network,
            self.decoder_config,
        )
