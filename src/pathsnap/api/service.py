"""Trip discovery, inference pipeline and server-side evaluation for the API.

Inference path for one trip (no ``.route`` file is opened):

    raw .track/.nodes/.arcs -> road graph + candidates (pathsnap.candidates)
    -> features (pathsnap.features) -> trained classifier P(match)
    -> HMM/Viterbi decoder (pathsnap.decoder) -> TripDecodeResult

Evaluation (student mode only) runs afterwards, on the server, and returns
aggregate numbers only. Reference arcs never leave this module.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from pathsnap.decoder.types import TripDecodeResult

logger = logging.getLogger("pathsnap.api")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"
DECODER_CONFIG_PATH = PROJECT_ROOT / "configs" / "decoder.yaml"
MODEL_SELECTION_PATH = PROJECT_ROOT / "results" / "model_selection.json"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"

# PATHSNAP_MODE picks which partition the API serves.
MODES = {"evaluator": "evaluator_test_partition", "student": "student_test_partition"}


class EvaluatorTripsUnavailable(RuntimeError):
    """Raised when the configured trip partition is not available."""


class PredictionUnavailable(RuntimeError):
    """Raised when the trained model or decoder cannot be loaded."""


def get_mode() -> str:
    mode = os.environ.get("PATHSNAP_MODE", "evaluator").strip().lower()
    if mode not in MODES:
        raise EvaluatorTripsUnavailable(
            f"PATHSNAP_MODE must be one of {sorted(MODES)}, got {mode!r}."
        )
    return mode


def _load_data_settings() -> tuple[Path, str]:
    """(data_root, partition) for the current mode."""
    mode = get_mode()
    try:
        with DATA_CONFIG_PATH.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        configured_root = os.environ.get("PATHSNAP_DATA_ROOT", config["data_root"])
        partition = config[MODES[mode]]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise EvaluatorTripsUnavailable(
            f"Could not load data settings from {DATA_CONFIG_PATH}."
        ) from exc

    data_root = Path(configured_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    return data_root, str(partition)


def list_evaluator_trip_ids() -> list[str]:
    """Trips of the served partition that have .track, .nodes and .arcs."""
    data_root, partition = _load_data_settings()
    partition_dir = data_root / partition
    if not partition_dir.is_dir():
        raise EvaluatorTripsUnavailable(
            f"Trip data is unavailable: {partition_dir} does not exist."
        )
    required_inputs = (".track", ".nodes", ".arcs")
    return [
        trip_dir.name
        for trip_dir in sorted(partition_dir.iterdir())
        if trip_dir.is_dir()
        and all((trip_dir / f"{trip_dir.name}{suffix}").is_file() for suffix in required_inputs)
    ]


def selected_model_name() -> str:
    try:
        return json.loads(MODEL_SELECTION_PATH.read_text(encoding="utf-8"))["selected_model"]
    except (OSError, KeyError, ValueError):
        return "logistic_regression"


def model_path() -> Path:
    override = os.environ.get("PATHSNAP_MODEL_PATH")
    return Path(override) if override else MODEL_DIR / f"{selected_model_name()}.joblib"


# ---------------------------------------------------------------- inference
@dataclass
class TripPrediction:
    result: TripDecodeResult
    observations: list[list[float]]  # [lat, lon] per GPS fix, for the map
    timings: dict[str, float]


class PathSnapPredictor:
    """Loads the trained classifier once and runs the full pipeline per trip."""

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        with self._lock:
            if self._model is None:
                path = model_path()
                if not path.is_file():
                    raise PredictionUnavailable(
                        f"Trained model not found at {path}. "
                        "Run: python scripts/train_final_model.py"
                    )
                import joblib

                logger.info("Loading model %s", path)
                self._model = joblib.load(path)
        return self._model

    def predict(self, trip_id: str) -> TripPrediction:
        from pathsnap.candidates.pipeline import load_data_config, prepare_trip
        from pathsnap.data.processed import candidate_table, gps_table
        from pathsnap.decoder.config import DecoderConfig
        from pathsnap.decoder.hmm import decode_trip
        from pathsnap.decoder.pipeline_adapter import decoder_inputs
        from pathsnap.features import extract_features, get_feature_columns
        from pathsnap.models.evaluate import positive_class_probability

        model = self._load()
        data_root, partition = _load_data_settings()
        timings: dict[str, float] = {}

        t = time.perf_counter()
        prepared = prepare_trip(partition, trip_id, load_data_config(DATA_CONFIG_PATH),
                                with_labels=False, data_root=data_root)
        timings["candidates_s"] = time.perf_counter() - t
        logger.info("trip %s: %d fixes, %d candidate rows", trip_id,
                    len(prepared.trip.track), len(prepared.candidates.table))

        t = time.perf_counter()
        features = extract_features(gps_table(prepared), candidate_table(prepared),
                                    road_graph=prepared.graph.to_networkx())
        timings["features_s"] = time.perf_counter() - t

        t = time.perf_counter()
        probability = positive_class_probability(model, features[get_feature_columns()])
        scores = pd.DataFrame({
            "obs_id": features["observation_id"].to_numpy(),
            "arc_id": features["arc_id"].to_numpy(),
            "match_probability": np.asarray(probability, dtype=float),
        })
        timings["scoring_s"] = time.perf_counter() - t

        t = time.perf_counter()
        observations, candidates, network = decoder_inputs(prepared, scores)
        result = decode_trip(trip_id, observations, candidates, network,
                             DecoderConfig.from_yaml(DECODER_CONFIG_PATH))
        timings["decoding_s"] = time.perf_counter() - t
        logger.info("trip %s: decoded, status=%s, %s", trip_id, result.status,
                    ", ".join(f"{k}={v:.2f}" for k, v in timings.items()))

        track = prepared.trip.track
        points = np.column_stack([track["lat"].to_numpy(), track["lon"].to_numpy()]).tolist()
        return TripPrediction(result, points, timings)


class PredictionService:
    """Boundary between FastAPI routes and the prediction pipeline.

    ``predictor`` may be any callable returning a ``TripDecodeResult`` (used
    in tests); by default the full trained pipeline is used.
    """

    def __init__(self, predictor: Callable[[str], TripDecodeResult] | None = None,
                 use_default_pipeline: bool = False) -> None:
        self._predictor = predictor
        self._pipeline = PathSnapPredictor() if use_default_pipeline else None

    @property
    def model_loaded(self) -> bool:
        return self._pipeline is not None and self._pipeline.model_loaded

    def predict_trip(self, trip_id: str) -> TripPrediction:
        if self._predictor is not None:
            result = self._predictor(trip_id)
            if not isinstance(result, TripDecodeResult):
                raise TypeError("The prediction pipeline must return TripDecodeResult.")
            return TripPrediction(result, [], {})
        if self._pipeline is None:
            raise PredictionUnavailable(
                "Prediction pipeline is not configured: no trained model scorer "
                "and decoder are attached to this service."
            )
        return self._pipeline.predict(trip_id)


# --------------------------------------------------------------- evaluation
def evaluate_trip(trip_id: str, result: TripDecodeResult) -> dict | None:
    """Aggregate accuracy against the reference route, student mode only.

    Runs after inference. Returns counts and rates only, never reference arcs.
    Returns None when evaluation does not apply (evaluator mode, no route).
    """
    if get_mode() != "student":
        return None
    from pathsnap.candidates.labels import reference_labels
    from pathsnap.candidates.pipeline import load_data_config
    from pathsnap.data.loader import load_trip
    from pathsnap.graph.road_graph import RoadGraph

    data_root, partition = _load_data_settings()
    try:
        trip = load_trip(partition, trip_id, data_root, with_route=True)
    except (OSError, ValueError):
        return None
    if not trip.has_route:
        return None
    cfg = load_data_config(DATA_CONFIG_PATH)
    graph = RoadGraph.from_trip(trip)
    xy = graph.projection.to_xy(trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy())
    labels = reference_labels(trip, graph, xy, cfg.max_label_dist_m)

    predicted = pd.Series(result.matched_arc_ids, index=result.observation_indices, dtype="object")
    valid = labels.index[labels["label_valid"].to_numpy()]
    pred = predicted.reindex(valid)
    correct = pred.notna().to_numpy() & (pred.to_numpy() == labels.loc[valid, "ref_arc_id"].to_numpy())
    n = int(len(valid))
    route_arcs = set(trip.route["arc_id"].tolist())
    decoded_arcs = set(result.expanded_route_arc_ids)
    return {
        "partition": partition,
        "labelled_observations": n,
        "pointwise_accuracy": float(correct.mean()) if n else None,
        "route_arc_recall": (len(route_arcs & decoded_arcs) / len(route_arcs)) if route_arcs else None,
        "route_arc_precision": (len(route_arcs & decoded_arcs) / len(decoded_arcs)) if decoded_arcs else None,
    }
