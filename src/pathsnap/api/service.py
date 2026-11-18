"""Data discovery and inference service boundary for the PathSnap API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import yaml

from pathsnap.decoder.types import TripDecodeResult


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


class EvaluatorTripsUnavailable(RuntimeError):
    """Raised when the configured evaluator input partition is not available."""


class PredictionUnavailable(RuntimeError):
    """Raised until an actual model-to-decoder predictor is configured."""


def _load_data_settings() -> tuple[Path, str]:
    try:
        with DATA_CONFIG_PATH.open(encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        configured_root = os.environ.get("PATHSNAP_DATA_ROOT", config["data_root"])
        partition = config["evaluator_test_partition"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise EvaluatorTripsUnavailable(
            f"Could not load evaluator data settings from {DATA_CONFIG_PATH}."
        ) from exc

    data_root = Path(configured_root)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    return data_root, str(partition)


def list_evaluator_trip_ids() -> list[str]:
    """List only trips with inference inputs; never opens or requires ``.route``."""
    data_root, partition = _load_data_settings()
    partition_dir = data_root / partition
    if not partition_dir.is_dir():
        raise EvaluatorTripsUnavailable(
            f"Evaluator trip source is unavailable: {partition_dir} does not exist."
        )

    required_inputs = (".track", ".nodes", ".arcs")
    trip_ids = [
        trip_dir.name
        for trip_dir in sorted(partition_dir.iterdir())
        if trip_dir.is_dir()
        and all((trip_dir / f"{trip_dir.name}{suffix}").is_file() for suffix in required_inputs)
    ]
    return trip_ids


class PredictionService:
    """Delegate prediction to the integrated model and HMM/Viterbi pipeline.

    The predictor callback should accept an evaluator ``trip_id`` and return
    the existing decoder ``TripDecodeResult``. Keep loading, feature scoring,
    and decoding behind this boundary rather than inside FastAPI routes.
    """

    def __init__(self, predictor: Callable[[str], TripDecodeResult] | None = None) -> None:
        self._predictor = predictor

    def predict_trip(self, trip_id: str) -> TripDecodeResult:
        if self._predictor is None:
            raise PredictionUnavailable(
                "Prediction pipeline is not configured: the trained model scorer and "
                "model-to-HMM/Viterbi integration are not available."
            )
        result = self._predictor(trip_id)
        if not isinstance(result, TripDecodeResult):
            raise TypeError("The prediction pipeline must return TripDecodeResult.")
        return result
