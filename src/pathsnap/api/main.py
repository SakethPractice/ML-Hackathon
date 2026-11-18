"""FastAPI application entry point for PathSnap.

Run on the backend laptop:

    python -m uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
import logging
import time
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .schemas import HealthResponse, PredictRequest, PredictionResponse, TripListResponse
from .service import (
    EvaluatorTripsUnavailable,
    PredictionService,
    PredictionUnavailable,
    evaluate_trip,
    get_mode,
    list_evaluator_trip_ids,
    _load_data_settings,
    model_path,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("pathsnap.api")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        mode = get_mode()
        _, partition = _load_data_settings()
    except EvaluatorTripsUnavailable as exc:
        mode, partition = "invalid", str(exc)
    logger.info("PathSnap API starting: mode=%s partition=%s model=%s", mode, partition, model_path())
    yield
    logger.info("PathSnap API stopping")


app = FastAPI(
    title="PathSnap API",
    description="Map-matching and route reconstruction for PathSnap trips.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.prediction_service = PredictionService(use_default_pipeline=True)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    client = request.client.host if request.client else "-"
    logger.info("%s %s from %s -> %d (%.2fs)", request.method, request.url.path,
                client, response.status_code, time.perf_counter() - started)
    return response


def get_prediction_service(request: Request) -> PredictionService:
    """Dependency seam for configuring and mocking the inference service."""
    return request.app.state.prediction_service


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    """Report that the API is running, which partition it serves and the model."""
    try:
        mode = get_mode()
        _, partition = _load_data_settings()
    except EvaluatorTripsUnavailable:
        mode, partition = "invalid", None
    return HealthResponse(
        status="ok", mode=mode, partition=partition, model=model_path().stem,
        model_loaded=request.app.state.prediction_service.model_loaded,
    )


@app.get("/trips", response_model=TripListResponse, tags=["evaluation"])
def get_trips() -> TripListResponse:
    """Return trip IDs found in the served partition."""
    try:
        return TripListResponse(mode=get_mode(), trips=list_evaluator_trip_ids())
    except EvaluatorTripsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse, tags=["evaluation"])
def predict(
    request_body: PredictRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """Run candidates -> classifier -> HMM/Viterbi for one trip."""
    trip_id = request_body.trip_id
    try:
        available_trip_ids = list_evaluator_trip_ids()
    except EvaluatorTripsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if trip_id not in available_trip_ids:
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} is not in the served partition.")

    logger.info("predict: trip %s", trip_id)
    try:
        prediction = prediction_service.predict_trip(trip_id)
    except PredictionUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    evaluation = None
    if prediction.observations:  # only for the real pipeline
        try:
            evaluation = evaluate_trip(trip_id, prediction.result)
        except Exception:  # evaluation must never break a prediction
            logger.exception("evaluation failed for trip %s", trip_id)
    if evaluation:
        logger.info("trip %s: evaluation %s", trip_id, evaluation)

    return PredictionResponse(
        **prediction.result.to_dict(),
        observations=prediction.observations,
        timings=prediction.timings,
        evaluation=evaluation,
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected failures and return a stable, non-sensitive JSON error."""
    logger.error(
        "Unhandled API error for %s %s",
        request.method,
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
