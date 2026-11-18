"""FastAPI application entry point for PathSnap."""

from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .schemas import PredictRequest, PredictionResponse, TripListResponse
from .service import (
    EvaluatorTripsUnavailable,
    PredictionService,
    PredictionUnavailable,
    list_evaluator_trip_ids,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("pathsnap.api")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Log API lifecycle events; future service initialization belongs here."""
    logger.info("PathSnap API starting")
    yield
    logger.info("PathSnap API stopping")


app = FastAPI(
    title="PathSnap API",
    description="Backend API foundation for the PathSnap route evaluation project.",
    version="0.1.0",
    lifespan=lifespan,
)
app.state.prediction_service = PredictionService()


def get_prediction_service(request: Request) -> PredictionService:
    """Dependency seam for configuring and mocking the inference service."""
    return request.app.state.prediction_service


@app.get("/health", tags=["system"], summary="Check API health")
async def health() -> dict[str, str]:
    """Report that the API process is running."""
    return {"status": "ok"}


@app.get("/trips", response_model=TripListResponse, tags=["evaluation"])
def get_trips() -> TripListResponse:
    """Return evaluator IDs discovered from configured raw inference inputs."""
    try:
        return TripListResponse(trips=list_evaluator_trip_ids())
    except EvaluatorTripsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["evaluation"],
    summary="Decode one evaluator trip",
)
def predict(
    request_body: PredictRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    """Run the configured predictor and serialize its real decoder result."""
    try:
        available_trip_ids = list_evaluator_trip_ids()
    except EvaluatorTripsUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if request_body.trip_id not in available_trip_ids:
        raise HTTPException(status_code=404, detail="Trip ID is not in the evaluator partition.")

    try:
        result = prediction_service.predict_trip(request_body.trip_id)
    except PredictionUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return PredictionResponse(**result.to_dict())


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected failures and return a stable, non-sensitive JSON error."""
    logger.error(
        "Unhandled API error for %s %s",
        request.method,
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
