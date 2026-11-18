"""FastAPI application entry point for PathSnap."""

from contextlib import asynccontextmanager
import logging
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


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


@app.get("/health", tags=["system"], summary="Check API health")
async def health() -> dict[str, str]:
    """Report that the API process is running."""
    return {"status": "ok"}


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
