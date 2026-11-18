"""Configuration for the PathSnap evaluator UI."""

import os


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"


def get_api_base_url() -> str:
    """Return the backend URL, configurable per laptop through an environment variable."""
    return os.getenv("PATHSNAP_API_BASE_URL", DEFAULT_API_BASE_URL).strip().rstrip("/")
