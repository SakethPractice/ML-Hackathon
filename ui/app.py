"""PathSnap evaluator dashboard."""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st
from streamlit_folium import st_folium

from config import get_api_base_url
from map_adapter import create_route_map


st.set_page_config(page_title="PathSnap | Evaluator", page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      .stApp { background: #f2f6f8; color: #142b3b; }
      [data-testid="stHeader"] { background: rgba(242,246,248,.94); }
      .block-container { max-width: 1240px; padding: 1.6rem 2rem 3rem; }
      h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] { color: #142b3b; }
      h2, h3 { letter-spacing: -.02em; }
      .hero { padding: 1.65rem 1.9rem; border-radius: 16px; color: #fff;
        background: linear-gradient(115deg, #102a43 0%, #123d50 62%, #087e83 100%);
        margin-bottom: 1.25rem; box-shadow: 0 10px 28px rgba(16,42,67,.15); }
      .hero h1 { margin: 0; color: #fff; font-size: 2rem; line-height: 1.2; }
      .hero p { margin: .5rem 0 0; color: #e4f2f3; font-size: 1rem; }
      [data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-color: #d7e2e8;
        border-radius: 14px; }
      [data-testid="stCaptionContainer"] p { color: #496273; }
      [data-testid="stAlert"] { border-radius: 10px; }
      [data-testid="stAlert"] p { color: #142b3b; }
      [data-testid="stSelectbox"] label, [data-testid="stWidgetLabel"] p {
        color: #19384a !important; font-weight: 650; }
      [data-baseweb="select"] > div { background: #fff; border-color: #9eb4c0; }
      [data-baseweb="select"] * { color: #142b3b; }
      .stButton > button { min-height: 2.75rem; border-radius: 9px; font-weight: 650;
        border: 1px solid #087e83; color: #075c62; background: #fff; }
      .stButton > button:hover { border-color: #075c62; color: #064b50; background: #e9f6f5; }
      .stButton > button[kind="primary"] { color: #fff; background: #087e83;
        border-color: #087e83; }
      .stButton > button[kind="primary"] p { color: #fff; }
      .stButton > button[kind="primary"]:hover { background: #06686d; border-color: #06686d; }
      [data-testid="stCode"] { border: 1px solid #d7e2e8; border-radius: 8px; }
      [data-testid="stJson"] { background: #f8fafb; border: 1px solid #e0e9ed;
        border-radius: 10px; padding: .5rem; }
      hr { border-color: #d7e2e8; margin: 1.5rem 0; }
      @media (max-width: 760px) {
        .block-container { padding: 1rem 1rem 2rem; }
        .hero { padding: 1.3rem 1.2rem; }
        .hero h1 { font-size: 1.65rem; }
      }
    </style>
    <div class="hero"><h1>PathSnap Evaluator</h1>
    <p>Inspect backend availability and request a route prediction for an evaluator trip.</p></div>
    """,
    unsafe_allow_html=True,
)


def request_json(method: str, url: str, *, timeout: float = 8, **kwargs: Any) -> Any:
    """Send an API request and return valid JSON, raising readable errors otherwise."""
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach the backend: {exc}") from exc
    if not response.ok:
        detail = response.text.strip() or "No error detail returned."
        raise RuntimeError(f"Backend returned HTTP {response.status_code}: {detail[:500]}")
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise RuntimeError("Backend response was not valid JSON.") from exc


def parse_trip_ids(payload: Any) -> list[str]:
    """Accept either a JSON list or an object with a `trips` list."""
    trips = payload.get("trips") if isinstance(payload, dict) else payload
    if not isinstance(trips, list):
        raise ValueError("Expected GET /trips to return a JSON list (or an object with a 'trips' list).")
    ids: list[str] = []
    for item in trips:
        trip_id = item.get("trip_id") if isinstance(item, dict) else item
        if isinstance(trip_id, (str, int)) and str(trip_id).strip():
            ids.append(str(trip_id))
        else:
            raise ValueError("Each trip must be a non-empty ID or an object containing 'trip_id'.")
    if not ids:
        raise ValueError("GET /trips returned no trip IDs.")
    return ids


def display_route_map(coordinates: list[list[float]]) -> None:
    """Render explicitly supplied route points; response mapping awaits API contract."""
    st_folium(create_route_map(coordinates), use_container_width=True, height=480)


api_base_url = get_api_base_url()
status_col, config_col = st.columns([1, 2])
with status_col.container(border=True):
    st.subheader("Backend connection")
    with st.spinner("Checking backend health…"):
        try:
            health = request_json("GET", f"{api_base_url}/health", timeout=4)
            st.success("CONNECTED · Backend is reachable")
            with st.expander("Health response"):
                st.json(health)
        except RuntimeError as exc:
            st.error(f"DISCONNECTED · {exc}")
with config_col.container(border=True):
    st.subheader("Connection target")
    st.code(api_base_url, language=None)
    st.caption("Set PATHSNAP_API_BASE_URL to connect to a backend on another laptop.")

st.divider()
with st.container(border=True):
    st.subheader("Evaluate a trip")
    st.caption("Choose an evaluator trip and request its route prediction.")
    refresh_col, selector_col = st.columns([1, 3])
    with refresh_col:
        refresh_trips = st.button("↻ Refresh trips", use_container_width=True)
if refresh_trips or "trip_ids" not in st.session_state:
    with st.spinner("Loading evaluator trips…"):
        try:
            st.session_state.trip_ids = parse_trip_ids(
                request_json("GET", f"{api_base_url}/trips")
            )
            st.session_state.trip_load_error = None
        except (RuntimeError, ValueError) as exc:
            st.session_state.trip_ids = []
            st.session_state.trip_load_error = str(exc)

if st.session_state.get("trip_load_error"):
    st.error(f"Unable to load trips: {st.session_state.trip_load_error}")

trip_ids = st.session_state.get("trip_ids", [])
with st.container(border=True):
    st.caption("Trip selection")
    with selector_col:
        selected_trip_id = st.selectbox(
            "Trip ID", options=trip_ids, disabled=not trip_ids,
            placeholder="Waiting for trips from the backend…",
        )
    run_prediction = st.button(
        "Run prediction", type="primary", disabled=not bool(trip_ids),
        use_container_width=True,
    )

if run_prediction and selected_trip_id:
    with st.spinner(f"Requesting prediction for trip {selected_trip_id}…"):
        try:
            prediction = request_json(
                "POST", f"{api_base_url}/predict", json={"trip_id": selected_trip_id}, timeout=60
            )
            st.session_state.prediction = prediction
            st.session_state.prediction_trip_id = selected_trip_id
            st.session_state.prediction_error = None
        except RuntimeError as exc:
            st.session_state.prediction_error = str(exc)

if st.session_state.get("prediction_error"):
    st.error(f"Prediction request failed: {st.session_state.prediction_error}")

st.divider()
prediction = st.session_state.get("prediction")
result_col, map_col = st.columns([1, 1], gap="large")
with result_col.container(border=True):
    st.subheader("Prediction result")
    if prediction is None:
        st.info("Prediction results will appear here after a successful API response.")
    else:
        st.caption(f"API response for trip {st.session_state.get('prediction_trip_id', '')}")
        st.json(prediction)

with map_col.container(border=True):
    st.subheader("Route map")
    if prediction is None:
        st.info("A map will be available after a prediction response and a defined route format.")
    else:
        st.info(
            "The API prediction response is shown above. Map rendering is ready in the separate "
            "adapter; the backend must define how route coordinates are represented before the UI "
            "can safely connect that response to the map."
        )
