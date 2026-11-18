"""PathSnap evaluator dashboard."""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st
from streamlit_folium import st_folium

from config import get_api_base_url
from map_adapter import create_route_map, parse_prediction_coordinates


# Illustrative UI-only points for checking map layers; never sent to the API.
SAMPLE_MAP_OBSERVATIONS = [
    (37.7739, -122.4312),
    (37.7750, -122.4284),
    (37.7761, -122.4257),
    (37.7774, -122.4229),
]
SAMPLE_MAP_ROUTE = [
    (37.7739, -122.4312),
    (37.7748, -122.4288),
    (37.7760, -122.4260),
    (37.7774, -122.4229),
]


st.set_page_config(page_title="PathSnap | Evaluator", page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      .stApp { background: #f3f6f8; color: #142b3b; }
      [data-testid="stHeader"] { background: rgba(243,246,248,.94); }
      [data-testid="stSidebar"] { background: #102a43; border-right: 1px solid #23465c; }
      [data-testid="stSidebar"] * { color: #e7f0f4; }
      [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #a9c2ce; }
      .block-container { max-width: 1440px; padding: 1rem 2rem 2rem; }
      h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] { color: #142b3b; }
      h2, h3 { letter-spacing: -.025em; }
      .hero { padding: 1.2rem 1.55rem; border-radius: 15px; color: #fff;
        background: linear-gradient(112deg, #102a43 0%, #143c53 68%, #087e83 100%);
        margin-bottom: .8rem; box-shadow: 0 8px 24px rgba(16,42,67,.13); }
      .hero h1 { margin: 0; color: #fff; font-size: 1.9rem; line-height: 1.15; }
      .hero p { margin: .35rem 0 0; color: #d4e8ec; font-size: .95rem; }
      .eyebrow { color: #6ee7d0; font-size: .7rem; font-weight: 750; letter-spacing: .16em;
        text-transform: uppercase; margin-bottom: .3rem; }
      .status-pill { display: inline-block; border-radius: 99px; padding: .35rem .7rem;
        background: #e5f6f0; color: #087e65; font-size: .78rem; font-weight: 750; }
      .status-down { background: #fff0ed; color: #b54732; }
      [data-testid="stVerticalBlockBorderWrapper"] { background: #fff; border-color: #dce5e9;
        border-radius: 13px; box-shadow: 0 2px 8px rgba(16,42,67,.035); }
      [data-testid="stCaptionContainer"] p { color: #607584; }
      [data-testid="stAlert"] { border-radius: 10px; }
      [data-testid="stAlert"] p { color: #142b3b; }
      [data-testid="stSelectbox"] label, [data-testid="stWidgetLabel"] p {
        color: #19384a !important; font-weight: 650; }
      [data-baseweb="select"] > div { background: #fff; border-color: #b8c9d0; border-radius: 9px; }
      [data-baseweb="select"] * { color: #142b3b; }
      .stButton > button { min-height: 2.55rem; border-radius: 9px; font-weight: 650;
        border: 1px solid #087e83; color: #075c62; background: #fff; }
      .stButton > button:hover { border-color: #075c62; color: #064b50; background: #e9f6f5; }
      .stButton > button[kind="primary"] { color: #fff; background: #087e83;
        border-color: #087e83; }
      .stButton > button[kind="primary"] p { color: #fff; }
      .stButton > button[kind="primary"]:hover { background: #06686d; border-color: #06686d; }
      .stButton > button:disabled, .stButton > button[kind="primary"]:disabled {
        background: #d9e1e5 !important; border-color: #d9e1e5 !important;
        color: #71818a !important; opacity: 1 !important; cursor: not-allowed !important;
        box-shadow: none !important; }
      .stButton > button:disabled p, .stButton > button[kind="primary"]:disabled p {
        color: #71818a !important; }
      [data-testid="stJson"] { background: #f8fafb; border: 1px solid #e0e9ed;
        border-radius: 10px; padding: .5rem; }
      hr { border-color: #dce5e9; margin: .9rem 0; }
      @media (max-width: 760px) {
        .block-container { padding: .8rem .8rem 1.5rem; }
        .hero { padding: 1rem; }
        .hero h1 { font-size: 1.65rem; }
      }
    </style>
    <div class="hero"><div class="eyebrow">PathSnap · Geospatial Analytics</div>
    <h1>Route Evaluation System</h1>
    <p>Explore map-matched routes against evaluator trip observations.</p></div>
    """,
    unsafe_allow_html=True,
)


def request_json(method: str, url: str, *, timeout: float = 8, **kwargs: Any) -> Any:
    """Send an API request and return valid JSON, raising readable errors otherwise."""
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.Timeout as exc:
        raise RuntimeError("The backend took too long to respond. Please retry in a moment.") from exc
    except requests.ConnectionError as exc:
        raise RuntimeError(
            "Backend is unreachable. Check the connection and retry."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError("The backend request could not be completed. Please retry.") from exc
    if not response.ok:
        if response.status_code == 404:
            detail = "The requested trip or API resource was not found. Refresh the trip list and retry."
        elif response.status_code == 503 and url.rstrip("/").endswith("/trips"):
            detail = "Evaluator dataset is not available yet. Please configure the evaluator_test dataset."
        elif response.status_code == 503:
            detail = "Prediction service is temporarily unavailable. Check evaluator data and retry."
        else:
            detail = f"The backend returned an unexpected response (HTTP {response.status_code})."
        raise RuntimeError(detail)
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise RuntimeError("Backend response could not be read. Please retry.") from exc


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


def display_route_map(
    observations: list[tuple[float, float]],
    predicted_route: list[tuple[float, float]],
    *,
    observations_label: str = "Original GPS observations",
    predicted_route_label: str = "Predicted route",
) -> None:
    """Render validated coordinates from the actual prediction response."""
    st_folium(
        create_route_map(
            observations,
            predicted_route,
            observations_label=observations_label,
            predicted_route_label=predicted_route_label,
        ),
        use_container_width=True,
        height=600,
    )


api_base_url = get_api_base_url()
with st.sidebar:
    st.markdown("<div class='eyebrow'>PATHSNAP</div>", unsafe_allow_html=True)
    st.markdown("### Route evaluation")
    st.caption("Geospatial analytics workspace")
    st.markdown("---")
    st.caption("Evaluator trips and route geometry are supplied by the connected API.")

with st.container(border=True):
    status_col, status_note_col = st.columns([1, 4], vertical_alignment="center")
    with status_col:
        st.markdown("**Backend**")
    with status_note_col:
        with st.spinner("Checking connection…"):
            try:
                request_json("GET", f"{api_base_url}/health", timeout=4)
                st.markdown("<span class='status-pill'>● Connected</span>", unsafe_allow_html=True)
            except RuntimeError as exc:
                st.markdown("<span class='status-pill status-down'>● Disconnected</span>", unsafe_allow_html=True)
                st.caption(str(exc))

with st.container(border=True):
    trip_title_col, refresh_col = st.columns([4, 1], vertical_alignment="center")
    with trip_title_col:
        st.subheader("Evaluate a trip")
        st.caption("Select an evaluator trip to request a route prediction.")
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
    st.warning(st.session_state.trip_load_error)

trip_ids = st.session_state.get("trip_ids", [])
with st.container(border=True):
    selector_col, action_col = st.columns([3, 1], vertical_alignment="bottom")
    with selector_col:
        selected_trip_id = st.selectbox(
            "Evaluator trip", options=trip_ids, disabled=not trip_ids,
            placeholder="Refresh trips when evaluator data is available…",
        )
    valid_trip_selected = (
        isinstance(selected_trip_id, str)
        and bool(selected_trip_id.strip())
        and selected_trip_id in trip_ids
    )
    with action_col:
        run_prediction = st.button(
            "Run prediction", type="primary",
            disabled=not valid_trip_selected,
            use_container_width=True,
        )

if run_prediction and valid_trip_selected:
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
    st.error(st.session_state.prediction_error)

prediction = st.session_state.get("prediction")
with st.container(border=True):
    result_heading, sample_toggle = st.columns([3, 1], vertical_alignment="center")
    with result_heading:
        st.subheader("Prediction result")
    with sample_toggle:
        show_sample_preview = st.checkbox("Sample map preview", help="Illustrative UI-only locations; not API data.")
    if prediction is None:
        st.caption("Prediction details will appear here after a successful API response.")
    else:
        try:
            observations, predicted_route = parse_prediction_coordinates(prediction)
        except ValueError as exc:
            st.warning(f"Prediction received; map geometry could not be displayed: {exc}")
        else:
            segment_count = len(predicted_route)
            point_count = sum(len(segment) for segment in predicted_route)
            metric_cols = st.columns(4)
            metric_cols[0].metric("Trip", str(st.session_state.get("prediction_trip_id", "—")))
            metric_cols[1].metric("Status", str(prediction.get("status", "—")).replace("_", " ").title())
            metric_cols[2].metric("Route points", point_count)
            metric_cols[3].metric("Segments", segment_count)

with st.container(border=True):
    st.subheader("Route map")
    st.caption("Toggle map layers to compare supplied GPS observations with the predicted route.")
    if show_sample_preview:
        st.markdown("**Sample map preview · illustrative only**")
        st.caption("These sample locations are not trip observations or model predictions.")
        display_route_map(
            SAMPLE_MAP_OBSERVATIONS, SAMPLE_MAP_ROUTE,
            observations_label="Sample GPS observations (preview)",
            predicted_route_label="Sample predicted route (preview)",
        )
    map_data = None
    if prediction is not None:
        try:
            map_data = parse_prediction_coordinates(prediction)
        except ValueError:
            map_data = None
    if map_data and map_data[1]:
        observations, predicted_route = map_data
        if not observations:
            st.caption("GPS observations are unavailable in this response; showing the predicted route only.")
        display_route_map(observations, predicted_route)
    elif prediction is not None:
        st.info("The response did not include route coordinates. A map will appear when route_geometry is available.")
    elif not show_sample_preview:
        st.info("Run a prediction to display its returned route geometry.")
