"""PathSnap evaluator UI (Laptop 2).

    $env:PATHSNAP_API_BASE_URL = "http://<backend-ip>:8000"
    streamlit run ui/app.py
"""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st
from streamlit_folium import st_folium

from config import get_api_base_url
from map_adapter import create_route_map, parse_prediction_coordinates


st.set_page_config(page_title="PathSnap Evaluator", page_icon="🧭", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background: #ffffff; }
      .block-container { max-width: 1200px; padding-top: 2rem; }
      h1 { font-weight: 700; letter-spacing: -.02em; margin-bottom: 0; }
      [data-testid="stMetric"] { background: #f8fafc; border: 1px solid #e5e7eb;
        border-radius: 10px; padding: .6rem .9rem; }
      [data-testid="stMetricValue"] { font-size: 1.4rem; }
      .pill { display: inline-block; padding: .15rem .6rem; border-radius: 999px;
        font-size: .85rem; font-weight: 600; }
      .ok { background: #dcfce7; color: #166534; }
      .down { background: #fee2e2; color: #991b1b; }
    </style>
    """,
    unsafe_allow_html=True,
)

API = get_api_base_url()
STATUS_TEXT = {
    "success": "Success",
    "success_with_unresolved": "Success (some fixes unresolved)",
    "split": "Split at gaps",
    "failed": "Failed",
}


def call_api(method: str, path: str, *, timeout: float, **kwargs: Any) -> Any:
    """Return JSON, or raise RuntimeError with the backend's own message."""
    try:
        response = requests.request(method, f"{API}{path}", timeout=timeout, **kwargs)
    except requests.Timeout as exc:
        raise RuntimeError(f"{path}: no response within {timeout:g} s.") from exc
    except requests.ConnectionError as exc:
        raise RuntimeError(f"Cannot reach the backend at {API}.") from exc
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"HTTP {response.status_code} from {path}: {detail}")
    return response.json()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Backend")
    st.code(API, language=None)
    try:
        health = call_api("GET", "/health", timeout=4)
        st.markdown("<span class='pill ok'>Connected</span>", unsafe_allow_html=True)
        st.caption(f"Mode: **{health.get('mode')}** · partition: **{health.get('partition')}**")
        st.caption(f"Model: **{health.get('model')}**")
    except RuntimeError as exc:
        health = None
        st.markdown("<span class='pill down'>Disconnected</span>", unsafe_allow_html=True)
        st.caption(str(exc))
    st.caption("Set PATHSNAP_API_BASE_URL before starting the UI to use another backend.")
    if st.button("Recheck", use_container_width=True):
        st.session_state.pop("trips", None)
        st.rerun()

# --------------------------------------------------------------------- main
st.title("PathSnap")
st.caption("Map-matching and route reconstruction. Pick a trip; the backend predicts its route.")

if "trips" not in st.session_state and health is not None:
    try:
        st.session_state.trips = call_api("GET", "/trips", timeout=10).get("trips", [])
        st.session_state.trips_error = None
    except RuntimeError as exc:
        st.session_state.trips, st.session_state.trips_error = [], str(exc)

trips = st.session_state.get("trips", [])
if st.session_state.get("trips_error"):
    st.warning(st.session_state.trips_error)
elif health is not None and not trips:
    st.warning("The backend returned no trips.")

col_trip, col_btn = st.columns([4, 1])
trip_id = col_trip.selectbox("Trip", trips, index=None if not trips else 0,
                             placeholder="No trips available", disabled=not trips)
col_btn.write("")
col_btn.write("")
run = col_btn.button("Predict route", type="primary", disabled=not trip_id,
                     use_container_width=True)

if run and trip_id:
    with st.spinner(f"Predicting trip {trip_id} on the backend…"):
        try:
            st.session_state.prediction = call_api(
                "POST", "/predict", json={"trip_id": trip_id}, timeout=180)
            st.session_state.prediction_error = None
        except RuntimeError as exc:
            st.session_state.prediction, st.session_state.prediction_error = None, str(exc)

if st.session_state.get("prediction_error"):
    st.error(st.session_state.prediction_error)

pred = st.session_state.get("prediction")
if pred:
    statuses = pred.get("observation_status", [])
    matched = sum(a is not None for a in pred.get("matched_arc_ids", []))
    st.subheader(f"Trip {pred['trip_id']}")
    m = st.columns(5)
    m[0].metric("Status", STATUS_TEXT.get(pred["status"], pred["status"]))
    m[1].metric("GPS fixes", len(statuses))
    m[2].metric("Matched fixes", f"{matched}")
    m[3].metric("Route segments", len(pred.get("route_segments", [])))
    m[4].metric("Backend time", f"{sum(pred.get('timings', {}).values()) or pred['runtime_seconds']:.1f} s")

    ev = pred.get("evaluation")
    if ev:
        e = st.columns(3)
        pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}%"  # noqa: E731
        e[0].metric("Pointwise accuracy", pct(ev.get("pointwise_accuracy")),
                    help=f"Over {ev['labelled_observations']} labelled fixes (server-side).")
        e[1].metric("Route arc recall", pct(ev.get("route_arc_recall")))
        e[2].metric("Route arc precision", pct(ev.get("route_arc_precision")))
        st.caption("Scores computed on the backend against the reference route "
                   f"({ev['partition']}). The reference itself is not sent to the UI.")

    try:
        gps, segments = parse_prediction_coordinates(pred)
    except ValueError as exc:
        st.warning(f"Map unavailable: {exc}")
    else:
        if segments or gps:
            st_folium(create_route_map(gps, segments), use_container_width=True, height=560,
                      returned_objects=[])
        else:
            st.info("The decoder returned no route geometry for this trip.")

    with st.expander("Raw API response"):
        st.json(pred, expanded=False)
else:
    st.info("Select a trip and press **Predict route**.")
