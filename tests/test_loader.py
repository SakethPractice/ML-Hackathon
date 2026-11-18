"""Loader and validation tests on small hand-written trips."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathsnap.data.preprocessing import TrackContextConfig, track_context
from pathsnap.data.loader import list_trips, load_trip, read_arcs, read_track
from pathsnap.data.validation import validate_route, validate_track, validate_trip
from pathsnap.graph.geo import LocalProjection


def _write_trip(root: Path, partition: str, trip_id: str, route=True, track_text=None):
    d = root / partition / trip_id
    d.mkdir(parents=True)
    (d / f"{trip_id}.track").write_text(
        track_text or "8.0\t49.0\t1.000000\r\n8.0001\t49.0001\t2.000000\r\n8.0002\t49.0002\t4.000000\r\n"
    )
    (d / f"{trip_id}.nodes").write_text("8.0\t49.0\n8.001\t49.0\n8.002\t49.0\n")
    (d / f"{trip_id}.arcs").write_text("0\t1\n1\t0\n1\t2\n")
    if route:
        (d / f"{trip_id}.route").write_text("0\n2\n")
    return d


def test_track_columns_and_crlf(tmp_path):
    d = _write_trip(tmp_path, "train", "00000001")
    tr = read_track(d / "00000001.track")
    assert list(tr.columns) == ["lon", "lat", "timestamp"]
    assert len(tr) == 3
    # column order is lon, lat - not lat, lon
    assert tr.loc[0, "lon"] == pytest.approx(8.0)
    assert tr.loc[0, "lat"] == pytest.approx(49.0)
    assert tr["timestamp"].tolist() == [1.0, 2.0, 4.0]


def test_ids_are_zero_based_row_positions(tmp_path):
    d = _write_trip(tmp_path, "train", "00000001")
    arcs = read_arcs(d / "00000001.arcs")
    assert arcs.index.tolist() == [0, 1, 2]
    assert arcs.loc[2].tolist() == [1, 2]
    assert arcs.dtypes.tolist() == [np.int64, np.int64]


def test_load_trip_and_route_toggle(tmp_path):
    _write_trip(tmp_path, "train", "00000001")
    trip = load_trip("train", "00000001", data_root=tmp_path)
    assert trip.has_route and trip.route["arc_id"].tolist() == [0, 2]
    blind = load_trip("train", "00000001", data_root=tmp_path, with_route=False)
    assert blind.route is None


def test_missing_route_is_allowed_missing_track_is_not(tmp_path):
    d = _write_trip(tmp_path, "student_test", "00000005", route=False)
    trip = load_trip("student_test", "00000005", data_root=tmp_path)
    assert not trip.has_route
    (d / "00000005.track").unlink()
    with pytest.raises(FileNotFoundError):
        load_trip("student_test", "00000005", data_root=tmp_path)


def test_list_trips(tmp_path):
    _write_trip(tmp_path, "train", "00000003")
    _write_trip(tmp_path, "train", "00000001")
    assert list_trips("train", data_root=tmp_path) == ["00000001", "00000003"]
    assert list_trips("evaluator_test", data_root=tmp_path) == []


def test_validate_track_gaps_and_order(tmp_path):
    text = "8.0 49.0 1\n8.0 49.0 2\n8.1 49.1 2\n8.2 49.2 40\n8.3 49.3 39\n"
    d = _write_trip(tmp_path, "train", "00000001", track_text=text)
    rep = validate_track(read_track(d / "00000001.track"))
    assert rep["n_duplicate_timestamps"] == 1
    assert rep["n_duplicate_positions"] == 1
    assert rep["n_non_increasing_steps"] == 2  # 2->2 and 40->39
    assert rep["timestamps_sorted"] is False
    assert rep["n_gaps_gt_30s"] == 1


def test_validate_route_breaks_and_invalid_refs(tmp_path):
    _write_trip(tmp_path, "train", "00000001")
    trip = load_trip("train", "00000001", data_root=tmp_path)
    # arc 0 (0->1) then arc 2 (1->2): connected
    assert validate_route(trip.route, trip.arcs)["route_n_breaks"] == 0
    # arc 2 (1->2) then arc 0 (0->1): break
    trip.route["arc_id"] = [2, 0]
    assert validate_route(trip.route, trip.arcs)["route_n_breaks"] == 1
    trip.route["arc_id"] = [0, 99]
    rep = validate_route(trip.route, trip.arcs)
    assert rep["route_n_invalid_arc_ids"] == 1


def test_validate_trip_flags_bad_node_reference(tmp_path):
    d = _write_trip(tmp_path, "train", "00000001")
    (d / "00000001.arcs").write_text("0\t1\n1\t7\n")
    rep = validate_trip(load_trip("train", "00000001", data_root=tmp_path))
    assert rep["n_arcs_invalid_node_ref"] == 1


def test_track_context_gaps_heading_and_stationary():
    proj = LocalProjection(8.0, 49.0)
    # 20 fixes moving east at 10 m/s, then 40 s gap, then 20 fixes standing still
    xy = np.vstack([np.column_stack([np.arange(20) * 10.0, np.zeros(20)]),
                    np.tile([[500.0, 0.0]], (20, 1))])
    ts = np.r_[np.arange(20), 59 + np.arange(20)].astype(float)
    lonlat = proj.to_lonlat(xy)
    track = pd.DataFrame({"lon": lonlat[:, 0], "lat": lonlat[:, 1], "timestamp": ts})
    ctx = track_context(track, proj, TrackContextConfig(gap_s=30, stationary_window=5, stationary_disp_m=20))

    assert np.isnan(ctx.loc[0, "dt_prev_s"])
    assert ctx.loc[5, "speed_prev_mps"] == pytest.approx(10.0, rel=1e-3)
    assert ctx.loc[5, "heading_deg"] == pytest.approx(90.0, abs=0.1)  # east
    assert ctx["gap_before"].sum() == 1 and bool(ctx.loc[20, "gap_before"])
    assert ctx["segment_id"].tolist() == [0] * 20 + [1] * 20
    assert not ctx.loc[10, "is_stationary"]
    assert ctx.loc[30, "is_stationary"]
    assert np.isnan(ctx.loc[30, "heading_deg"])
