"""Processed-table export: round trip, label separation, feature-extraction compatibility."""

import numpy as np
import pandas as pd
import pytest

from pathsnap.candidates.pipeline import DataConfig, prepare_trip
from pathsnap.data.processed import (
    CANDIDATE_LABEL_FILE,
    LABEL_FILE,
    load_partition,
    load_processed,
    save_processed,
)
from pathsnap.features import extract_features, get_feature_matrix
from pathsnap.graph.geo import LocalProjection

LABEL_COLUMNS = {"ref_arc_id", "route_pos", "align_dist_m", "label_valid", "is_reference"}


def _write_trip(root, partition="train", trip_id="00000001", route=True):
    """Two-way east-west road (arcs 0/1) and a north-going side road (arc 2)."""
    d = root / partition / trip_id
    d.mkdir(parents=True)
    proj = LocalProjection(8.0, 49.0)
    nodes = proj.to_lonlat(np.array([[0.0, 0.0], [400.0, 0.0], [200.0, 0.0], [200.0, 300.0]]))
    (d / f"{trip_id}.nodes").write_text("\n".join(f"{x:.8f} {y:.8f}" for x, y in nodes) + "\n")
    (d / f"{trip_id}.arcs").write_text("0 1\n1 0\n2 3\n")
    xy = np.column_stack([np.linspace(20, 380, 10), np.full(10, 4.0)])
    pts = proj.to_lonlat(xy)
    (d / f"{trip_id}.track").write_text(
        "\n".join(f"{x:.8f} {y:.8f} {i + 1}" for i, (x, y) in enumerate(pts)) + "\n")
    if route:
        (d / f"{trip_id}.route").write_text("0\n")
    return d


def _prepare(tmp_path, **kw):
    cfg = DataConfig(data_root=str(tmp_path / "raw"))
    return prepare_trip(kw.get("partition", "train"), "00000001", cfg,
                        with_labels=kw.get("with_labels", True))


def test_round_trip_and_columns(tmp_path):
    _write_trip(tmp_path / "raw")
    pt = _prepare(tmp_path)
    row = save_processed(pt, tmp_path / "processed")
    assert row["n_observations"] == 10 and row["has_labels"]

    t = load_processed("train", "00000001", tmp_path / "processed", with_labels=True)
    gps, cand = t["gps_observations"], t["candidate_segments"]
    assert gps["trip_id"].iloc[0] == "00000001"  # leading zeros kept
    assert gps["observation_id"].tolist() == list(range(10))
    assert np.allclose(gps["timestamp"], np.arange(1, 11))
    assert len(cand) == len(pt.candidates.table)
    # both directions of the main road are candidates for every fix
    assert set(cand.loc[cand.observation_id == 0, "arc_id"]) >= {0, 1}
    assert cand["segment_bearing_deg"].between(0, 360).all()
    # arc 0 runs east (90 deg), arc 1 west (270 deg)
    b = cand.drop_duplicates("arc_id").set_index("arc_id")["segment_bearing_deg"]
    assert b[0] == pytest.approx(90, abs=0.01) and b[1] == pytest.approx(270, abs=0.01)

    lab, cl = t["labels"], t["candidate_labels"]
    assert (lab["ref_arc_id"] == 0).all() and lab["label_valid"].all()
    # exactly one candidate row per fix is the reference
    assert cl.groupby("observation_id")["is_reference"].sum().eq(1).all()


def test_input_tables_carry_no_labels(tmp_path):
    _write_trip(tmp_path / "raw")
    save_processed(_prepare(tmp_path), tmp_path / "processed")
    t = load_processed("train", "00000001", tmp_path / "processed")
    assert set(t) == {"gps_observations", "candidate_segments"}
    for df in t.values():
        assert not LABEL_COLUMNS & set(df.columns)


def test_unlabelled_export_writes_no_label_files(tmp_path):
    _write_trip(tmp_path / "raw", partition="student_test")
    pt = _prepare(tmp_path, partition="student_test", with_labels=False)
    row = save_processed(pt, tmp_path / "processed")
    d = tmp_path / "processed" / "student_test" / "00000001"
    assert not row["has_labels"]
    assert not (d / LABEL_FILE).exists() and not (d / CANDIDATE_LABEL_FILE).exists()


def test_compatible_with_feature_extraction(tmp_path):
    _write_trip(tmp_path / "raw")
    pt = _prepare(tmp_path)
    save_processed(pt, tmp_path / "processed")
    t = load_partition("train", tmp_path / "processed")
    feats = extract_features(t["gps_observations"], t["candidate_segments"], pt.graph.to_networkx())
    assert len(feats) == len(t["candidate_segments"])
    X = get_feature_matrix(feats)
    # node degrees come through from the graph conversion
    assert X["from_node_out_degree"].notna().all()


def test_to_networkx_keeps_ids_and_duplicates():
    from pathsnap.graph.road_graph import RoadGraph

    g = RoadGraph.from_frames(
        pd.DataFrame([[8.0, 49.0], [8.001, 49.0], [8.002, 49.0]], columns=["lon", "lat"]),
        pd.DataFrame([[0, 1], [1, 0], [1, 2], [1, 2]], columns=["from_node", "to_node"]),
    )
    nxg = g.to_networkx()
    assert nxg.is_directed() and nxg.number_of_edges() == 4
    assert nxg.out_degree(1) == 3  # 1->0 and the duplicated 1->2
    assert set(k for _, _, k in nxg.edges(keys=True)) == {0, 1, 2, 3}
