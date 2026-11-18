"""Spatial index, candidate generation and route alignment tests."""

import numpy as np
import pandas as pd
import pytest

from pathsnap.candidates.generator import CandidateConfig, CandidateGenerator
from pathsnap.data.route_alignment import align_track_to_route
from pathsnap.graph.geo import LocalProjection
from pathsnap.graph.road_graph import RoadGraph
from pathsnap.graph.spatial_index import ArcIndex


def _metric_graph(node_xy, arcs):
    """Build a graph from node coordinates given directly in metres around (8E, 49N)."""
    proj = LocalProjection(8.0, 49.0)
    lonlat = proj.to_lonlat(np.asarray(node_xy, dtype=float))
    return RoadGraph.from_frames(
        pd.DataFrame(lonlat, columns=["lon", "lat"]),
        pd.DataFrame(arcs, columns=["from_node", "to_node"]),
        projection=proj,
    )


# A long east-west road (2 km, one arc each way) and a short north-south
# road crossing it at x = 1000.
#   arc 0: (0,0) -> (2000,0)    arc 1: reverse of arc 0
#   arc 2: (1000,-100) -> (1000,100)
NODE_XY = [[0, 0], [2000, 0], [1000, -100], [1000, 100], [5000, 5000], [5001, 5000]]
ARCS = [[0, 1], [1, 0], [2, 3], [4, 5]]


def test_index_finds_middle_of_long_arc():
    g = _metric_graph(NODE_XY, ARCS)
    idx = ArcIndex(g, step_m=25)
    # 20 m north of the long road, 500 m from both of its nodes
    found = idx.arcs_within(np.array([[500.0, 20.0]]), radius_m=30)[0]
    assert sorted(found.tolist()) == [0, 1]
    # outside the radius
    assert len(idx.arcs_within(np.array([[500.0, 40.0]]), radius_m=30)[0]) == 0


def test_index_matches_brute_force():
    g = _metric_graph(NODE_XY, ARCS)
    idx = ArcIndex(g, step_m=25)
    rng = np.random.default_rng(0)
    pts = rng.uniform([-100, -150], [2100, 150], size=(200, 2))
    found = idx.arcs_within(pts, radius_m=60)
    for p, f in zip(pts, found):
        d, _, _ = idx.distances(np.broadcast_to(p, (g.n_arcs, 2)), np.arange(g.n_arcs))
        assert sorted(f.tolist()) == np.flatnonzero(d <= 60).tolist()


def test_candidates_sorted_ranked_and_cut():
    g = _metric_graph(NODE_XY, ARCS)
    gen = CandidateGenerator(g, CandidateConfig(radius_m=50, top_k=2))
    cs = gen.generate(np.array([[995.0, 10.0]]))
    t = cs.table
    # arc 2 is 5 m away, arcs 0/1 are 10 m away; top_k=2 keeps arc 2 and the lower arc ID of the tie
    assert t["arc_id"].tolist() == [2, 0]
    assert t["rank"].tolist() == [0, 1]
    assert t["distance_m"].tolist() == pytest.approx([5.0, 10.0])
    assert (t["radius_used_m"] == 50).all()


def test_fallback_radius_and_empty_obs():
    g = _metric_graph(NODE_XY, ARCS)
    gen = CandidateGenerator(g, CandidateConfig(radius_m=50, top_k=4, max_radius_m=400))
    pts = np.array([[500.0, 0.0], [500.0, 150.0], [3000.0, 3000.0]])
    cs = gen.generate(pts)
    t = cs.table
    assert set(t.loc[t.obs_id == 0, "radius_used_m"]) == {50}
    # 150 m away -> found at the second expansion (200 m)
    assert set(t.loc[t.obs_id == 1, "radius_used_m"]) == {200}
    # far from everything -> no candidates, reported as empty
    assert cs.empty_obs.tolist() == [2]


def test_candidate_generation_is_deterministic():
    g = _metric_graph(NODE_XY, ARCS)
    gen = CandidateGenerator(g, CandidateConfig(radius_m=80, top_k=3))
    pts = np.random.default_rng(1).uniform([0, -60], [2000, 60], size=(50, 2))
    pd.testing.assert_frame_equal(gen.generate(pts).table, gen.generate(pts).table)


def test_contains():
    g = _metric_graph(NODE_XY, ARCS)
    cs = CandidateGenerator(g, CandidateConfig(radius_m=50, top_k=8)).generate(np.array([[500.0, 5.0]]))
    assert cs.contains([0, 0, 0], [0, 1, 2]).tolist() == [True, True, False]


def test_alignment_is_monotone_and_handles_loops():
    # square loop driven twice: 0->1->2->3->0->1->2->3
    node_xy = [[0, 0], [100, 0], [100, 100], [0, 100]]
    arcs = [[0, 1], [1, 2], [2, 3], [3, 0]]
    g = _metric_graph(node_xy, arcs)
    route = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    # walk around the square twice, 5 points per side, 2 m off the road
    side = np.linspace(10, 90, 5)
    one_lap = np.concatenate([
        np.column_stack([side, np.full(5, -2)]),
        np.column_stack([np.full(5, 102), side]),
        np.column_stack([side[::-1], np.full(5, 102)]),
        np.column_stack([np.full(5, -2), side[::-1]]),
    ])
    xy = np.vstack([one_lap, one_lap])
    al = align_track_to_route(xy, route, g)
    assert np.all(np.diff(al["route_pos"]) >= 0)
    assert al["route_pos"].tolist() == np.repeat(np.arange(8), 5).tolist()
    assert al["ref_arc_id"].tolist() == np.repeat(route, 5).tolist()
    assert al["align_dist_m"].to_numpy() == pytest.approx(2.0)


def test_prepare_trip_without_labels_never_reads_route(tmp_path, monkeypatch):
    import pathsnap.data.loader as loader
    from pathsnap.candidates.pipeline import DataConfig, prepare_trip

    d = tmp_path / "student_test" / "00000001"
    d.mkdir(parents=True)
    proj = LocalProjection(8.0, 49.0)
    nodes = proj.to_lonlat(np.array([[0.0, 0.0], [500.0, 0.0]]))
    (d / "00000001.nodes").write_text("\n".join(f"{x:.7f} {y:.7f}" for x, y in nodes) + "\n")
    (d / "00000001.arcs").write_text("0 1\n1 0\n")
    track = proj.to_lonlat(np.array([[100.0, 5.0], [200.0, 5.0], [300.0, 5.0]]))
    (d / "00000001.track").write_text("\n".join(f"{x:.7f} {y:.7f} {i + 1}" for i, (x, y) in enumerate(track)) + "\n")
    (d / "00000001.route").write_text("0\n")

    def boom(*a, **k):
        raise AssertionError("route was read")

    monkeypatch.setattr(loader, "read_route", boom)
    pt = prepare_trip("student_test", "00000001", DataConfig(data_root=str(tmp_path)))
    assert pt.labels is None and pt.trip.route is None
    assert set(pt.candidates.table["arc_id"]) == {0, 1}
    assert len(pt.context) == 3


def test_load_data_config_reads_repo_config():
    from pathsnap.candidates.pipeline import load_data_config

    cfg = load_data_config()
    assert cfg.candidates.radius_m > 0 and cfg.candidates.top_k > 0
