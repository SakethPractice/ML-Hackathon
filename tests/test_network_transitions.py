import numpy as np
import pytest
from pathsnap.decoder.network import DirectedRoadNetwork,haversine_m
from pathsnap.decoder.types import ObservationStates,DecodedState
from pathsnap.decoder.config import TransitionConfig,DecoderConfig
from pathsnap.decoder.transitions import calculate_transition_scores,state_pair_distance,is_feasible,search_bound_m
from pathsnap.decoder.route_reconstruction import reconstruct_route

def _state(arc,frac,t):
    z=np.zeros(1)
    return ObservationStates(0,77.59,12.97,t,np.array([arc]),np.array([frac]),z,np.array([.5]),z)

def test_haversine_uses_metres():
    assert float(haversine_m(0,0,0,1))==pytest.approx(111195,rel=1e-3)

def test_network_arc_lengths(net):
    assert 100 < net.arc_length[0] < 115
    assert net.n_arcs==8

def test_directed_path_and_parallel(net):
    assert net.shortest_node_path(0,3,1000)==[0,1,2,3]
    assert net.shortest_node_path(0,6,10000) is None
    assert net.path_arc_ids([0,1,2])==[0,1]

def test_cache_bucket_does_not_violate_cutoff(net):
    d=float(net.arc_length[1])
    # 1->2 is in the bucketed Dijkstra search but exceeds specified bound
    assert net.shortest_node_path(1,2,d/2) is None
    assert net.shortest_node_path(1,2,d*1.5)==[1,2]

def test_same_arc_forward_distance(net):
    d=state_pair_distance(net,1,.3,1,.8,500,5)
    assert d==pytest.approx(net.arc_length[1]*.5)

def test_directed_multiarc_distance(net):
    d=state_pair_distance(net,0,.5,2,.5,1000,5)
    expected=net.arc_length[0]*.5+net.arc_length[1]+net.arc_length[2]*.5
    assert d==pytest.approx(expected)

def test_disconnected_transition_rejected(net):
    prev=_state(0,.5,0)
    cur=_state(5,.5,10)
    scores, distances=calculate_transition_scores(prev,cur,net,TransitionConfig())
    assert np.isneginf(scores[0,0]) and np.isinf(distances[0,0])

def test_speed_constraint(net):
    prev=_state(0,.5,0)
    cur=_state(3,.5,1)
    scores,distances=calculate_transition_scores(prev,cur,net,TransitionConfig())
    assert np.isneginf(scores[0,0])

def test_route_reconstruction_inserts_intermediate(net):
    seg=[[DecodedState(0,0,.5,0.),DecodedState(1,2,.5,10.)]]
    out=reconstruct_route(seg,net)
    assert out.segments==[[0,1,2]]
    assert out.n_inserted_arcs==1

def test_route_reconstruction_splits_when_too_fast(net):
    seg=[[DecodedState(0,0,.5,0.),DecodedState(1,3,.5,0.1)]]
    out=reconstruct_route(seg,net)
    assert len(out.segments)==2
    assert out.gap_boundaries[0]['reason']=='no_path_in_reconstruction'

def test_distance_bound_exact(net):
    assert search_bound_m(1.,TransitionConfig())==pytest.approx(145.)
    assert is_feasible(40,1,TransitionConfig())
    assert not is_feasible(50,1,TransitionConfig())
