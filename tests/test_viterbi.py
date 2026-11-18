import itertools
import numpy as np
import pytest
from pathsnap.decoder.viterbi import viterbi_decode
NEG = -np.inf

@pytest.mark.parametrize('seed',range(6))
def test_viterbi_matches_bruteforce(seed):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1,5,size=5)
    e = [np.log(rng.dirichlet(np.ones(n))) for n in sizes]
    trans = [np.log(rng.uniform(0.01,1,size=(sizes[t],sizes[t+1]))) for t in range(4)]
    got = viterbi_decode(e,lambda i,j:trans[i])
    all_paths = itertools.product(*(range(len(v)) for v in e))
    choices = [(sum(e[t][p[t]] for t in range(5)) +
                sum(trans[t][p[t],p[t+1]] for t in range(4)), list(p)) for p in all_paths]
    best, path = max(choices,key=lambda a:a[0])
    assert len(got.segments)==1
    assert got.segments[0].states==path
    assert got.segments[0].log_score == pytest.approx(best)

def test_global_decoding_beats_local_choice():
    e = [np.log([.5,.5]),np.log([.9,.1]),np.log([.5,.5])]
    a = np.array([[NEG,NEG],[NEG,0.]])
    b = np.array([[NEG,NEG],[0.,NEG]])
    result = viterbi_decode(e,lambda i,j:[a,b][i])
    assert result.segments[0].states == [1,1,0]

def test_unreachable_state_not_selected():
    e=[np.log([.5,.5]),np.log([.2,.8])]
    t=np.array([[0.,NEG],[0.,NEG]])
    result=viterbi_decode(e,lambda i,j:t)
    assert result.segments[0].states[1]==0
    assert np.isfinite(result.segments[0].log_score)

def test_impossible_transitions_split():
    result=viterbi_decode([np.log([.5,.5]),np.log([.3,.7])],lambda i,j:np.full((2,2),NEG))
    assert [x.positions for x in result.segments] == [[0],[1]]
    assert result.breaks == [(0,1,'no_feasible_transition')]

def test_ties_deterministic():
    result=viterbi_decode([np.zeros(3)]*4,lambda i,j:np.zeros((3,3)))
    assert result.segments[0].states == [0]*4

@pytest.mark.parametrize('bad',[np.array([np.nan,0]),np.array([np.inf,0]),np.array([]),np.zeros((2,2))])
def test_invalid_emissions_rejected(bad):
    with pytest.raises(ValueError):
        viterbi_decode([np.zeros(2),bad],lambda i,j:np.zeros((2,max(bad.size,1))))

@pytest.mark.parametrize('bad',[np.array([[np.nan,0],[0,0]]),np.array([[np.inf,0],[0,0]]),np.zeros((3,2))])
def test_invalid_transition_rejected(bad):
    with pytest.raises(ValueError):
        viterbi_decode([np.zeros(2),np.zeros(2)],lambda i,j:bad)

def test_empty_and_single():
    assert viterbi_decode([],lambda i,j:None).segments == []
    assert viterbi_decode([np.log([.2,.8])],lambda i,j:None).segments[0].states == [1]

def test_all_negative_infinity_emission_must_raise():
    with pytest.raises(ValueError):
        viterbi_decode([np.array([-np.inf,-np.inf])],lambda i,j:None)
