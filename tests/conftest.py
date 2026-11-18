from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from pathsnap.decoder.network import DirectedRoadNetwork

@pytest.fixture()
def net():
    # Synthetic chain 0->1->2->3->4 with branch 1->5 and disconnected 6->7
    nodes = np.array([[77.59 + 0.001*k,12.97] for k in range(5)] +
                     [[77.591,12.971],[77.70,12.97],[77.701,12.97]])
    arcs = np.array([[0,1],[1,2],[2,3],[3,4],[1,5],[6,7],[2,1],[1,2]])
    return DirectedRoadNetwork(nodes,arcs)

@pytest.fixture()
def observations():
    return pd.DataFrame({'trip_id':['00000001']*4,'observation_idx':[0,1,2,3],
        'lon':[77.5905,77.5915,77.5925,77.5935],
        'lat':[12.97]*4,'timestamp':[0.,10.,20.,30.]})

@pytest.fixture()
def candidates():
    rows = []
    for i, (a,p) in enumerate([(0,.9),(1,.7),(2,.7),(3,.9)]):
        rows.append({'trip_id':'00000001','observation_idx':i,'arc_id':a,'match_probability':p})
        if i==2: rows.append({'trip_id':'00000001','observation_idx':i,'arc_id':4,'match_probability':.9})
    return pd.DataFrame(rows)
