# Data

Raw data is not committed. Place the supplied partitions like this:

```
data/raw/
  train/<trip_id>/<trip_id>.{track,nodes,arcs,route}          80 trips
  student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}   10 trips
  evaluator_test/<trip_id>/...                                 only if supplied
```

The root can be changed with `PATHSNAP_DATA_ROOT` or `data_root` in
`configs/data.yaml`.

| File | Columns (whitespace separated, no header) | ID |
|---|---|---|
| `.track` | longitude, latitude, timestamp (s) | row index = `obs_id` |
| `.nodes` | longitude, latitude | row index = `node_id` |
| `.arcs` | from_node_id, to_node_id (directed) | row index = `arc_id` |
| `.route` | arc_id, in travel order | row index = route position |

Node and arc IDs are local to each trip.

## Using the data in code

```python
from pathsnap.candidates.pipeline import load_data_config, prepare_trip

cfg = load_data_config()                                   # configs/data.yaml
pt = prepare_trip("train", "00000001", cfg, with_labels=True)

pt.graph            # RoadGraph: arc_from, arc_to, arc_length_m, arc_bearing_deg, successors(), follows()
pt.context          # per obs_id: x, y, dt_prev_s, speed_prev_mps, heading_deg, is_stationary, segment_id, ...
pt.candidates.table # per (obs_id, arc_id): rank, distance_m, t_along, proj_x, proj_y, radius_used_m
pt.labels           # per obs_id: route_pos, ref_arc_id, align_dist_m, label_valid, ref_has_duplicate
```

For inference, use `with_labels=False` (the default). The route file is
then never opened.

`data/processed/` is ignored by git. Regenerate anything there from the
scripts.

## Scripts

| Script | What it writes |
|---|---|
| `scripts/run_eda.py` | `results/dataset_inventory.csv`, `data_quality.csv`, `missing_values.csv`, `projection_check.csv`, `eda_partition_summary.csv`, `results/figures/eda_*.png`, `maps/html/eda_*.html` |
| `scripts/candidate_sweep.py` | `results/candidate_sweep*.csv` (train only) |
| `scripts/candidate_report.py` | `results/candidate_miss_rate*.csv` (train + student_test) |

Findings and decisions: `reports/data_quality_log.md`, `reports/decision_log.md`.
