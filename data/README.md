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

## Processed data

`python scripts/export_processed.py` (about 4 minutes) writes, for every
train and student_test trip:

```
data/processed/<partition>/<trip_id>/
  gps_observations.csv.gz    per fix: observation_id, longitude, latitude, timestamp,
                             x_m, y_m, dt_prev_s, step_prev_m, speed_prev_mps,
                             heading_deg, window_disp_m, is_stationary, gap_before, segment_id
  candidate_segments.csv.gz  per (fix, arc): observation_id, arc_id, from_node_id, to_node_id,
                             distance_to_segment_m, segment_length_m, segment_bearing_deg,
                             distance_rank, t_along, closest_longitude, closest_latitude, radius_used_m
  labels.csv.gz              per fix: route_pos, ref_arc_id, align_dist_m, label_valid, ref_has_duplicate
  candidate_labels.csv.gz    per (fix, arc): is_reference, label_valid
```

Every table also has `trip_id`; use it as the group for cross-validation.
The first two tables contain no reference information and match the inputs
of `pathsnap.features.extract_features`. Labels live in separate files and
are only written for train and student_test.

```python
from pathsnap.data.processed import load_partition
from pathsnap.features import extract_features

t = load_partition("train", with_labels=True)
feats = extract_features(t["gps_observations"], t["candidate_segments"])
```

For node-degree features, pass `RoadGraph.from_trip(trip).to_networkx()` as
the graph, one trip at a time, because node IDs are local to each trip.

The files (about 58 MB, 360 files) are committed so teammates can use them
without re-running the pipeline. Re-running the script regenerates them.
Per-trip row counts are in `results/processed_manifest.csv`.

## Scripts

| Script | What it writes |
|---|---|
| `scripts/run_eda.py` | `results/dataset_inventory.csv`, `data_quality.csv`, `missing_values.csv`, `projection_check.csv`, `eda_partition_summary.csv`, `results/figures/eda_*.png`, `maps/html/eda_*.html` |
| `scripts/candidate_sweep.py` | `results/candidate_sweep*.csv` (train only) |
| `scripts/candidate_report.py` | `results/candidate_miss_rate*.csv` (train + student_test) |
| `scripts/export_processed.py` | `data/processed/...`, `results/processed_manifest.csv` |

Findings and decisions: `reports/data_quality_log.md`, `reports/decision_log.md`.
