# Data-quality log, assumptions and decisions (Member 1)

All numbers below come from `scripts/run_eda.py`, `scripts/candidate_sweep.py`
and `scripts/candidate_report.py`; the tables they write are in `results/`.
Partition used is stated for every item.

## Partitions supplied

| Partition | Official count | Supplied | Used here for |
|---|---:|---:|---|
| train | 80 | 80 | everything below, parameter choices, label analysis |
| student_test | 10 | 10 | structure checks, development-stage miss rate only |
| evaluator_test | 10 | 0 | not supplied; `run_eda.py` does not read this folder |

Trip IDs are 8-digit folder names; train and student_test IDs do not overlap.

## File format checks (train + student_test)

Source: `results/dataset_inventory.csv`, `results/missing_values.csv`.

- Every trip folder has all four files (`.track`, `.nodes`, `.arcs`, `.route`).
- No blank lines, no lines with the wrong number of columns, no missing values,
  no coordinates outside valid lon/lat ranges.
- Column order confirmed as longitude, latitude (e.g. trip 00000001 lies at
  ~8.6 E, 49.8-50.2 N; plotting with this order puts the GPS on the roads).
- `.track` files use CRLF line endings; the other files use LF. The reader
  handles both.
- All arc endpoints reference existing nodes; all route arc IDs exist.

## GPS tracks

Source: `results/data_quality.csv`, `results/figures/eda_sampling_intervals.png`.

- train: 208,417 fixes (361-19,313 per trip, median 1,529);
  student_test: 19,160 fixes (262-3,958, median 1,420).
- Timestamps are strictly increasing in every trip: no duplicates, no
  reordering needed. Median step is 1 s in every trip.
- Gaps: 28 of 80 train trips and 2 of 10 student_test trips have at least one
  gap over 30 s. Extreme cases: train 00000090 (153,064 s, about 42 h) and
  train 00000078 (52,810 s). These are treated as segment boundaries
  (`segment_id` in the GPS context), not as continuous driving.
- Some timestamps are non-integer (e.g. steps of 1.015 s in train 00000002).
- Repeated positions exist (up to 402 in one train trip); they are kept
  because timestamps differ.

## Road networks

Source: `results/data_quality.csv`, `results/figures/eda_network_size_connectivity.png`.

- Each trip has its own network; node and arc IDs are local to that trip.
  Networks range from 299 to 155,003 nodes (train).
- Networks are directed. On average 92% of arcs have a reverse arc (two-way
  streets stored as two arcs); lowest is 35% in one train trip.
- No isolated nodes. Most networks have several weakly connected
  components; the largest one usually covers more than 96% of nodes. Lowest:
  train 00000095 (69%).
- Anomalies kept as-is and reported: duplicate arcs with identical from/to
  nodes (1,775 train, 128 student_test), self-loops (22 / 2), zero-length
  arcs (100 / 6). Duplicate arcs are geometrically identical, so no GPS-based
  method can tell them apart; `ref_has_duplicate` flags them (0.014% of
  train reference labels).
- Arc lengths: median ~25 m; longest single arc up to 9.3 km.
  Because of long arcs, nearby-road lookup samples along each arc rather
  than using endpoints.

## Distances and projection

Source: `results/projection_check.csv`.

- Lengths use haversine on a sphere of radius 6,371,008.8 m.
- Point-to-segment geometry uses an equirectangular projection centred on
  each trip's network. Network extents are at most ~78 km. Projected arc
  lengths differ from haversine by a median of 0.01% and at most 0.7%
  across all 90 trips, which is acceptable at GPS noise scales.

## Reference routes

Source: `results/data_quality.csv`.

- Consecutive route arcs share a node almost always. Exceptions: 10 train
  trips (up to 3 breaks) and 1 student_test trip (00000076, 1 break). These
  are not repaired.
- 9 train trips and 1 student_test trip pass the same arc more than once
  (loops). Alignment works on route *positions*, so loops are fine.
- The route has no timestamps and no link to GPS rows. Per-fix reference
  arcs come from a monotone DP alignment (`src/pathsnap/data/route_alignment.py`).
- After alignment (train), on average 90% of route positions get at least
  one fix; the rest are mostly short arcs driven between two 1 Hz fixes.

## Fixes far from the reference route (train)

Source: `results/figures/eda_train_alignment_distance.png`, `maps/html/eda_train_00000054.html`.

- Distance from a fix to its aligned arc is bimodal: most fixes are within
  a few metres (median 5.7 m); a second group sits at 100-220 m.
- Inspected on plots: the far group is mostly parked vehicles. The GPS
  dwells off the road network (car parks, buildings) at the start or end of
  the trip, and the route does not go there. Some cases are stretches where
  the GPS continues where the route does not (end of train 00000002).
- Single-step speed does not detect parking. GPS jitter while parked gives
  ~7 m per step. A windowed displacement (±15 fixes < 30 m) is used instead:
  9.1% of train and 6.7% of student_test fixes are stationary by that rule.
- **Decision:** a fix gets a valid reference label only if it is within
  50 m of its aligned arc. This excludes 23,255 of 208,417 train fixes
  (11.2%). Two trips account for most of them: 00000054 (89% of its fixes)
  and 00000033 (59%). Excluded fixes stay in the data with `label_valid = False`.
  They are not relabelled.

## Candidate generation

Source: `results/candidate_sweep.csv` (train only), `results/candidate_miss_rate*.csv`.

- Swept radius 15-200 m and top-k 2-24 on train.
- **Caveat:** at radius 50 m and above, with no top-k limit, the miss rate is
  0 by construction, because valid labels are by definition within 50 m. The
  radius is therefore tied to the label threshold, and top-k is the real
  trade-off.
- **Decision:** radius 50 m, top-k 12, fallback doubling the radius up to
  400 m when nothing is found.
  - Train: 0.57% pooled miss rate over valid labels, 6.85 candidates per fix
    on average, no fix left without candidates; 18,229 fixes needed the
    fallback.
  - Student_test (development check, not used for tuning): 0.07% miss rate,
    no fallback needed.
- The closest candidate is the reference arc for only ~56% of valid fixes.
  The main reason is that both directions of a two-way road tie on
  distance, so heading/direction features are essential downstream.
- Known failure: dense clusters of 1-5 m arcs (junction modelling, e.g.
  train 00000072, 15.6% miss at top-8) crowd the top-k list.

## Leakage controls in this part of the code

- `load_trip(..., with_route=False)` and `prepare_trip(..., with_labels=False)`
  never open the `.route` file. A unit test enforces this.
- Candidate generation and GPS context use only the track and the network.
- Parameters (radius, top-k, label threshold, stationary rule) were chosen
  from train data only.
