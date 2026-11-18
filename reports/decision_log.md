# Decision log

## D1 - Distance method and projection
- Date: 2026-11-18
- Owner: Member 1 (Vrinda)
- Problem observed: trips are spread across several countries and
  latitudes; raw coordinates are degrees.
- Options considered: Euclidean distance on degrees (rejected: not metres);
  UTM via pyproj (extra dependency); haversine plus a local tangent-plane
  projection.
- Chosen approach: haversine for lengths; per-trip local equirectangular
  projection for point-to-segment geometry.
- Evidence: projected vs haversine arc length error median 0.01%, max 0.7%
  over 90 trips (`results/projection_check.csv`).
- Limitations: accuracy degrades for networks much larger than ~100 km
  (none in the data).

## D2 - Per-fix reference labels by monotone alignment
- Date: 2026-11-18
- Owner: Member 1
- Problem observed: `.route` is an ordered arc list without timestamps; GPS
  rows and route rows do not correspond.
- Options considered: nearest route arc per fix (breaks on loops and where
  the route passes the same place twice); monotone DP alignment.
- Chosen approach: DP alignment minimising the summed point-to-segment
  distance, with non-decreasing route position.
- Evidence: unit test with a route driven twice around a loop; on train,
  median fix-to-aligned-arc distance is 5.7 m.

## D3 - Exclude labels further than 50 m from the aligned arc
- Date: 2026-11-18
- Owner: Member 1
- Problem observed: bimodal alignment distance; the far mode is parked or
  off-route GPS (see `reports/data_quality_log.md`).
- Chosen approach: `label_valid = align_dist_m <= 50`. Train exclusions:
  11.2% of fixes, 89% of those from trips 00000054 and 00000033.
- Limitations: a genuine on-road fix with more than 50 m GPS error loses its
  label. It still gets candidates and predictions, but it is not scored.

## D4 - Candidate radius 50 m, top-k 12, fallback to 400 m
- Date: 2026-11-18
- Owner: Member 1
- Evidence: `results/candidate_sweep.csv` (train only): top-8 gives 0.85%
  miss, top-12 0.57%, top-16 0.46%. Candidates per fix: 5.2 / 6.5 / 7.3.
- Chosen approach: top-12, as a balance between miss rate and the number
  of HMM states.
- Limitations: radius is not independent of the label threshold (see
  data-quality log).
- To revisit: Member 3 may want a larger top-k if the decoder benefits
  from it.

## D5 - Stationary detection by windowed displacement
- Date: 2026-11-18
- Owner: Member 1
- Problem observed: parked vehicles show ~7 m per 1 s step of jitter, so
  speed thresholds fail.
- Chosen approach: `is_stationary` if displacement between fix i-15 and
  i+15 is under 30 m.
- Note: uses future fixes inside the same trip. That is available at
  inference because the API processes a whole trip at once.

## D6 - Processed tables exported per trip
- Date: 2026-11-18
- Owner: Member 1
- Problem observed: processing only existed in memory (`prepare_trip`).
  Feature extraction expects ready-made GPS and candidate tables with its
  own column names.
- Chosen approach: `scripts/export_processed.py` writes per-trip gzip CSVs
  to `data/processed/` in the `extract_features` input format. Labels go in
  separate files, written for train and student_test only.
- Evidence: a unit test runs `extract_features` on exported tables. On
  train 00000095 all 15 feature columns are produced. Totals: 208,417 fixes
  and 1,428,465 candidate rows (train); 19,160 and 147,060 (student_test).
- Storage: first kept out of git (about 58 MB), then committed on
  2026-11-18 so the team can use them directly. Each file is under 5 MB.
  The script regenerates them in about 4 minutes.
