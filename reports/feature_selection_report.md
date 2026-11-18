# Feature Selection Report

## Data and partition

- Partition: 80 supplied training trips
- Valid candidate rows: 1,312,143
- Valid observations represented: 185,141
- Correct candidates: 184,088
- Incorrect candidates: 1,128,055
- Missing feature values: 59,263
- Grouping: `trip_id`
- Cross-validation: five-fold `GroupKFold`
- Student-testing trips: reserved and not used for feature selection
- Evaluator-testing trips: not supplied

Invalid reference alignments and observations marked with duplicate reference arcs
were excluded using `label_valid` and `ref_has_duplicate`.

## Initial feature set

The 15 feature columns are documented in
`src/pathsnap/features/extract.py`: three spatial, four movement/temporal,
two candidate-ranking/direction, and six road-network features.

## Feature importance

Random Forest ranking:

1. candidate_rank
2. heading_difference_deg
3. distance_to_segment_m
4. segment_length_m
5. segment_bearing_deg
6. gps_heading_deg
7. gps_displacement_m
8. implied_speed_mps
9. to_node_degree
10. from_node_degree

LightGBM gain ranking:

1. candidate_rank
2. heading_difference_deg
3. distance_to_segment_m
4. segment_length_m
5. segment_bearing_deg
6. gps_displacement_m
7. from_node_degree
8. to_node_degree
9. from_node_in_degree
10. to_node_in_degree

Held-out permutation importance ranking by Macro F1:

1. candidate_rank
2. heading_difference_deg
3. distance_to_segment_m
4. from_node_out_degree
5. time_delta_s
6. implied_speed_mps
7. from_node_degree
8. to_node_out_degree
9. from_node_in_degree
10. gps_heading_deg

## Selection decision

A reproducible training-selected subset is saved in
`results/selected_features.json`. It contains the union of top training-fold
Random Forest, LightGBM gain, and held-out permutation rankings.

`candidate_rank` and `heading_difference_deg` dominate importance analyses.
Spatial distance and segment geometry are consistently important.
The road-network degree features have smaller but non-zero importance.

No final feature subset should be declared superior to all 15 until the
complete five-model GroupKFold ablation matrix is present. The existing
workspace contains analysis and importance outputs but no completed validation
matrix, because the long validation process did not finish and produced no
fold-result CSV.

## Limitations

- The 1.31 million candidate rows make a full five-model, eight-configuration
  matrix computationally expensive.
- LightGBM importances are impurity-based gain/split measures and may favor
  correlated or frequently split predictors.
- Permutation importance was calculated only on the first held-out GroupKFold
  fold in the selection stage.
- Duplicate road arcs make some reference labels geometrically ambiguous;
  these observations were excluded rather than relabelled.
- The reserved evaluator-test partition was not supplied and was not used.
