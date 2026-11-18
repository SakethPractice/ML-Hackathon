"""Write processed tables for every trip to data/processed.

For each trip in the chosen partitions this runs the configured pipeline
(configs/data.yaml): load -> road graph -> GPS context -> candidates, plus
reference labels for partitions that are allowed to have them, and writes
the tables described in ``pathsnap.data.processed``.

The evaluator partition is never exported with labels. It is only exported
at all when passed explicitly with --partitions.

Usage:
    python scripts/export_processed.py                 # train + student_test
    python scripts/export_processed.py --partitions train

Outputs:
    data/processed/<partition>/<trip_id>/*.csv.gz   (git-ignored, regenerate with this script)
    results/processed_manifest.csv                  (committed summary)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from pathsnap.candidates.pipeline import load_data_config, prepare_trip
from pathsnap.data.loader import list_trips
from pathsnap.data.processed import DEFAULT_PROCESSED_ROOT, save_processed

LABELLED_PARTITIONS = {"train", "student_test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    ap.add_argument("--out", default=str(DEFAULT_PROCESSED_ROOT))
    ap.add_argument("--manifest", default="results/processed_manifest.csv")
    args = ap.parse_args()
    cfg = load_data_config(args.config)

    rows = []
    t0 = time.time()
    for part in args.partitions:
        with_labels = part in LABELLED_PARTITIONS
        trips = list_trips(part, cfg.data_root)
        if not trips:
            print(f"{part}: no trips found under {cfg.data_root}")
        for trip_id in trips:
            pt = prepare_trip(part, trip_id, cfg, with_labels=with_labels)
            rows.append(save_processed(pt, args.out))
            print(f"{part:12s} {trip_id}  obs={rows[-1]['n_observations']:6d}  "
                  f"cands={rows[-1]['n_candidates']:7d}  {time.time() - t0:6.1f}s", flush=True)

    manifest = pd.DataFrame(rows)
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False)
    total = sum(f.stat().st_size for f in Path(args.out).rglob("*.csv.gz"))
    print(manifest.groupby("partition")[["n_observations", "n_candidates"]].sum())
    print(f"written {total / 1e6:.1f} MB to {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
