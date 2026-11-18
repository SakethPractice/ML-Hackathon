"""Run real Random Forest predictions through PathSnap HMM/Viterbi.

Inference uses no reference labels. Trips with incomplete score
coverage are skipped and reported explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from pathsnap.candidates.pipeline import (
    load_data_config,
    prepare_trip,
)
from pathsnap.decoder.pipeline_adapter import decoder_inputs
from pathsnap.decoder.hmm import decode_trip
from pathsnap.decoder.frontend_payload import decoder_frontend_payload


def trip_code(value):
    """Normalise numeric trip IDs to eight-digit dataset IDs."""
    text = str(value).strip()
    if text.isdigit():
        return text.zfill(8)
    raise ValueError(f"Invalid trip ID: {value!r}")


def run(args):
    prediction_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)

    if not args.data_root:
        raise ValueError(
            "Set PATHSNAP_DATA_ROOT or pass --data-root."
        )
    # Use the exact candidate configuration from M1's export pipeline.
    cfg = load_data_config("configs/data.yaml")

    print("Using M1 candidate configuration: configs/data.yaml")

    print("Loading Random Forest predictions...", flush=True)
    predictions = pd.read_csv(prediction_path)

    required = {
        "trip_id",
        "observation_id",
        "arc_id",
        "match_probability",
    }
    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing)}"
        )

    # Reference labels, rankings and binary classifications
    # are deliberately NOT passed to the decoder.
    predictions = predictions[
        [
            "trip_id",
            "observation_id",
            "arc_id",
            "match_probability",
        ]
    ].copy()

    predictions["trip_id"] = (
        predictions["trip_id"].map(trip_code)
    )

    predictions["observation_id"] = pd.to_numeric(
        predictions["observation_id"], errors="raise"
    ).astype("int64")

    predictions["arc_id"] = pd.to_numeric(
        predictions["arc_id"], errors="raise"
    ).astype("int64")

    predictions["match_probability"] = pd.to_numeric(
        predictions["match_probability"], errors="raise"
    )

    if predictions["match_probability"].isna().any():
        raise ValueError("Predictions contain missing probabilities.")

    if not predictions["match_probability"].between(0, 1).all():
        raise ValueError("Probabilities must be between 0 and 1.")

    if predictions.duplicated(
        ["trip_id", "observation_id", "arc_id"]
    ).any():
        raise ValueError("Duplicate Random Forest candidate keys.")

    available_trips = sorted(predictions["trip_id"].unique())

    if args.trip_id:
        requested = trip_code(args.trip_id)
        if requested not in available_trips:
            raise ValueError(
                f"Trip {requested} not found in predictions."
            )
        available_trips = [requested]

    print(f"Trips selected: {len(available_trips)}", flush=True)
    print(f"Raw data: {args.data_root}", flush=True)
    print(f"Output: {output_dir.resolve()}", flush=True)

    summary = []

    for index, trip_id in enumerate(available_trips, start=1):
        start = time.perf_counter()

        print(
            f"\n[{index}/{len(available_trips)}] "
            f"Trip {trip_id}",
            flush=True,
        )

        trip_scores = predictions.loc[
            predictions["trip_id"].eq(trip_id),
            ["observation_id", "arc_id", "match_probability"],
        ].rename(
            columns={"observation_id": "obs_id"}
        ).copy()

        row = {
            "trip_id": trip_id,
            "prediction_rows": len(trip_scores),
            "candidate_rows": None,
            "missing_predictions": None,
            "extra_predictions": None,
            "decode_status": None,
            "observations": None,
            "matched_observations": None,
            "route_segments": None,
            "runtime_seconds": None,
            "message": "",
        }

        try:
            # No reference route and no labels are loaded.
            prepared = prepare_trip(
        partition="student_test",
        trip_id=trip_id,
        cfg=cfg,
        with_labels=False,
        data_root=args.data_root,
        )

            candidate_keys = prepared.candidates.table[
                ["obs_id", "arc_id"]
            ].copy()

            row["candidate_rows"] = len(candidate_keys)

            # Compare the exact candidate-key sets before decoding.
            coverage = candidate_keys.merge(
                trip_scores[["obs_id", "arc_id"]],
                on=["obs_id", "arc_id"],
                how="outer",
                validate="one_to_one",
                indicator=True,
            )

            missing_count = int(
                coverage["_merge"].eq("left_only").sum()
            )
            extra_count = int(
                coverage["_merge"].eq("right_only").sum()
            )

            row["missing_predictions"] = missing_count
            row["extra_predictions"] = extra_count

            if missing_count or extra_count:
                row["decode_status"] = "skipped_incomplete_scores"
                row["message"] = (
                    f"Missing {missing_count}, extra {extra_count}"
                )
                print(
                    f"SKIPPED: {row['message']}",
                    flush=True,
                )
                continue

            print(
                f"Candidate coverage: {len(candidate_keys)}/"
                f"{len(candidate_keys)}",
                flush=True,
            )

            observations, candidates, network = decoder_inputs(
                prepared,
                trip_scores,
            )

            print("Running HMM/Viterbi...", flush=True)

            result = decode_trip(
                trip_id,
                observations,
                candidates,
                network,
            )

            payload = decoder_frontend_payload(result)
            geojson = payload["route_geojson"]

            # Save the full JSON contract for M4.
            json_path = output_dir / f"trip_{trip_id}.json"
            geojson_path = output_dir / f"trip_{trip_id}.geojson"

            json_path.write_text(
                json.dumps(
                    payload,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

            geojson_path.write_text(
                json.dumps(
                    geojson,
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

            matched = sum(
                pd.notna(arc)
                for arc in result.matched_arc_ids
            )

            row["decode_status"] = result.status
            row["observations"] = len(
                result.observation_indices
            )
            row["matched_observations"] = matched
            row["route_segments"] = len(
                result.route_segments
            )

            print(f"Decoder status: {result.status}")
            print(
                f"Matched observations: "
                f"{matched}/{row['observations']}"
            )
            print(
                f"GeoJSON features: "
                f"{len(geojson['features'])}"
            )
            print(f"Saved: {json_path}")
            print(f"Saved: {geojson_path}")

        except Exception as exc:
            row["decode_status"] = "integration_error"
            row["message"] = f"{type(exc).__name__}: {exc}"
            print(f"ERROR: {row['message']}", flush=True)

        finally:
            row["runtime_seconds"] = round(
                time.perf_counter() - start, 3
            )
            summary.append(row)

            pd.DataFrame(summary).to_csv(
                output_dir / "integration_summary.csv",
                index=False,
            )

    result_table = pd.DataFrame(summary)

    print("\n=== INTEGRATION SUMMARY ===")
    print(
        result_table[
            [
                "trip_id",
                "decode_status",
                "observations",
                "matched_observations",
                "route_segments",
                "missing_predictions",
            ]
        ].to_string(index=False)
    )

    print(
        "\nSaved summary:",
        output_dir / "integration_summary.csv",
    )

    errors = result_table["decode_status"].eq(
        "integration_error"
    ).sum()

    if errors:
        raise SystemExit(
            f"{errors} trip(s) encountered integration errors."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random Forest to HMM/Viterbi to GeoJSON"
    )
    parser.add_argument(
        "--predictions",
        default=(
            "results/predictions/"
            "student_test_random_forest.csv.gz"
        ),
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("PATHSNAP_DATA_ROOT"),
    )
    parser.add_argument(
        "--output-dir",
        default="results/hmm_random_forest",
    )
    parser.add_argument(
        "--trip-id",
        default=None,
        help="Optional single trip, e.g. 00000000",
    )

    run(parser.parse_args())
