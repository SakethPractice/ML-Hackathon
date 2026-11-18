"""Fit the selected candidate classifier on all training trips and save it.

The API loads this file for inference. Model choice comes from
results/model_selection.json (written by scripts/run_all_models.py); the
pipeline definition comes from pathsnap.models.train. Only the train
partition is used.

    python scripts/train_final_model.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.features import get_feature_columns  # noqa: E402
from pathsnap.models.train import make_model_definitions  # noqa: E402
from scripts.run_feature_analysis import build_labeled_data  # noqa: E402

SELECTION = ROOT / "results" / "model_selection.json"
ARTIFACTS = ROOT / "artifacts" / "models"


def main() -> None:
    selected = json.loads(SELECTION.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=selected["selected_model"])
    parser.add_argument("--seed", type=int, default=selected.get("random_seed", 42))
    args = parser.parse_args()

    print("Building training features...", flush=True)
    _, X, y = build_labeled_data("train")
    X = X[get_feature_columns()]

    model = make_model_definitions(random_seed=args.seed)[args.model]
    started = perf_counter()
    model.fit(X, y)
    seconds = perf_counter() - started

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS / f"{args.model}.joblib"
    joblib.dump(model, out)
    print(f"Fitted {args.model} on {len(X)} training rows in {seconds:.1f}s -> {out}")


if __name__ == "__main__":
    main()
