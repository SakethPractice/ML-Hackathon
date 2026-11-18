"""Classifier-only (Pipeline A) vs classifier + HMM/Viterbi (Pipeline B) on identical trips.

Both pipelines use the same candidate sets, the same alignment table, the same denominators
and the same continuity test. Reference alignments / reference routes are used ONLY here.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..decoder.config import DecoderConfig
from ..decoder.hmm import decode_prepared, prepare_trip
from ..decoder.network import DirectedRoadNetwork
from ..decoder.types import normalise_trip_id
from .route_metrics import (candidate_level_metrics, collapse_repeats, count_matched_discontinuities,
                            derive_alignment_from_labels, direct_adjacency_breaks, observation_outcomes,
                            reference_route_metrics, summarise_outcomes)

logger = logging.getLogger(__name__)
A_NAME, B_NAME = "classifier_only", "classifier_hmm"
_SUM_COLS = ["n_observations", "n_aligned", "n_ambiguous_excluded", "n_unaligned_excluded",
             "n_candidate_miss", "n_unresolved", "n_correct", "n_consecutive_pairs",
             "n_discontinuities", "n_route_segments", "n_gaps", "n_inserted_arcs",
             "direct_adjacency_breaks", "runtime_seconds"]


@dataclass
class DecoderComparison:
    per_trip: pd.DataFrame
    summary: pd.DataFrame
    candidate_metrics: pd.DataFrame
    improved: pd.DataFrame
    worsened: pd.DataFrame
    output_files: list[str] = field(default_factory=list)


def _summarise(per_trip: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ok = per_trip[per_trip["status"] != "error"]
    for name in (A_NAME, B_NAME):
        d = ok[ok["pipeline"] == name]
        row: dict[str, Any] = {"pipeline": name, "n_trips": int(len(d))}
        for c in _SUM_COLS:
            row[c] = float(d[c].sum()) if c in d else float("nan")
        row["observation_top1_accuracy"] = row["n_correct"] / row["n_aligned"] if row["n_aligned"] else float("nan")
        valid_candidates = row["n_aligned"] - row["n_candidate_miss"]
        row["candidate_table_top1_accuracy"] = row["n_correct"] / valid_candidates if valid_candidates else float("nan")
        row["observation_top1_accuracy"] = row["observation_top1_accuracy"]  # backward compatibility
        row["candidate_miss_rate"] = row["n_candidate_miss"] / row["n_aligned"] if row["n_aligned"] else float("nan")
        row["continuity_rate"] = (1 - row["n_discontinuities"] / row["n_consecutive_pairs"]
                                  if row["n_consecutive_pairs"] else float("nan"))
        for c in ("route_length_f1", "route_lcs_similarity"):
            row[f"mean_{c}"] = float(d[c].mean()) if c in d and d[c].notna().any() else float("nan")
        row["mean_runtime_seconds_per_trip"] = row["runtime_seconds"] / row["n_trips"] if row["n_trips"] else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows)
    num = [c for c in summary.columns if c != "pipeline"]
    delta = {"pipeline": "delta_hmm_minus_classifier"}
    for c in num:
        delta[c] = summary.loc[1, c] - summary.loc[0, c]
    return pd.concat([summary, pd.DataFrame([delta])], ignore_index=True)


def _plots(per_trip: pd.DataFrame, summary: pd.DataFrame, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    s = summary[summary["pipeline"].isin([A_NAME, B_NAME])]
    metrics = ["observation_top1_accuracy", "continuity_rate", "mean_route_lcs_similarity", "mean_route_length_f1"]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.35
    x = np.arange(len(metrics))
    for k, name in enumerate((A_NAME, B_NAME)):
        vals = [float(s.loc[s["pipeline"] == name, m].iloc[0]) for m in metrics]
        ax.bar(x + (k - 0.5) * w, vals, w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Classifier-only vs classifier + HMM (aggregate)")
    fig.tight_layout()
    p = out / "comparison_metrics.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    files.append(str(p))

    ok = per_trip[per_trip["status"] != "error"]
    a = ok[ok["pipeline"] == A_NAME].set_index("trip_id")["observation_top1_accuracy"]
    b = ok[ok["pipeline"] == B_NAME].set_index("trip_id")["observation_top1_accuracy"]
    j = pd.concat([a.rename("A"), b.rename("B")], axis=1).dropna()
    if len(j):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(j["A"], j["B"])
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("Observation Top-1, classifier only")
        ax.set_ylabel("Observation Top-1, classifier + HMM")
        ax.set_title("Per-trip Top-1 (above line = HMM better)")
        fig.tight_layout()
        p = out / "per_trip_top1_scatter.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        files.append(str(p))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar([A_NAME, B_NAME], [float(s.loc[s["pipeline"] == n, "runtime_seconds"].iloc[0]) for n in (A_NAME, B_NAME)])
    ax.set_ylabel("total decoder runtime (s)")
    ax.set_title("Runtime (preparation included)")
    fig.tight_layout()
    p = out / "runtime_comparison.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    files.append(str(p))
    return files


def _unique_by_trip(mapping: Mapping[str, Any], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in mapping.items():
        tid = normalise_trip_id(key)
        if tid in result:
            raise ValueError(f"duplicate trip_id {tid} in {label} after normalisation")
        result[tid] = item
    return result


def evaluate_decoder(scored_candidates: pd.DataFrame, observations: pd.DataFrame,
                     networks: Mapping[str, DirectedRoadNetwork],
                     config: DecoderConfig | None = None, *,
                     reference_alignment: Optional[pd.DataFrame] = None,
                     reference_routes: Optional[Mapping[str, Sequence[int]]] = None,
                     trip_ids: Optional[Sequence[str]] = None,
                     output_dir: Optional[str | Path] = None,
                     write_debug_traces: bool = False) -> DecoderComparison:
    """Evaluate both pipelines. Only permitted trips (e.g. train CV folds / student-testing) should be passed.

    scored_candidates : trip_id, observation_idx, arc_id, match_probability, [label]
    observations      : trip_id, observation_idx, lon, lat, timestamp
    networks          : normalised trip id -> DirectedRoadNetwork
    reference_alignment: trip_id, observation_idx, ref_arc_id, alignment_status (optional;
                         otherwise derived from the ``label`` column)
    reference_routes  : trip id -> ordered reference arc ids (optional, evaluation only)
    """
    config = config or DecoderConfig()
    sc = scored_candidates.copy()
    sc["trip_id"] = sc["trip_id"].map(normalise_trip_id)
    obs_all = observations.copy()
    obs_all["trip_id"] = obs_all["trip_id"].map(normalise_trip_id)
    nets = _unique_by_trip(networks, "networks")
    refs = _unique_by_trip(reference_routes or {}, "reference_routes")
    if reference_alignment is not None:
        align = reference_alignment.copy()
        align["trip_id"] = align["trip_id"].map(normalise_trip_id)
    elif "label" in sc.columns:
        align = derive_alignment_from_labels(sc)
    else:
        raise ValueError("Provide reference_alignment or a 'label' column in scored_candidates")

    trips = [normalise_trip_id(t) for t in (trip_ids if trip_ids is not None else sorted(sc["trip_id"].unique()))]
    if len(set(trips)) != len(trips):
        raise ValueError("Duplicate trip IDs in trip_ids after normalisation")
    out = Path(output_dir) if output_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        if write_debug_traces:
            (out / "debug_traces").mkdir(exist_ok=True)

    rows, improved, worsened = [], [], []
    chosen_store: dict[str, tuple[dict, dict]] = {}
    for tid in trips:
        try:
            net = nets[tid]
            t0 = time.perf_counter()
            prepared = prepare_trip(tid, obs_all, sc, net, config)
            t_prep = time.perf_counter() - t0

            # ---- Pipeline A: highest-scoring candidate (ties -> lowest arc id) -------
            t1 = time.perf_counter()
            chosen_a = [int(np.argmax(s.match_scores)) for s in prepared.states]
            pred_a = {s.observation_idx: int(s.arc_ids[k]) for s, k in zip(prepared.states, chosen_a)}
            runtime_a = t_prep + (time.perf_counter() - t1)

            # ---- Pipeline B: Viterbi + reconstruction ---------------------------------
            res_b = decode_prepared(prepared, net, config)
            runtime_b = t_prep + res_b.runtime_seconds
            pred_b = {i: a for i, a in zip(res_b.observation_indices, res_b.matched_arc_ids) if a is not None}
            arc_to_k = [{int(a): k for k, a in enumerate(s.arc_ids)} for s in prepared.states]
            chosen_b = [arc_to_k[p][pred_b[s.observation_idx]] for p, s in enumerate(prepared.states)]

            chosen_store[tid] = (pred_a, pred_b)
            cand_arcs = {s.observation_idx: set(int(a) for a in s.arc_ids) for s in prepared.states}
            tal = align.loc[align["trip_id"] == tid]
            out_a = observation_outcomes(prepared.observation_indices, pred_a, cand_arcs, tal)
            out_b = observation_outcomes(prepared.observation_indices, pred_b, cand_arcs, tal)

            cont_a = count_matched_discontinuities(prepared.states, chosen_a, net, config)
            cont_b = count_matched_discontinuities(prepared.states, chosen_b, net, config)
            route_a = collapse_repeats([pred_a[s.observation_idx] for s in prepared.states])

            for name, outcomes, cont, runtime in ((A_NAME, out_a, cont_a, runtime_a),
                                                  (B_NAME, out_b, cont_b, runtime_b)):
                row = {"trip_id": tid, "pipeline": name, "status": "ok", **summarise_outcomes(outcomes),
                       "n_consecutive_pairs": cont["n_consecutive_pairs"],
                       "n_discontinuities": cont["n_discontinuities"],
                       "continuity_rate": cont["continuity_rate"], "runtime_seconds": runtime}
                if name == A_NAME:
                    row.update(n_route_segments=1 if route_a else 0, n_gaps=cont["n_discontinuities"],
                               n_inserted_arcs=0, direct_adjacency_breaks=direct_adjacency_breaks([route_a], net))
                    route = route_a
                else:
                    row.update(n_route_segments=len(res_b.route_segments), n_gaps=len(res_b.gap_boundaries),
                               n_inserted_arcs=res_b.diagnostics["n_inserted_intermediate_arcs"],
                               direct_adjacency_breaks=direct_adjacency_breaks(res_b.route_segments, net))
                    route = res_b.expanded_route_arc_ids
                if tid in refs:
                    row.update(reference_route_metrics(route, refs[tid], net.arc_length))
                rows.append(row)

            m = out_a.merge(out_b, on="observation_idx", suffixes=("_a", "_b"))
            m = m[m["in_denominator_a"]]
            cols = {"observation_idx": "observation_idx", "ref_arc_id_a": "ref_arc_id",
                    "pred_arc_id_a": "classifier_arc_id", "pred_arc_id_b": "hmm_arc_id"}
            sel = lambda df: df[list(cols)].rename(columns=cols).assign(trip_id=tid)
            improved.append(sel(m[~m["correct_a"] & m["correct_b"]]))
            worsened.append(sel(m[m["correct_a"] & ~m["correct_b"]]))

            if out and write_debug_traces:
                trace = {"decode_result": res_b.to_dict(), "classifier_only_arcs": pred_a,
                         "continuity_A": cont_a, "continuity_B": cont_b}
                (out / "debug_traces" / f"{tid}.json").write_text(json.dumps(trace, default=str), encoding="utf-8")
        except Exception as exc:  # keep evaluating the remaining trips
            logger.exception("Evaluation failed for trip %s", tid)
            for name in (A_NAME, B_NAME):
                rows.append({"trip_id": tid, "pipeline": name, "status": "error", "error": str(exc)})

    per_trip = pd.DataFrame(rows)
    summary = _summarise(per_trip)

    cand_rows = []
    if "label" in sc.columns:
        sub = sc.loc[sc["trip_id"].isin(list(chosen_store))]
        y = sub["label"].astype(int).to_numpy()
        p = pd.to_numeric(sub["match_probability"], errors="coerce").fillna(0.0).to_numpy()
        cand_rows.append({"scope": "classifier_threshold_0.5", **candidate_level_metrics(y, (p >= 0.5).astype(int), p)})
        for name, picker in ((A_NAME, 0), (B_NAME, 1)):
            sel_flags = []
            for tid, preds in chosen_store.items():
                t_sub = sub.loc[sub["trip_id"] == tid]
                chosen = preds[picker]
                flag = [int(chosen.get(int(o)) == int(a)) for o, a in zip(t_sub["observation_idx"], t_sub["arc_id"])]
                sel_flags.append((t_sub["label"].astype(int).to_numpy(), np.asarray(flag)))
            if sel_flags:
                yy = np.concatenate([a for a, _ in sel_flags])
                ff = np.concatenate([b for _, b in sel_flags])
                cand_rows.append({"scope": f"{name}_selected_candidate", **candidate_level_metrics(yy, ff)})
    cand_df = pd.DataFrame(cand_rows)

    imp = pd.concat(improved, ignore_index=True) if improved else pd.DataFrame()
    wor = pd.concat(worsened, ignore_index=True) if worsened else pd.DataFrame()
    files: list[str] = []
    if out:
        for name, df in (("decoder_comparison.csv", summary), ("per_trip_results.csv", per_trip),
                         ("candidate_level_metrics.csv", cand_df),
                         ("cases_viterbi_improved.csv", imp), ("cases_viterbi_worsened.csv", wor)):
            df.to_csv(out / name, index=False)
            files.append(str(out / name))
        (out / "decoder_config_used.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        if not per_trip.empty and (per_trip["status"] == "ok").any():
            files += _plots(per_trip, summary, out)
    return DecoderComparison(per_trip, summary, cand_df, imp, wor, files)
