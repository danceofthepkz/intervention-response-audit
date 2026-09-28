"""Post-collection reproducibility for quantities reported in the paper.

This module was added after confirmatory collection.  It performs no new data
collection and no re-selection of states; it only reproduces reported
quantities from the frozen states and already-collected v2 data.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from intervention_response_audit.v2.collect import read_attempt_log
from intervention_response_audit.v2.live import load_states
from intervention_response_audit.v2.schema import AttemptRecord, StateRecord, V2FrozenConfig, write_json
from intervention_response_audit.v2.secondary import bernoulli_kl, entropy
from intervention_response_audit.v2.structured import run_structured_crossfit


GlobalWeighting = Literal["row", "scenario_equal"]
TotalWeighting = Literal["state_side_mean", "call"]
IDENTITY_TOLERANCE = 64.0 * np.finfo(float).eps


class DegenerateMidpointError(ValueError):
    """Raised when the Taylor denominator is zero for at least one state."""


def generalized_js(probabilities: np.ndarray) -> float:
    """Generalized Jensen-Shannon divergence for Bernoulli probabilities."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("J requires a non-empty one-dimensional probability array")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Probabilities must lie in [0, 1]")
    return float(entropy(np.asarray([np.mean(values)]))[0] - np.mean(entropy(values)))


def j6_replicate_split(
    authority_calls: np.ndarray,
    hedge_calls: np.ndarray,
    state_ids: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Compute and verify the exact J_6 = between + within decomposition."""

    authority = np.asarray(authority_calls, dtype=float)
    hedge = np.asarray(hedge_calls, dtype=float)
    if authority.ndim != 2 or hedge.shape != authority.shape or authority.shape[1] != 3:
        raise ValueError("J_6 requires matching state-by-3 authority and hedge arrays")
    ids = list(state_ids or [str(index) for index in range(len(authority))])
    if len(ids) != len(authority):
        raise ValueError("state_ids length differs from the call arrays")

    rows = []
    for index, state_id in enumerate(ids):
        a = authority[index]
        h = hedge[index]
        pooled = np.concatenate([a, h])
        j6 = generalized_js(pooled)
        between = generalized_js(np.asarray([np.mean(a), np.mean(h)]))
        within = 0.5 * (generalized_js(a) + generalized_js(h))
        residual = j6 - between - within
        rows.append(
            {
                "state_id": state_id,
                "j6": j6,
                "between": between,
                "within": within,
                "identity_residual": residual,
            }
        )

    failures = [
        row for row in rows if abs(row["identity_residual"]) > IDENTITY_TOLERANCE
    ]
    if failures:
        details = ", ".join(
            f"{row['state_id']}={row['identity_residual']:.17g}" for row in failures
        )
        raise RuntimeError(f"J_6 decomposition failed per state: {details}")
    return {
        "means": {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("j6", "between", "within")
        },
        "maximum_absolute_identity_residual": float(
            max(abs(row["identity_residual"]) for row in rows)
        ),
        "identity_tolerance": IDENTITY_TOLERANCE,
        "per_state": rows,
    }


def second_order_approximation(
    authority_means: np.ndarray,
    hedge_means: np.ndarray,
    state_ids: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Compare exact per-state J_2 with its second-order approximation."""

    authority = np.asarray(authority_means, dtype=float)
    hedge = np.asarray(hedge_means, dtype=float)
    if authority.ndim != 1 or hedge.shape != authority.shape:
        raise ValueError("Authority and hedge means must be matching vectors")
    ids = list(state_ids or [str(index) for index in range(len(authority))])
    if len(ids) != len(authority):
        raise ValueError("state_ids length differs from the mean arrays")

    midpoint = (authority + hedge) / 2.0
    degenerate = [
        ids[index]
        for index in np.flatnonzero((midpoint <= 0.0) | (midpoint >= 1.0))
    ]
    if degenerate:
        raise DegenerateMidpointError(
            "Second-order approximation has m_j at 0 or 1 for states: "
            + ", ".join(degenerate)
        )

    delta = authority - hedge
    exact = entropy(midpoint) - 0.5 * (entropy(authority) + entropy(hedge))
    approx = delta**2 / (8.0 * midpoint * (1.0 - midpoint))
    discrepancy = approx - exact
    relative = np.full(len(exact), np.nan)
    nonzero = exact != 0.0
    relative[nonzero] = discrepancy[nonzero] / exact[nonzero]
    correlation = float(np.corrcoef(exact, approx)[0, 1])

    rows = [
        {
            "state_id": state_id,
            "authority_mean": float(authority[index]),
            "hedge_mean": float(hedge[index]),
            "midpoint": float(midpoint[index]),
            "delta": float(delta[index]),
            "exact": float(exact[index]),
            "approx": float(approx[index]),
            "discrepancy": float(discrepancy[index]),
            "absolute_discrepancy": float(abs(discrepancy[index])),
            "relative_discrepancy": (
                float(relative[index]) if np.isfinite(relative[index]) else None
            ),
            "absolute_relative_discrepancy": (
                float(abs(relative[index])) if np.isfinite(relative[index]) else None
            ),
        }
        for index, state_id in enumerate(ids)
    ]
    largest_absolute_index = int(np.argmax(np.abs(discrepancy)))
    finite_relative = np.flatnonzero(np.isfinite(relative))
    largest_relative_index = (
        int(finite_relative[np.argmax(np.abs(relative[finite_relative]))])
        if len(finite_relative)
        else None
    )
    aggregate_authority = float(np.mean(authority))
    aggregate_hedge = float(np.mean(hedge))
    aggregate_midpoint = (aggregate_authority + aggregate_hedge) / 2.0
    aggregate_delta = aggregate_authority - aggregate_hedge
    exact_mean = float(np.mean(exact))
    approx_mean = float(np.mean(approx))
    return {
        "mean_exact": exact_mean,
        "mean_approx": approx_mean,
        "relative_error_of_mean": approx_mean / exact_mean - 1.0,
        "pearson_r": correlation,
        "largest_absolute_discrepancy": rows[largest_absolute_index],
        "largest_relative_discrepancy": (
            rows[largest_relative_index] if largest_relative_index is not None else None
        ),
        "naive_aggregate_plugin": {
            "authority_mean": aggregate_authority,
            "hedge_mean": aggregate_hedge,
            "midpoint": aggregate_midpoint,
            "delta": aggregate_delta,
            "approx": aggregate_delta**2
            / (8.0 * aggregate_midpoint * (1.0 - aggregate_midpoint)),
        },
        "degenerate_midpoint_state_ids": [],
        "per_state": rows,
    }


def global_regret(
    predictions: pd.DataFrame,
    weighting: GlobalWeighting,
) -> tuple[float, list[dict]]:
    """Compute R_global with the requested, explicit natural-row weighting."""

    required = {"scenario_id", "oracle_probability", "structured_probability"}
    if not required <= set(predictions):
        raise ValueError(f"Structured predictions lack columns: {sorted(required - set(predictions))}")
    work = predictions.loc[:, sorted(required)].copy()
    work["kl"] = bernoulli_kl(
        work["oracle_probability"].to_numpy(dtype=float),
        work["structured_probability"].to_numpy(dtype=float),
    )
    per_scenario = (
        work.groupby("scenario_id", sort=True)["kl"]
        .agg([("row_count", "size"), ("mean_kl", "mean")])
        .reset_index()
    )
    if weighting == "row":
        estimate = float(work["kl"].mean())
    elif weighting == "scenario_equal":
        estimate = float(per_scenario["mean_kl"].mean())
    else:
        raise ValueError(f"Unknown R_global weighting: {weighting}")
    rows = [
        {
            "scenario_id": str(row.scenario_id),
            "row_count": int(row.row_count),
            "mean_kl": float(row.mean_kl),
        }
        for row in per_scenario.itertuples(index=False)
    ]
    return estimate, rows


def total_regret(
    authority_calls: np.ndarray,
    hedge_calls: np.ndarray,
    anchors: np.ndarray,
    weighting: TotalWeighting,
) -> tuple[float, np.ndarray]:
    """Compute R_total with explicit state-side-mean or individual-call weighting."""

    authority = np.asarray(authority_calls, dtype=float)
    hedge = np.asarray(hedge_calls, dtype=float)
    anchor = np.asarray(anchors, dtype=float)
    if authority.ndim != 2 or hedge.shape != authority.shape:
        raise ValueError("Authority and hedge calls must be matching matrices")
    if anchor.shape != (len(authority),):
        raise ValueError("Anchor vector length differs from the call matrices")
    if weighting == "state_side_mean":
        state_values = 0.5 * (
            bernoulli_kl(np.mean(authority, axis=1), anchor)
            + bernoulli_kl(np.mean(hedge, axis=1), anchor)
        )
    elif weighting == "call":
        all_calls = np.concatenate([authority, hedge], axis=1)
        state_values = np.mean(bernoulli_kl(all_calls, anchor[:, None]), axis=1)
    else:
        raise ValueError(f"Unknown R_total weighting: {weighting}")
    return float(np.mean(state_values)), state_values


def _confirmatory_call_arrays(
    attempts: tuple[AttemptRecord, ...],
    states: tuple[StateRecord, ...],
    selected_k: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    confirmatory = tuple(state for state in states if state.allocation == "confirmatory")
    if len(confirmatory) != 60 or len({state.scenario_id for state in confirmatory}) != 60:
        raise ValueError("Expected 60 scenario-unique confirmatory states")
    grouped: dict[tuple[str, str], list[AttemptRecord]] = defaultdict(list)
    for attempt in attempts:
        if attempt.accepted:
            if attempt.phase != "confirmatory" or attempt.parsed_output is None:
                raise ValueError(f"Invalid accepted attempt: {attempt.attempt_id}")
            grouped[(attempt.state_id, attempt.condition)].append(attempt)
    state_ids = sorted(state.state_id for state in confirmatory)
    side_arrays: dict[str, list[np.ndarray]] = {"authority": [], "hedge": []}
    for state_id in state_ids:
        for side in ("authority", "hedge"):
            cell = sorted(
                grouped[(state_id, side)], key=lambda attempt: attempt.replicate_index
            )
            if len(cell) != selected_k or [item.replicate_index for item in cell] != list(
                range(1, selected_k + 1)
            ):
                raise ValueError(f"Incomplete or non-canonical cell: {state_id}|{side}")
            side_arrays[side].append(
                np.asarray([item.parsed_output.reshare for item in cell], dtype=float)
            )
    return state_ids, np.stack(side_arrays["authority"]), np.stack(side_arrays["hedge"])


def build_report(
    attempts: tuple[AttemptRecord, ...],
    states: tuple[StateRecord, ...],
    normal_rows: pd.DataFrame,
    selected_k: int,
    config: V2FrozenConfig | None = None,
) -> dict:
    """Compute all reported quantities without mutating source data or selections."""

    if selected_k != 3:
        raise ValueError(f"The reported J_6 split requires selected_k=3, found {selected_k}")
    frozen = config or V2FrozenConfig()
    state_ids, authority, hedge = _confirmatory_call_arrays(
        attempts, states, selected_k
    )
    split = j6_replicate_split(authority, hedge, state_ids)
    approximation = second_order_approximation(
        np.mean(authority, axis=1), np.mean(hedge, axis=1), state_ids
    )

    structured = run_structured_crossfit(normal_rows, states, config=frozen)
    r_global_row, per_scenario = global_regret(structured.predictions, "row")
    r_global_scenario, _ = global_regret(structured.predictions, "scenario_equal")
    anchors = np.asarray(
        [structured.anchor_prediction_by_state[state_id] for state_id in state_ids]
    )
    r_total_means, total_by_mean = total_regret(
        authority, hedge, anchors, "state_side_mean"
    )
    r_total_calls, total_by_call = total_regret(authority, hedge, anchors, "call")
    total_per_state = [
        {
            "state_id": state_id,
            "anchor_probability": float(anchors[index]),
            "state_side_mean_regret": float(total_by_mean[index]),
            "call_weighted_regret": float(total_by_call[index]),
        }
        for index, state_id in enumerate(state_ids)
    ]
    return {
        "provenance": {
            "added_after_confirmatory_collection": True,
            "new_data_collection": False,
            "state_reselection": False,
            "selected_k": selected_k,
            "crossfit_seed": frozen.crossfit_seed,
        },
        "j6_replicate_split": split,
        "second_order_approximation": approximation,
        "aligned_regret": {
            "r_global": {
                "row": {
                    "estimate": r_global_row,
                    "weighting": "Each of 4,350 natural rollout rows has equal weight.",
                },
                "scenario_equal": {
                    "estimate": r_global_scenario,
                    "weighting": "Each of 80 natural-rollout scenarios has equal weight after within-scenario averaging.",
                },
                "per_scenario": per_scenario,
            },
            "r_total": {
                "state_side_mean": {
                    "estimate": r_total_means,
                    "weighting": "Each of 60 selected states has equal weight; K=3 calls are first averaged within side.",
                },
                "call": {
                    "estimate": r_total_calls,
                    "weighting": "Each of the 360 confirmatory calls has equal weight.",
                },
                "per_state": total_per_state,
            },
            "matched_comparisons": {
                "scenario_equal": {
                    "r_global": r_global_scenario,
                    "r_total": r_total_means,
                    "unit": "scenario/state, equal weight",
                },
                "observation_equal": {
                    "r_global": r_global_row,
                    "r_total": r_total_calls,
                    "unit": "natural row/confirmatory call, equal weight",
                },
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce post-collection quantities reported in the v2 paper"
    )
    parser.add_argument(
        "--normal-parquet",
        type=Path,
        default=Path("data/oracle_logs/phase1_full_v1.parquet"),
    )
    parser.add_argument(
        "--stage0-dir", type=Path, default=Path("prereg/v2/stage0")
    )
    parser.add_argument(
        "--attempts",
        type=Path,
        default=Path("results/v2/confirmatory/attempts.jsonl"),
    )
    parser.add_argument(
        "--pilot-decision",
        type=Path,
        default=Path("results/v2/pilot/pilot_decision.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/v2/reported_quantities/reported_quantities.json"),
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite reported quantities: {args.output}")
    selected_k = int(
        json.loads(args.pilot_decision.read_text(encoding="utf-8"))["decision"][
            "selected_k"
        ]
    )
    config = V2FrozenConfig.model_validate_json(
        (args.stage0_dir / "frozen_config.json").read_text(encoding="utf-8")
    )
    report = build_report(
        read_attempt_log(args.attempts),
        load_states(args.stage0_dir),
        pd.read_parquet(args.normal_parquet),
        selected_k,
        config,
    )
    write_json(args.output, report)
    print("V2_REPORTED_QUANTITIES: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
