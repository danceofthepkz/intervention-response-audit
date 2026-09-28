"""Exact Stage-0 secondary formulas, implemented after primary collection.

The estimands and seeds are frozen in the Stage-0 protocol.  This module must
be reported as a post-primary implementation of pre-specified analyses.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from intervention_response_audit.audit import EPS
from intervention_response_audit.v2.schema import AttemptRecord, StateRecord, V2FrozenConfig, write_json
from intervention_response_audit.v2.structured import StructuredCrossfitResult, run_structured_crossfit


def entropy(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def bernoulli_kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    left = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    right = np.clip(np.asarray(q, dtype=float), EPS, 1.0 - EPS)
    return left * np.log(left / right) + (1.0 - left) * np.log(
        (1.0 - left) / (1.0 - right)
    )


def text_blind_regret(authority: np.ndarray, hedge: np.ndarray) -> np.ndarray:
    mean = (np.asarray(authority) + np.asarray(hedge)) / 2.0
    return entropy(mean) - (entropy(authority) + entropy(hedge)) / 2.0


def anchor_decomposition(
    authority: np.ndarray,
    hedge: np.ndarray,
    anchor: np.ndarray,
) -> dict[str, float]:
    mean = (np.asarray(authority) + np.asarray(hedge)) / 2.0
    r_text = float(np.mean(text_blind_regret(authority, hedge)))
    r_struct = float(np.mean(bernoulli_kl(mean, anchor)))
    r_total = float(
        np.mean(
            (bernoulli_kl(authority, anchor) + bernoulli_kl(hedge, anchor)) / 2.0
        )
    )
    return {
        "r_text_raw": r_text,
        "r_struct_cf_anchor": r_struct,
        "r_total_cf_anchor": r_total,
        "decomposition_error": r_total - r_text - r_struct,
    }


def _interval(draws: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def analyze_secondary(
    attempts: Iterable[AttemptRecord],
    states: Iterable[StateRecord],
    selected_k: int,
    structured: StructuredCrossfitResult,
    config: V2FrozenConfig | None = None,
    *,
    _bootstrap_resamples: int = 20_000,
    _template_resamples: int = 20_000,
    _permutation_resamples: int = 100_000,
) -> dict:
    frozen = config or V2FrozenConfig()
    state_rows = tuple(state for state in states if state.allocation == "confirmatory")
    if len(state_rows) != 60:
        raise ValueError("Secondary analysis requires 60 confirmatory states")
    state_by_id = {state.state_id: state for state in state_rows}
    accepted = tuple(item for item in attempts if item.accepted)
    if len(accepted) != 60 * 2 * selected_k:
        raise ValueError("Secondary analysis call count differs from frozen K")
    grouped: dict[tuple[str, str], list[AttemptRecord]] = defaultdict(list)
    for item in accepted:
        if item.phase != "confirmatory" or item.parsed_output is None:
            raise ValueError("Secondary analysis received an invalid attempt")
        grouped[(item.state_id, item.condition)].append(item)

    reshare_means: dict[str, tuple[float, float]] = {}
    belief_contrasts: dict[str, float] = {}
    raw_reshare: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for state_id in sorted(state_by_id):
        cell_arrays = {}
        for condition in ("authority", "hedge"):
            cell = sorted(
                grouped[(state_id, condition)], key=lambda item: item.replicate_index
            )
            if len(cell) != selected_k:
                raise ValueError(f"Incomplete secondary cell {state_id}|{condition}")
            cell_arrays[condition] = {
                "reshare": np.asarray([item.parsed_output.reshare for item in cell]),
                "believe": np.asarray([item.parsed_output.believe for item in cell]),
            }
        a_r = cell_arrays["authority"]["reshare"]
        h_r = cell_arrays["hedge"]["reshare"]
        raw_reshare[state_id] = (a_r, h_r)
        reshare_means[state_id] = (float(np.mean(a_r)), float(np.mean(h_r)))
        belief_contrasts[state_id] = float(
            np.mean(cell_arrays["authority"]["believe"])
            - np.mean(cell_arrays["hedge"]["believe"])
        )

    cell_belief: dict[str, np.ndarray] = {}
    for cell in sorted({state.template_cell for state in state_rows}):
        ids = [state.state_id for state in state_rows if state.template_cell == cell]
        cell_belief[cell] = np.asarray([belief_contrasts[state_id] for state_id in ids])
    belief_rng = np.random.default_rng(frozen.primary_bootstrap_seed)
    belief_draws = np.empty(_bootstrap_resamples)
    for draw in range(_bootstrap_resamples):
        belief_draws[draw] = float(
            np.mean(
                np.concatenate(
                    [belief_rng.choice(values, len(values), replace=True)
                     for values in cell_belief.values()]
                )
            )
        )

    cluster_means = {cell: float(np.mean(values)) for cell, values in cell_belief.items()}
    cluster_counts = {cell: len(values) for cell, values in cell_belief.items()}
    template_rng = np.random.default_rng(frozen.template_bootstrap_seed)
    template_draws = np.empty(_template_resamples)
    topics = sorted({state.topic for state in state_rows})
    for draw in range(_template_resamples):
        topic_means = []
        for topic in topics:
            cells = [f"{topic}|{tier}" for tier in ("low", "mid", "high")]
            weights = np.asarray([cluster_counts[cell] for cell in cells], dtype=float)
            weights /= weights.sum()
            sampled = template_rng.choice(cells, 3, replace=True, p=weights)
            topic_means.append(float(np.mean([cluster_means[str(cell)] for cell in sampled])))
        template_draws[draw] = float(np.mean(topic_means))

    ordered_ids = sorted(state_by_id)
    authority = np.asarray([reshare_means[state_id][0] for state_id in ordered_ids])
    hedge = np.asarray([reshare_means[state_id][1] for state_id in ordered_ids])
    raw_text = float(np.mean(text_blind_regret(authority, hedge)))
    if selected_k == 1:
        permutation_baseline = 0.0
        actual_permutations = 0
    else:
        permutation_rng = np.random.default_rng(frozen.permutation_seed)
        total = 0.0
        for _ in range(_permutation_resamples):
            perm_a = []
            perm_h = []
            for state_id in ordered_ids:
                a_values, h_values = raw_reshare[state_id]
                permuted = permutation_rng.permutation(np.concatenate([a_values, h_values]))
                perm_a.append(float(np.mean(permuted[:selected_k])))
                perm_h.append(float(np.mean(permuted[selected_k:])))
            total += float(np.mean(text_blind_regret(np.asarray(perm_a), np.asarray(perm_h))))
        permutation_baseline = total / _permutation_resamples
        actual_permutations = _permutation_resamples

    heterogeneity = {}
    for attribute in ("topic", "credibility_tier", "exposure_bin", "step_bin", "template_cell"):
        groups: dict[str, list[float]] = defaultdict(list)
        for state_id in ordered_ids:
            state = state_by_id[state_id]
            groups[str(getattr(state, attribute))].append(
                reshare_means[state_id][0] - reshare_means[state_id][1]
            )
        heterogeneity[attribute] = {
            label: {"n": len(values), "mean_signed_contrast": float(np.mean(values))}
            for label, values in sorted(groups.items())
        }

    anchor = np.asarray(
        [structured.anchor_prediction_by_state[state_id] for state_id in ordered_ids]
    )
    decomposition = anchor_decomposition(authority, hedge, anchor)
    if abs(decomposition["decomposition_error"]) > 1e-10:
        raise RuntimeError("Counterfactual-anchor decomposition identity failed")
    return {
        "belief": {
            "estimate": float(np.mean(list(belief_contrasts.values()))),
            "fixed_template_ci": _interval(belief_draws),
            "template_cluster_ci": _interval(template_draws),
        },
        "reshare": {
            "raw_absolute_contrast": float(np.mean(np.abs(authority - hedge))),
            "r_text_raw": raw_text,
            "r_text_permutation_null": permutation_baseline,
            "r_text_adjusted": raw_text - permutation_baseline,
            "permutation_resamples": actual_permutations,
        },
        "structured": {
            "r_global_full": structured.r_global_full,
            **decomposition,
            "feature_count": structured.feature_count,
            "selected_c_by_outer_fold": structured.selected_c_by_outer_fold,
        },
        "heterogeneity": heterogeneity,
        "provenance_note": "Pre-specified Stage-0 formulas; execution code implemented after primary collection and cannot alter the primary verdict.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pre-specified v2 secondary analyses")
    parser.add_argument("--normal-parquet", type=Path, default=Path("data/oracle_logs/phase1_full_v1.parquet"))
    parser.add_argument("--states", type=Path, default=Path("prereg/v2/stage0/state_allocation.jsonl"))
    parser.add_argument("--attempts", type=Path, default=Path("results/v2/confirmatory/attempts.jsonl"))
    parser.add_argument("--pilot-decision", type=Path, default=Path("results/v2/pilot/pilot_decision.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/v2/secondary"))
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite secondary results: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    states = tuple(
        StateRecord.model_validate_json(line)
        for line in args.states.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    attempts = tuple(
        AttemptRecord.model_validate_json(line)
        for line in args.attempts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    import json

    selected_k = int(json.loads(args.pilot_decision.read_text())["decision"]["selected_k"])
    structured = run_structured_crossfit(pd.read_parquet(args.normal_parquet), states)
    result = analyze_secondary(attempts, states, selected_k, structured)
    structured.predictions.to_parquet(args.output_dir / "structured_oof_predictions.parquet", index=False)
    structured.predictions.to_csv(args.output_dir / "structured_oof_predictions.csv", index=False)
    write_json(args.output_dir / "secondary_result.json", result)
    write_json(
        args.output_dir / "structured_anchor_predictions.json",
        structured.anchor_prediction_by_state,
    )
    print("E8_SECONDARY_ANALYSES: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
