"""Frozen statistical decisions for pilot and confirmatory v2 data."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from intervention_response_audit.v2.schema import (
    AttemptRecord,
    ConfirmatoryResult,
    PilotDecision,
    StateRecord,
    V2FrozenConfig,
)


def analyze_pilot_k(
    attempts: Iterable[AttemptRecord],
    config: V2FrozenConfig | None = None,
) -> PilotDecision:
    """Apply the preregistered deterministic/K3/K5/K10/HALT decision rule."""

    frozen = config or V2FrozenConfig()
    accepted = tuple(attempt for attempt in attempts if attempt.accepted)
    if len(accepted) != 200:
        raise ValueError(f"Pilot K analysis requires exactly 200 accepted slots, found {len(accepted)}")
    if any(attempt.phase != "pilot" for attempt in accepted):
        raise ValueError("Pilot K analysis received a non-pilot attempt")
    slot_ids = [attempt.slot_id for attempt in accepted]
    response_ids = [attempt.response_id for attempt in accepted]
    usage_ids = [attempt.usage_event_id for attempt in accepted]
    if len(set(slot_ids)) != 200:
        raise ValueError("Pilot accepted slots are not unique")
    if None in response_ids or len(set(response_ids)) != 200:
        raise ValueError("Pilot response IDs are missing or reused")
    if None in usage_ids or len(set(usage_ids)) != 200:
        raise ValueError("Pilot usage-event IDs are missing or reused")

    grouped: dict[tuple[str, str], list[AttemptRecord]] = defaultdict(list)
    for attempt in accepted:
        if attempt.parsed_output is None:
            raise ValueError(f"Accepted slot lacks parsed output: {attempt.slot_id}")
        grouped[(attempt.state_id, attempt.condition)].append(attempt)
    states = sorted({state_id for state_id, _ in grouped})
    if len(states) != 10 or len(grouped) != 20:
        raise ValueError(f"Expected 10 states and 20 cells, found {len(states)} and {len(grouped)}")
    for state_id in states:
        if {(state_id, "authority"), (state_id, "hedge")} - set(grouped):
            raise ValueError(f"Pilot state lacks an A/H cell: {state_id}")

    variances: dict[str, float] = {}
    v_by_state: list[float] = []
    for state_id in states:
        state_variance = 0.0
        for condition in ("authority", "hedge"):
            cell = grouped[(state_id, condition)]
            if len(cell) != frozen.pilot_replicates_per_side:
                raise ValueError(
                    f"Cell {state_id}|{condition} has {len(cell)} replicates, expected 10"
                )
            replicate_indexes = {attempt.replicate_index for attempt in cell}
            if replicate_indexes != set(range(1, 11)):
                raise ValueError(f"Cell {state_id}|{condition} has invalid replicate indexes")
            values = np.asarray(
                [attempt.parsed_output.reshare for attempt in cell], dtype=float
            )
            variance = float(np.var(values, ddof=1))
            variances[f"{state_id}|{condition}"] = variance
            state_variance += variance
        v_by_state.append(state_variance)

    deterministic = all(value == 0.0 for value in variances.values())
    if deterministic:
        return PilotDecision(
            decision_code="K_1_DETERMINISTIC",
            selected_k=1,
            deterministic=True,
            valid_slots=200,
            state_count=10,
            cell_count=20,
            q90_v=0.0,
            se_call_by_k={str(k): 0.0 for k in frozen.k_candidates},
            cell_variance_by_key=variances,
            criterion=frozen.practical_delta / 4.0,
            near_deterministic=True,
        )

    q90_v = float(np.quantile(np.asarray(v_by_state), 0.9, method="higher"))
    criterion = frozen.practical_delta / 4.0
    se_by_k = {
        str(k): float(np.sqrt(q90_v / (frozen.confirmatory_states * k)))
        for k in frozen.k_candidates
    }
    selected_k = next(
        (k for k in frozen.k_candidates if se_by_k[str(k)] <= criterion),
        None,
    )
    if selected_k is None:
        decision_code = "HALT_K_INSUFFICIENT"
    else:
        decision_code = f"K_{selected_k}"
    return PilotDecision(
        decision_code=decision_code,
        selected_k=selected_k,
        deterministic=False,
        valid_slots=200,
        state_count=10,
        cell_count=20,
        q90_v=q90_v,
        se_call_by_k=se_by_k,
        cell_variance_by_key=variances,
        criterion=criterion,
        near_deterministic=q90_v < 1e-12,
    )


def _directional_label(interval: tuple[float, float]) -> str:
    lower, upper = interval
    if lower > 0.0:
        return "directional_replication"
    if upper < 0.0:
        return "sign_reversal"
    return "direction_unresolved"


def _practical_label(interval: tuple[float, float], delta: float) -> str:
    lower, upper = interval
    if lower > delta:
        return "operationally_meaningful_positive"
    if upper < -delta:
        return "operationally_meaningful_negative"
    if lower >= -delta and upper <= delta:
        return "practical_equivalence"
    return "practical_magnitude_unresolved"


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def analyze_confirmatory_reshare(
    attempts: Iterable[AttemptRecord],
    states: Iterable[StateRecord],
    selected_k: int,
    config: V2FrozenConfig | None = None,
    *,
    _primary_resamples: int = 20_000,
    _template_resamples: int = 20_000,
    _permutation_resamples: int = 100_000,
) -> ConfirmatoryResult:
    """Run the preregistered primary reshare analysis on integrity-passed calls.

    Underscored resample arguments exist only for fast deterministic unit tests;
    production callers and the future CLI use the frozen defaults.
    """

    frozen = config or V2FrozenConfig()
    if selected_k not in {1, 3, 5, 10}:
        raise ValueError("selected_k must be one of 1, 3, 5, 10")
    confirmatory_states = tuple(state for state in states if state.allocation == "confirmatory")
    if len(confirmatory_states) != 60 or len({state.scenario_id for state in confirmatory_states}) != 60:
        raise ValueError("Confirmatory analysis requires 60 scenario-unique frozen states")
    state_by_id = {state.state_id: state for state in confirmatory_states}
    accepted = tuple(attempt for attempt in attempts if attempt.accepted)
    expected_calls = 60 * 2 * selected_k
    if len(accepted) != expected_calls:
        raise ValueError(f"Expected {expected_calls} accepted confirmatory calls, found {len(accepted)}")
    if any(attempt.phase != "confirmatory" for attempt in accepted):
        raise ValueError("Confirmatory analysis received a non-confirmatory attempt")
    if len({attempt.slot_id for attempt in accepted}) != expected_calls:
        raise ValueError("Confirmatory accepted slots are not unique")
    response_ids = [attempt.response_id for attempt in accepted]
    usage_ids = [attempt.usage_event_id for attempt in accepted]
    if None in response_ids or len(set(response_ids)) != expected_calls:
        raise ValueError("Confirmatory response IDs are missing or reused")
    if None in usage_ids or len(set(usage_ids)) != expected_calls:
        raise ValueError("Confirmatory usage events are missing or reused")

    grouped: dict[tuple[str, str], list[AttemptRecord]] = defaultdict(list)
    for attempt in accepted:
        if attempt.state_id not in state_by_id or attempt.parsed_output is None:
            raise ValueError(f"Unknown state or missing output for {attempt.slot_id}")
        grouped[(attempt.state_id, attempt.condition)].append(attempt)
    if len(grouped) != 120:
        raise ValueError(f"Expected 120 state-condition cells, found {len(grouped)}")

    raw_by_state: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    contrasts: dict[str, float] = {}
    for state_id in sorted(state_by_id):
        arrays = []
        for condition in ("authority", "hedge"):
            cell = grouped.get((state_id, condition), [])
            if len(cell) != selected_k:
                raise ValueError(f"Cell {state_id}|{condition} has {len(cell)} calls, expected {selected_k}")
            if {attempt.replicate_index for attempt in cell} != set(range(1, selected_k + 1)):
                raise ValueError(f"Cell {state_id}|{condition} has invalid replicate indexes")
            ordered = sorted(cell, key=lambda attempt: attempt.replicate_index)
            arrays.append(np.asarray([attempt.parsed_output.reshare for attempt in ordered]))
        authority, hedge = arrays
        raw_by_state[state_id] = (authority, hedge)
        contrasts[state_id] = float(np.mean(authority) - np.mean(hedge))

    topic_counts: dict[str, int] = defaultdict(int)
    cell_values: dict[str, np.ndarray] = {}
    for state in confirmatory_states:
        topic_counts[state.topic] += 1
    if set(topic_counts.values()) != {12} or len(topic_counts) != 5:
        raise ValueError(f"Confirmatory topic balance is invalid: {dict(topic_counts)}")
    for cell in sorted({state.template_cell for state in confirmatory_states}):
        ids = [state.state_id for state in confirmatory_states if state.template_cell == cell]
        cell_values[cell] = np.asarray([contrasts[state_id] for state_id in ids], dtype=float)
    estimate = float(np.mean(list(contrasts.values())))

    primary_rng = np.random.default_rng(frozen.primary_bootstrap_seed)
    primary_draws = np.empty(_primary_resamples, dtype=float)
    for draw in range(_primary_resamples):
        sampled = [
            primary_rng.choice(values, size=len(values), replace=True)
            for values in cell_values.values()
        ]
        primary_draws[draw] = float(np.mean(np.concatenate(sampled)))
    fixed_interval = _percentile_interval(primary_draws)

    cluster_means = {
        cell: float(np.mean(values)) for cell, values in cell_values.items()
    }
    cluster_counts = {cell: len(values) for cell, values in cell_values.items()}
    template_rng = np.random.default_rng(frozen.template_bootstrap_seed)
    template_draws = np.empty(_template_resamples, dtype=float)
    topics = sorted(topic_counts)
    for draw in range(_template_resamples):
        topic_draws = []
        for topic in topics:
            cells = [f"{topic}|{tier}" for tier in ("low", "mid", "high")]
            probabilities = np.asarray([cluster_counts[cell] for cell in cells], dtype=float)
            probabilities /= probabilities.sum()
            sampled_cells = template_rng.choice(cells, size=3, replace=True, p=probabilities)
            topic_draws.append(float(np.mean([cluster_means[str(cell)] for cell in sampled_cells])))
        template_draws[draw] = float(np.mean(topic_draws))
    template_interval = _percentile_interval(template_draws)

    deterministic = selected_k == 1
    if deterministic:
        permutation_p = None
        actual_permutations = 0
    else:
        permutation_rng = np.random.default_rng(frozen.permutation_seed)
        extreme = 0
        for _ in range(_permutation_resamples):
            permuted_contrasts = []
            for state_id in sorted(raw_by_state):
                authority, hedge = raw_by_state[state_id]
                pooled = np.concatenate([authority, hedge])
                permuted = permutation_rng.permutation(pooled)
                permuted_contrasts.append(
                    float(np.mean(permuted[:selected_k]) - np.mean(permuted[selected_k:]))
                )
            if abs(float(np.mean(permuted_contrasts))) >= abs(estimate):
                extreme += 1
        permutation_p = (1.0 + extreme) / (1.0 + _permutation_resamples)
        actual_permutations = _permutation_resamples

    direction = _directional_label(fixed_interval)
    practical = _practical_label(fixed_interval, frozen.practical_delta)
    template_direction = _directional_label(template_interval)
    template_practical = _practical_label(template_interval, frozen.practical_delta)
    return ConfirmatoryResult(
        selected_k=selected_k,
        deterministic=deterministic,
        state_count=60,
        estimate=estimate,
        fixed_template_ci=fixed_interval,
        template_cluster_ci=template_interval,
        directional_label=direction,
        practical_label=practical,
        template_sensitive=(direction, practical) != (template_direction, template_practical),
        permutation_p_value=permutation_p,
        primary_bootstrap_resamples=_primary_resamples,
        template_bootstrap_resamples=_template_resamples,
        permutation_resamples=actual_permutations,
        state_contrasts=contrasts,
    )
