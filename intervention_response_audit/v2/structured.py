"""Frozen scenario-level cross-fitting for the secondary v2 structured predictor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from intervention_response_audit.audit import EPS, build_feature_frame
from intervention_response_audit.network import TOPICS
from intervention_response_audit.v2.schema import StateRecord, V2FrozenConfig, sha256_text


C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)


@dataclass(frozen=True)
class StructuredCrossfitResult:
    predictions: pd.DataFrame
    outer_fold_by_scenario: dict[str, int]
    selected_c_by_outer_fold: dict[int, float]
    anchor_prediction_by_state: dict[str, float]
    r_global_full: float
    feature_count: int


def _add_constraint(rows, lower, upper, values, lo, hi) -> None:
    rows.append(np.asarray(values, dtype=float))
    lower.append(float(lo))
    upper.append(float(hi))


def constrained_outer_folds(rows: pd.DataFrame, seed: int = 20260720) -> dict[str, int]:
    """Assign 80 scenarios to five balanced 16-scenario folds by MILP."""

    metadata = (
        rows.assign(scenario_id=rows["scenario_id"].astype(str))
        .sort_values("scenario_id", kind="mergesort")
        .groupby("scenario_id", sort=True)
        .agg(topic=("topic", "first"), credibility=("credibility", "first"),
             topic_n=("topic", "nunique"), credibility_n=("credibility", "nunique"))
        .reset_index()
    )
    if len(metadata) != 80:
        raise ValueError(f"Structured analysis requires exactly 80 normal scenarios, found {len(metadata)}")
    if metadata["topic_n"].max() != 1 or metadata["credibility_n"].max() != 1:
        raise ValueError("Topic and credibility must be scenario-constant")
    folds = 5
    scenario_count = len(metadata)
    variables = scenario_count * folds
    constraints: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    for scenario_index in range(scenario_count):
        row = np.zeros(variables)
        row[scenario_index * folds : (scenario_index + 1) * folds] = 1.0
        _add_constraint(constraints, lower, upper, row, 1, 1)
    for fold in range(folds):
        row = np.zeros(variables)
        row[fold::folds] = 1.0
        _add_constraint(constraints, lower, upper, row, 16, 16)
    for topic in TOPICS:
        total = int((metadata["topic"] == topic).sum())
        for fold in range(folds):
            row = np.zeros(variables)
            matching = np.flatnonzero((metadata["topic"] == topic).to_numpy())
            row[matching * folds + fold] = 1.0
            _add_constraint(
                constraints, lower, upper, row, total // folds, int(np.ceil(total / folds))
            )
    for credibility in sorted(metadata["credibility"].unique()):
        total = int((metadata["credibility"] == credibility).sum())
        for fold in range(folds):
            row = np.zeros(variables)
            matching = np.flatnonzero((metadata["credibility"] == credibility).to_numpy())
            row[matching * folds + fold] = 1.0
            _add_constraint(
                constraints, lower, upper, row, total // folds, int(np.ceil(total / folds))
            )

    rng = np.random.default_rng(seed)
    result = milp(
        rng.random(variables),
        integrality=np.ones(variables),
        bounds=Bounds(np.zeros(variables), np.ones(variables)),
        constraints=LinearConstraint(np.vstack(constraints), np.asarray(lower), np.asarray(upper)),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Outer-fold allocation is infeasible: {result.message}")
    assignment: dict[str, int] = {}
    matrix = result.x.reshape(scenario_count, folds)
    for scenario_index, scenario_id in enumerate(metadata["scenario_id"]):
        selected = np.flatnonzero(matrix[scenario_index] > 0.5)
        if len(selected) != 1:
            raise RuntimeError(f"Scenario {scenario_id} was not assigned exactly once")
        assignment[str(scenario_id)] = int(selected[0])
    return assignment


def _inner_folds(scenarios: Iterable[str], seed: int, outer_fold: int) -> dict[str, int]:
    ordered = sorted(
        {str(item) for item in scenarios},
        key=lambda item: sha256_text(f"{seed}|inner|{outer_fold}|{item}"),
    )
    return {scenario: index % 5 for index, scenario in enumerate(ordered)}


def _make_model(c_value: float, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            solver="liblinear",
            max_iter=1000,
            random_state=seed,
        ),
    )


def _fit_soft(model, x: pd.DataFrame, p: np.ndarray):
    expanded_x = pd.concat([x, x], ignore_index=True)
    y = np.concatenate([np.ones(len(x), dtype=int), np.zeros(len(x), dtype=int)])
    weights = np.concatenate([p, 1.0 - p])
    model.fit(expanded_x, y, logisticregression__sample_weight=weights)
    return model


def _predict(model, x: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict_proba(x)[:, 1], EPS, 1.0 - EPS)


def _cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0 - EPS)
    q = np.clip(q, EPS, 1.0 - EPS)
    return float(np.mean(-(p * np.log(q) + (1.0 - p) * np.log(1.0 - q))))


def _kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    q = np.clip(q, EPS, 1.0 - EPS)
    return p * np.log(p / q) + (1.0 - p) * np.log((1.0 - p) / (1.0 - q))


def run_structured_crossfit(
    rows: pd.DataFrame,
    states: Iterable[StateRecord],
    target: str = "reshare",
    config: V2FrozenConfig | None = None,
) -> StructuredCrossfitResult:
    """Fit honest soft-label predictions and map them to frozen v2 anchors."""

    frozen = config or V2FrozenConfig()
    if target not in {"reshare", "believe"}:
        raise ValueError("Structured target must be reshare or believe")
    work = rows.reset_index(drop=True).copy()
    if len(work) == 0 or work["scenario_id"].astype(str).nunique() != 80:
        raise ValueError("Only the complete 80-scenario v1 normal corpus is permitted")
    if set(work["scaffold_id"].astype(int).unique()) != {0}:
        raise ValueError("Scaffold rows are not permitted in structured cross-fitting")
    if work[target].isna().any():
        raise ValueError(f"Target {target} contains missing values")

    features = build_feature_frame(work)
    if features.shape[1] != 27:
        raise ValueError(f"Frozen structured feature count must be 27, found {features.shape[1]}")
    scenario_ids = work["scenario_id"].astype(str)
    assignment = constrained_outer_folds(work, frozen.crossfit_seed)
    predictions = np.full(len(work), np.nan)
    selected_cs: dict[int, float] = {}

    for outer_fold in range(5):
        outer_valid = scenario_ids.map(assignment).to_numpy() == outer_fold
        outer_train = ~outer_valid
        inner_assignment = _inner_folds(
            scenario_ids[outer_train].unique(), frozen.crossfit_seed, outer_fold
        )
        scores: list[tuple[float, float]] = []
        for c_value in C_GRID:
            weighted_loss = 0.0
            validation_rows = 0
            for inner_fold in range(5):
                inner_valid = outer_train & (
                    scenario_ids.map(inner_assignment).fillna(-1).to_numpy() == inner_fold
                )
                inner_train = outer_train & ~inner_valid
                model = _make_model(c_value, frozen.crossfit_seed + outer_fold)
                _fit_soft(
                    model,
                    features.loc[inner_train],
                    work.loc[inner_train, target].to_numpy(dtype=float),
                )
                p_valid = work.loc[inner_valid, target].to_numpy(dtype=float)
                loss = _cross_entropy(p_valid, _predict(model, features.loc[inner_valid]))
                weighted_loss += loss * int(inner_valid.sum())
                validation_rows += int(inner_valid.sum())
            scores.append((weighted_loss / validation_rows, c_value))
        selected_c = min(scores, key=lambda item: (item[0], item[1]))[1]
        selected_cs[outer_fold] = selected_c
        model = _make_model(selected_c, frozen.crossfit_seed + outer_fold)
        _fit_soft(
            model,
            features.loc[outer_train],
            work.loc[outer_train, target].to_numpy(dtype=float),
        )
        predictions[outer_valid] = _predict(model, features.loc[outer_valid])

    if np.isnan(predictions).any():
        raise RuntimeError("Cross-fitting failed to predict every normal row")
    p = work[target].to_numpy(dtype=float)
    output = pd.DataFrame(
        {
            "source_jsonl_line": np.arange(1, len(work) + 1),
            "scenario_id": scenario_ids,
            "query_index": work["query_index"].astype(int),
            "agent_id": work["agent_id"].astype(int),
            "step": work["step"].astype(int),
            "target": target,
            "oracle_probability": p,
            "structured_probability": predictions,
            "outer_fold": scenario_ids.map(assignment).astype(int),
        }
    )
    anchors: dict[str, float] = {}
    anchor_lookup = {
        (str(row.scenario_id), int(row.step), int(row.agent_id), int(row.query_index)): index
        for index, row in output.iterrows()
    }
    if len(anchor_lookup) != len(output):
        raise ValueError("Normal-row anchor keys are not unique")
    for state in states:
        if state.allocation != "confirmatory":
            continue
        key = (state.scenario_id, state.step, state.agent_id, state.query_index)
        if key not in anchor_lookup:
            raise ValueError(f"Anchor row key is missing for {state.state_id}")
        row_index = anchor_lookup[key]
        if assignment[state.scenario_id] != int(output.iloc[row_index]["outer_fold"]):
            raise RuntimeError(f"Anchor fold mismatch for {state.state_id}")
        anchors[state.state_id] = float(predictions[row_index])
    if len(anchors) != 60:
        raise ValueError(f"Expected 60 confirmatory anchor predictions, found {len(anchors)}")
    return StructuredCrossfitResult(
        predictions=output,
        outer_fold_by_scenario=assignment,
        selected_c_by_outer_fold=selected_cs,
        anchor_prediction_by_state=anchors,
        r_global_full=float(np.mean(_kl(p, predictions))),
        feature_count=features.shape[1],
    )
