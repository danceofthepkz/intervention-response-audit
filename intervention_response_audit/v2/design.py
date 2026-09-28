"""Offline Stage-0 state allocation and authority/hedge prompt freezing."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from intervention_response_audit.network import CREDIBILITY_TIERS, TOPICS
from intervention_response_audit.v2.schema import (
    AcceptanceCheck,
    AcceptanceReport,
    PromptRecord,
    ReserveRenderCheck,
    StateRecord,
    TaskRecord,
    V2FrozenConfig,
    canonical_json,
    content_sha256,
    sha256_bytes,
    sha256_text,
    write_json,
    write_jsonl,
)
from intervention_response_audit.v2.templates import V2_TEMPLATE_BANK


SELECTOR_COLUMNS = (
    "scenario_id",
    "query_index",
    "scaffold_id",
    "step",
    "agent_id",
    "n_exposures",
    "sum_tie_strength",
    "max_tie_strength",
    "same_major_exposers",
    "same_club_exposers",
    "source_diversity",
    "n_endorse",
    "n_neutral",
    "n_doubt",
    "topic",
    "urgency",
    "credibility",
    "controversy",
    "socialness",
    "trust",
    "topic_interest",
    "year",
    "mechanism",
)

STRUCTURED_FEATURE_COLUMNS = (
    "n_exposures",
    "sum_tie_strength",
    "max_tie_strength",
    "same_major_exposers",
    "same_club_exposers",
    "source_diversity",
    "n_endorse",
    "n_neutral",
    "n_doubt",
    "step",
    "urgency",
    "credibility",
    "controversy",
    "socialness",
    "trust",
    "topic_interest",
    "year",
    "topic",
    "mechanism",
    "scaffold_id",
)

FORBIDDEN_SELECTOR_COLUMNS = frozenset(
    {
        "believe",
        "reshare",
        "attend",
        "reason_code",
        "sampled_reshare",
        "framing_type",
        "wording_id",
        "condition_text",
        "parsed_output",
        "raw_response",
    }
)

CREDIBILITY_VALUE_TO_TIER = {0.25: "low", 0.55: "mid", 0.82: "high"}
ALLOCATION_ORDER = {"confirmatory": 0, "pilot": 1, "reserve": 2}


@dataclass(frozen=True)
class ArchivedPrompt:
    scenario_id: str
    step: int
    agent_id: int
    full_prompt: str
    full_prompt_sha256: str
    source_jsonl_line: int


@dataclass(frozen=True)
class Stage0BuildResult:
    output_dir: Path
    report: AcceptanceReport
    states: tuple[StateRecord, ...]
    prompts: tuple[PromptRecord, ...]


def _native(value: Any) -> int | float | str:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (int, float, str)):
        return value
    raise TypeError(f"Unsupported manifest scalar: {type(value).__name__}")


def _key3(row: pd.Series | dict[str, Any]) -> tuple[str, int, int]:
    return (str(row["scenario_id"]), int(row["step"]), int(row["agent_id"]))


def _key4(row: pd.Series | dict[str, Any]) -> tuple[str, int, int, int]:
    return (*_key3(row), int(row["query_index"]))


def state_id_for(row: pd.Series | dict[str, Any]) -> str:
    """Derive the canonical state ID; callers cannot supply one manually."""

    scenario_id, step, agent_id, query_index = _key4(row)
    identity = f"{scenario_id}|{agent_id}|{step}|{query_index}"
    return sha256_text(identity)


def load_archive_prompt_index(path: str | Path) -> dict[tuple[str, int, int], ArchivedPrompt]:
    """Project the archived log to prompt-integrity fields only.

    Scientific outputs exist in the source JSONL but are never accessed or
    returned by this function.  The resulting mapping is the only archive object
    visible to the selector.
    """

    index: dict[tuple[str, int, int], ArchivedPrompt] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            raw = json.loads(line)
            if raw.get("valid") is not True:
                continue
            full_prompt = raw.get("full_prompt")
            if not isinstance(full_prompt, str) or not full_prompt:
                continue
            key = (str(raw["scenario_id"]), int(raw["step"]), int(raw["agent_id"]))
            if key in index:
                raise ValueError(f"Archived prompt key is not unique: {key}")
            index[key] = ArchivedPrompt(
                scenario_id=key[0],
                step=key[1],
                agent_id=key[2],
                full_prompt=full_prompt,
                full_prompt_sha256=sha256_text(full_prompt),
                source_jsonl_line=line_number,
            )
    if not index:
        raise ValueError("No valid archived prompts found")
    return index


def load_selector_view(
    parquet_path: str | Path,
    eligible_prompt_keys: set[tuple[str, int, int]],
) -> pd.DataFrame:
    """Load only treatment-preceding columns and derive frozen strata."""

    if FORBIDDEN_SELECTOR_COLUMNS.intersection(SELECTOR_COLUMNS):
        raise AssertionError("Selector whitelist contains a forbidden outcome column")
    frame = pd.read_parquet(parquet_path, columns=list(SELECTOR_COLUMNS)).copy()
    if frame.empty:
        raise ValueError("Selector input is empty")
    frame = frame[frame.apply(lambda row: _key3(row) in eligible_prompt_keys, axis=1)].copy()
    if frame.empty:
        raise ValueError("No selector rows have archived full prompts")
    if frame.duplicated(["scenario_id", "step", "agent_id"]).any():
        raise ValueError("(scenario_id, step, agent_id) must uniquely identify normal rows")

    tiers: list[str] = []
    for value in frame["credibility"].astype(float):
        matches = [tier for scalar, tier in CREDIBILITY_VALUE_TO_TIER.items() if np.isclose(value, scalar)]
        if len(matches) != 1:
            raise ValueError(f"Unexpected credibility value in selector view: {value}")
        tiers.append(matches[0])
    frame["credibility_tier"] = tiers
    frame["exposure_bin"] = np.where(
        frame["n_exposures"].astype(int) == 1,
        "1",
        np.where(frame["n_exposures"].astype(int) == 2, "2", "3+"),
    )
    frame["step_bin"] = np.where(
        frame["step"].astype(int) == 1,
        "1",
        np.where(frame["step"].astype(int) == 2, "2", "3+"),
    )
    frame["template_cell"] = frame["topic"].astype(str) + "|" + frame["credibility_tier"]
    return frame.sort_values(
        ["scenario_id", "step", "agent_id", "query_index"], kind="mergesort"
    ).reset_index(drop=True)


def _add_constraint(
    rows: list[np.ndarray],
    lower: list[float],
    upper: list[float],
    mask: np.ndarray,
    lo: float,
    hi: float,
) -> None:
    rows.append(mask.astype(float))
    lower.append(float(lo))
    upper.append(float(hi))


def select_confirmatory_states(frame: pd.DataFrame, config: V2FrozenConfig) -> pd.DataFrame:
    """Solve the frozen exact-margin binary allocation problem."""

    candidates = frame.sort_values(
        ["scenario_id", "step", "agent_id", "query_index"], kind="mergesort"
    ).reset_index(drop=True)
    n = len(candidates)
    constraint_rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    _add_constraint(constraint_rows, lower, upper, np.ones(n), config.confirmatory_states, config.confirmatory_states)
    for topic, target in sorted(config.topic_targets.items()):
        _add_constraint(constraint_rows, lower, upper, (candidates["topic"] == topic).to_numpy(), target, target)
    for tier, target in sorted(config.credibility_targets.items()):
        _add_constraint(
            constraint_rows,
            lower,
            upper,
            (candidates["credibility_tier"] == tier).to_numpy(),
            target,
            target,
        )
    for count_bin, target in sorted(config.exposure_targets.items()):
        _add_constraint(
            constraint_rows,
            lower,
            upper,
            (candidates["exposure_bin"] == count_bin).to_numpy(),
            target,
            target,
        )
    for step_bin, target in sorted(config.step_targets.items()):
        _add_constraint(
            constraint_rows,
            lower,
            upper,
            (candidates["step_bin"] == step_bin).to_numpy(),
            target,
            target,
        )
    for scenario_id in sorted(candidates["scenario_id"].astype(str).unique()):
        _add_constraint(
            constraint_rows,
            lower,
            upper,
            (candidates["scenario_id"] == scenario_id).to_numpy(),
            0,
            1,
        )
    for topic in TOPICS:
        for tier in CREDIBILITY_TIERS:
            _add_constraint(
                constraint_rows,
                lower,
                upper,
                ((candidates["topic"] == topic) & (candidates["credibility_tier"] == tier)).to_numpy(),
                config.min_states_per_template_cell,
                np.inf,
            )

    rng = np.random.default_rng(config.selector_seed)
    objective = rng.random(n)
    result = milp(
        objective,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(
            np.vstack(constraint_rows), np.asarray(lower), np.asarray(upper)
        ),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Confirmatory allocation is infeasible: {result.message}")
    selected = candidates.loc[result.x > 0.5].copy()
    if len(selected) != config.confirmatory_states:
        raise RuntimeError(f"MILP selected {len(selected)} confirmatory rows, expected 60")
    selected["allocation"] = "confirmatory"
    return selected


def select_pilot_states(
    remaining: pd.DataFrame, config: V2FrozenConfig
) -> pd.DataFrame:
    """Select two pilot scenarios per topic while matching main-study bins."""

    candidates = remaining.sort_values(
        ["scenario_id", "step", "agent_id", "query_index"], kind="mergesort"
    ).reset_index(drop=True)
    n = len(candidates)
    categories: list[tuple[str, str, float]] = []
    for label, target in sorted(config.exposure_targets.items()):
        categories.append(("exposure_bin", label, target * config.pilot_states / config.confirmatory_states))
    for label, target in sorted(config.step_targets.items()):
        categories.append(("step_bin", label, target * config.pilot_states / config.confirmatory_states))
    deviation_count = 2 * len(categories)
    total_variables = n + deviation_count
    constraints: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def padded(mask: np.ndarray) -> np.ndarray:
        return np.concatenate([mask.astype(float), np.zeros(deviation_count)])

    _add_constraint(constraints, lower, upper, padded(np.ones(n)), config.pilot_states, config.pilot_states)
    for topic in TOPICS:
        _add_constraint(
            constraints,
            lower,
            upper,
            padded((candidates["topic"] == topic).to_numpy()),
            2,
            2,
        )
    for scenario_id in sorted(candidates["scenario_id"].astype(str).unique()):
        _add_constraint(
            constraints,
            lower,
            upper,
            padded((candidates["scenario_id"] == scenario_id).to_numpy()),
            0,
            1,
        )
    for index, (column, label, target) in enumerate(categories):
        row = padded((candidates[column] == label).to_numpy())
        row[n + 2 * index] = -1.0
        row[n + 2 * index + 1] = 1.0
        _add_constraint(constraints, lower, upper, row, target, target)

    rng = np.random.default_rng(config.selector_seed)
    objective = np.concatenate([rng.random(n) * 1e-6, np.ones(deviation_count)])
    integrality = np.concatenate([np.ones(n), np.zeros(deviation_count)])
    bounds = Bounds(
        np.zeros(total_variables),
        np.concatenate([np.ones(n), np.full(deviation_count, np.inf)]),
    )
    result = milp(
        objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(np.vstack(constraints), np.asarray(lower), np.asarray(upper)),
        options={"mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Pilot allocation is infeasible: {result.message}")
    selected = candidates.loc[result.x[:n] > 0.5].copy()
    if len(selected) != config.pilot_states:
        raise RuntimeError(f"MILP selected {len(selected)} pilot rows, expected 10")
    selected["allocation"] = "pilot"
    return selected


def select_reserve_states(
    remaining: pd.DataFrame,
    used_scenarios: set[str],
    config: V2FrozenConfig,
) -> pd.DataFrame:
    """Choose one renderable state from each unused reserve scenario."""

    reserve_pool = remaining[
        ~remaining["scenario_id"].astype(str).isin(used_scenarios)
    ].copy()
    rows: list[pd.Series] = []
    for scenario_id, group in reserve_pool.groupby("scenario_id", sort=True):
        ranked = group.copy()
        ranked["_rank"] = ranked.apply(
            lambda row: sha256_text(
                f"{config.selector_seed}|reserve|{state_id_for(row)}"
            ),
            axis=1,
        )
        rows.append(ranked.sort_values("_rank", kind="mergesort").iloc[0].drop(labels="_rank"))
    selected = pd.DataFrame(rows).reset_index(drop=True)
    if len(selected) != config.reserve_states:
        raise RuntimeError(f"Selected {len(selected)} reserve scenarios, expected 10")
    selected["allocation"] = "reserve"
    return selected


def allocate_all_states(frame: pd.DataFrame, config: V2FrozenConfig) -> pd.DataFrame:
    confirmatory = select_confirmatory_states(frame, config)
    remaining = frame[
        ~frame["scenario_id"].astype(str).isin(confirmatory["scenario_id"].astype(str))
    ].copy()
    pilot = select_pilot_states(remaining, config)
    reserve = select_reserve_states(
        remaining,
        set(pilot["scenario_id"].astype(str)),
        config,
    )
    allocated = pd.concat([confirmatory, pilot, reserve], ignore_index=True)
    allocated["_allocation_order"] = allocated["allocation"].map(ALLOCATION_ORDER)
    return allocated.sort_values(
        ["_allocation_order", "scenario_id"], kind="mergesort"
    ).drop(columns="_allocation_order").reset_index(drop=True)


def load_condition_texts(
    parquet_path: str | Path,
    selected_keys: set[tuple[str, int, int, int]],
) -> dict[tuple[str, int, int, int], str]:
    """Load original wording only after state identities have been frozen."""

    columns = ["scenario_id", "step", "agent_id", "query_index", "condition_text"]
    frame = pd.read_parquet(parquet_path, columns=columns)
    output: dict[tuple[str, int, int, int], str] = {}
    for _, row in frame.iterrows():
        key = _key4(row)
        if key not in selected_keys:
            continue
        if key in output:
            raise ValueError(f"Condition-text key is not unique: {key}")
        output[key] = str(row["condition_text"])
    missing = selected_keys.difference(output)
    if missing:
        raise ValueError(f"Missing condition text for {len(missing)} selected states")
    return output


def state_records_from_allocation(
    allocated: pd.DataFrame,
    archive_index: dict[tuple[str, int, int], ArchivedPrompt],
) -> tuple[StateRecord, ...]:
    records: list[StateRecord] = []
    for _, row in allocated.iterrows():
        archived = archive_index[_key3(row)]
        structured = {column: _native(row[column]) for column in STRUCTURED_FEATURE_COLUMNS}
        source_hash = content_sha256(
            {
                "identity": list(_key4(row)),
                "structured_features": structured,
            }
        )
        records.append(
            StateRecord(
                state_id=state_id_for(row),
                scenario_id=str(row["scenario_id"]),
                agent_id=int(row["agent_id"]),
                step=int(row["step"]),
                query_index=int(row["query_index"]),
                allocation=str(row["allocation"]),
                topic=str(row["topic"]),
                credibility_tier=str(row["credibility_tier"]),
                exposure_bin=str(row["exposure_bin"]),
                step_bin=str(row["step_bin"]),
                template_cell=str(row["template_cell"]),
                structured_features=structured,
                source_row_sha256=source_hash,
                archived_full_prompt_sha256=archived.full_prompt_sha256,
                source_jsonl_line=archived.source_jsonl_line,
            )
        )
    return tuple(records)


def render_prompt_pair(
    state: StateRecord,
    archived: ArchivedPrompt,
    original_message: str,
    config_sha256: str,
) -> tuple[PromptRecord, PromptRecord]:
    """Replace exactly one archived message span and nothing else."""

    if archived.full_prompt.count(original_message) != 1:
        raise ValueError(
            f"Original message must occur exactly once for {state.state_id}; "
            f"found {archived.full_prompt.count(original_message)}"
        )
    start = archived.full_prompt.index(original_message)
    source_end = start + len(original_message)
    prefix = archived.full_prompt[:start]
    suffix = archived.full_prompt[source_end:]
    outside_hash = content_sha256({"prefix": prefix, "suffix": suffix})
    records: list[PromptRecord] = []
    for condition in ("authority", "hedge"):
        variant = V2_TEMPLATE_BANK[state.topic][state.credibility_tier][condition]
        message = str(variant["text"])
        full_prompt = prefix + message + suffix
        records.append(
            PromptRecord(
                state_id=state.state_id,
                scenario_id=state.scenario_id,
                condition=condition,
                template_id=f"{state.topic}:{state.credibility_tier}:{condition}",
                full_prompt=full_prompt,
                prompt_sha256=sha256_text(full_prompt),
                outside_message_sha256=outside_hash,
                message_text=message,
                message_start=start,
                message_end=start + len(message),
                message_start_byte=len(prefix.encode("utf-8")),
                message_end_byte=len((prefix + message).encode("utf-8")),
                config_sha256=config_sha256,
            )
        )
    if records[0].prompt_sha256 == records[1].prompt_sha256:
        raise ValueError(f"A/H prompts are byte-identical for {state.state_id}")
    return records[0], records[1]


def render_frozen_prompts(
    states: Iterable[StateRecord],
    archive_index: dict[tuple[str, int, int], ArchivedPrompt],
    condition_texts: dict[tuple[str, int, int, int], str],
    config_sha256: str,
) -> tuple[tuple[PromptRecord, ...], tuple[ReserveRenderCheck, ...]]:
    prompts: list[PromptRecord] = []
    reserve_checks: list[ReserveRenderCheck] = []
    for state in states:
        key3 = (state.scenario_id, state.step, state.agent_id)
        key4 = (*key3, state.query_index)
        authority, hedge = render_prompt_pair(
            state,
            archive_index[key3],
            condition_texts[key4],
            config_sha256,
        )
        if state.allocation == "reserve":
            reserve_checks.append(
                ReserveRenderCheck(
                    state_id=state.state_id,
                    scenario_id=state.scenario_id,
                    authority_prompt_sha256=authority.prompt_sha256,
                    hedge_prompt_sha256=hedge.prompt_sha256,
                    outside_message_sha256=authority.outside_message_sha256,
                    renderable=authority.outside_message_sha256 == hedge.outside_message_sha256,
                )
            )
        else:
            prompts.extend((authority, hedge))
    return tuple(prompts), tuple(reserve_checks)


def _state_frame(states: Iterable[StateRecord]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state in states:
        row = state.model_dump(mode="json")
        features = row.pop("structured_features")
        row.update({f"feature_{key}": value for key, value in features.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _template_validation_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for topic in TOPICS:
        for tier in CREDIBILITY_TIERS:
            rows.append(
                {
                    "template_cell": f"{topic}|{tier}",
                    "topic": topic,
                    "credibility_tier": tier,
                    "authority_text": V2_TEMPLATE_BANK[topic][tier]["authority"]["text"],
                    "hedge_text": V2_TEMPLATE_BANK[topic][tier]["hedge"]["text"],
                    "factual_payload_constant": "",
                    "authority_manipulation_success": "",
                    "hedge_manipulation_success": "",
                    "reviewer_id": "",
                    "reviewed_at_utc": "",
                    "notes": "",
                }
            )
    return pd.DataFrame(rows)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def _counts(records: Iterable[StateRecord], attribute: str) -> dict[str, int]:
    output: dict[str, int] = {}
    for record in records:
        key = str(getattr(record, attribute))
        output[key] = output.get(key, 0) + 1
    return dict(sorted(output.items()))


def _manual_review_result(frame: pd.DataFrame) -> tuple[str, str]:
    """Classify the blind review as pending, passing, or substantively failing."""

    required_boolean = (
        "factual_payload_constant",
        "authority_manipulation_success",
        "hedge_manipulation_success",
    )
    truthy = {"yes", "true", "pass", "1"}
    falsy = {"no", "false", "fail", "0"}
    if len(frame) != 15:
        return "PENDING_MANUAL_REVIEW", f"reviewed_pairs=0/15; rows={len(frame)}"
    normalized: dict[str, pd.Series] = {}
    for column in required_boolean:
        if column not in frame:
            return "PENDING_MANUAL_REVIEW", f"missing_column={column}"
        normalized[column] = frame[column].fillna("").astype(str).str.strip().str.lower()
        if not normalized[column].isin(truthy | falsy).all():
            completed = int(normalized[column].isin(truthy | falsy).sum())
            return "PENDING_MANUAL_REVIEW", f"completed_values={completed}/15; column={column}"
    for column in ("reviewer_id", "reviewed_at_utc"):
        if column not in frame or frame[column].fillna("").astype(str).str.strip().eq("").any():
            return "PENDING_MANUAL_REVIEW", f"missing_reviewer_metadata={column}"
    failed_mask = pd.Series(False, index=frame.index)
    for column in required_boolean:
        failed_mask |= normalized[column].isin(falsy)
    if failed_mask.any():
        failed_rows = frame.loc[failed_mask]
        notes_complete = failed_rows["notes"].fillna("").astype(str).str.strip().ne("").all()
        if not notes_complete:
            return "PENDING_MANUAL_REVIEW", "failed_rows_have_missing_notes"
        failed_cells = sorted(failed_rows["template_cell"].astype(str).tolist())
        return "FAIL", f"reviewed_pairs=15/15; failed_cells={canonical_json(failed_cells)}"
    return "PASS", "reviewed_pairs=15/15; failed_cells=[]"


def verify_stage0_records(
    states: tuple[StateRecord, ...],
    prompts: tuple[PromptRecord, ...],
    reserve_checks: tuple[ReserveRenderCheck, ...],
    config: V2FrozenConfig,
    validation_frame: pd.DataFrame,
    selector_loaded_columns: Iterable[str] = SELECTOR_COLUMNS,
    reproducible_manifest: bool = True,
) -> AcceptanceReport:
    """Recompute the complete E1 gate from frozen records."""

    checks: list[AcceptanceCheck] = []

    def add(check_id: str, passed: bool, evidence: str) -> None:
        checks.append(AcceptanceCheck(check_id=check_id, status="PASS" if passed else "FAIL", evidence=evidence))

    role_counts = _counts(states, "allocation")
    add(
        "E1_STATE_ROLE_COUNTS",
        role_counts == {"confirmatory": 60, "pilot": 10, "reserve": 10},
        canonical_json(role_counts),
    )
    scenario_ids = [state.scenario_id for state in states]
    add(
        "E1_SCENARIO_UNIQUENESS",
        len(states) == 80 and len(set(scenario_ids)) == 80,
        f"states={len(states)}, unique_scenarios={len(set(scenario_ids))}",
    )
    state_hash_integrity = True
    for state in states:
        expected_state_id = sha256_text(
            f"{state.scenario_id}|{state.agent_id}|{state.step}|{state.query_index}"
        )
        expected_source_hash = content_sha256(
            {
                "identity": [
                    state.scenario_id,
                    state.step,
                    state.agent_id,
                    state.query_index,
                ],
                "structured_features": state.structured_features,
            }
        )
        state_hash_integrity = state_hash_integrity and (
            state.state_id == expected_state_id
            and state.source_row_sha256 == expected_source_hash
        )
    add(
        "E1_STATE_HASH_INTEGRITY",
        state_hash_integrity,
        f"recomputed_records={len(states)}",
    )
    confirmatory = tuple(state for state in states if state.allocation == "confirmatory")
    pilot = tuple(state for state in states if state.allocation == "pilot")
    reserve = tuple(state for state in states if state.allocation == "reserve")
    add("E1_TOPIC_MARGINS", _counts(confirmatory, "topic") == dict(sorted(config.topic_targets.items())), canonical_json(_counts(confirmatory, "topic")))
    add("E1_CREDIBILITY_MARGINS", _counts(confirmatory, "credibility_tier") == dict(sorted(config.credibility_targets.items())), canonical_json(_counts(confirmatory, "credibility_tier")))
    add("E1_EXPOSURE_MARGINS", _counts(confirmatory, "exposure_bin") == dict(sorted(config.exposure_targets.items())), canonical_json(_counts(confirmatory, "exposure_bin")))
    add("E1_STEP_MARGINS", _counts(confirmatory, "step_bin") == dict(sorted(config.step_targets.items())), canonical_json(_counts(confirmatory, "step_bin")))
    cell_counts = _counts(confirmatory, "template_cell")
    add(
        "E1_TEMPLATE_CELL_COVERAGE",
        len(cell_counts) == 15 and min(cell_counts.values(), default=0) >= config.min_states_per_template_cell,
        canonical_json(cell_counts),
    )
    expected_small_topics = {topic: 2 for topic in sorted(TOPICS)}
    add("E1_PILOT_TOPIC_BALANCE", _counts(pilot, "topic") == expected_small_topics, canonical_json(_counts(pilot, "topic")))
    add("E1_RESERVE_TOPIC_BALANCE", _counts(reserve, "topic") == expected_small_topics, canonical_json(_counts(reserve, "topic")))
    loaded = set(selector_loaded_columns)
    forbidden_seen = sorted(loaded.intersection(FORBIDDEN_SELECTOR_COLUMNS))
    add("E1_OUTPUT_BLIND_SELECTOR", not forbidden_seen, f"forbidden_loaded={forbidden_seen}; loaded={sorted(loaded)}")
    add("E1_REPRODUCIBLE_MANIFEST", reproducible_manifest, f"reproducible_manifest={reproducible_manifest}")

    add("E1_PROMPT_COUNT", len(prompts) == 140, f"prompt_records={len(prompts)}")
    prompt_groups: dict[str, list[PromptRecord]] = {}
    for prompt in prompts:
        prompt_groups.setdefault(prompt.state_id, []).append(prompt)
    pair_integrity = len(prompt_groups) == 70
    state_by_id = {state.state_id: state for state in states}
    prompt_hash_integrity = True
    duplicate_hashes: dict[str, list[str]] = {}
    by_hash: dict[str, list[str]] = {}
    for state_id, group in prompt_groups.items():
        sides = {item.condition for item in group}
        outside = {item.outside_message_sha256 for item in group}
        hashes = {item.prompt_sha256 for item in group}
        state = state_by_id.get(state_id)
        pair_integrity = (
            pair_integrity
            and state is not None
            and state.allocation != "reserve"
            and len(group) == 2
            and sides == {"authority", "hedge"}
            and len(outside) == 1
            and len(hashes) == 2
        )
        for item in group:
            valid_span = (
                0 <= item.message_start <= item.message_end <= len(item.full_prompt)
                and item.full_prompt[item.message_start:item.message_end] == item.message_text
            )
            prompt_bytes = item.full_prompt.encode("utf-8")
            valid_byte_span = (
                0 <= item.message_start_byte <= item.message_end_byte <= len(prompt_bytes)
                and prompt_bytes[item.message_start_byte:item.message_end_byte]
                == item.message_text.encode("utf-8")
            )
            if valid_span:
                prefix = item.full_prompt[:item.message_start]
                suffix = item.full_prompt[item.message_end:]
                recomputed_outside = content_sha256({"prefix": prefix, "suffix": suffix})
            else:
                recomputed_outside = ""
            expected_message = ""
            if state is not None:
                expected_message = str(
                    V2_TEMPLATE_BANK[state.topic][state.credibility_tier][item.condition]["text"]
                )
            prompt_hash_integrity = prompt_hash_integrity and (
                item.prompt_sha256 == sha256_text(item.full_prompt)
                and item.config_sha256 == content_sha256(config)
                and item.scenario_id == (state.scenario_id if state is not None else None)
                and valid_span
                and valid_byte_span
                and item.outside_message_sha256 == recomputed_outside
                and item.message_text == expected_message
                and item.template_id
                == f"{state.topic}:{state.credibility_tier}:{item.condition}"
                if state is not None
                else False
            )
            by_hash.setdefault(item.prompt_sha256, []).append(state_id)
    duplicate_hashes = {key: sorted(set(value)) for key, value in by_hash.items() if len(set(value)) > 1}
    add("E1_PROMPT_PAIR_INTEGRITY", pair_integrity, f"pairs={len(prompt_groups)}, cross_state_duplicate_hashes={canonical_json(duplicate_hashes)}")
    add(
        "E1_PROMPT_HASH_INTEGRITY",
        prompt_hash_integrity,
        f"recomputed_prompts={len(prompts)}",
    )
    reserve_ok = len(reserve_checks) == 10 and all(item.renderable for item in reserve_checks)
    add("E1_RESERVE_RENDERABILITY", reserve_ok, f"reserve_checks={len(reserve_checks)}, all_renderable={reserve_ok}")

    expected_checklist: dict[str, tuple[str, str, str, str]] = {}
    for topic in TOPICS:
        for tier in CREDIBILITY_TIERS:
            expected_checklist[f"{topic}|{tier}"] = (
                topic,
                tier,
                str(V2_TEMPLATE_BANK[topic][tier]["authority"]["text"]),
                str(V2_TEMPLATE_BANK[topic][tier]["hedge"]["text"]),
            )
    observed_checklist: dict[str, tuple[str, str, str, str]] = {}
    required_text_columns = {
        "template_cell",
        "topic",
        "credibility_tier",
        "authority_text",
        "hedge_text",
    }
    if required_text_columns.issubset(validation_frame.columns):
        for _, row in validation_frame.iterrows():
            observed_checklist[str(row["template_cell"])] = (
                str(row["topic"]),
                str(row["credibility_tier"]),
                str(row["authority_text"]),
                str(row["hedge_text"]),
            )
    add(
        "E1_TEMPLATE_CHECKLIST_INTEGRITY",
        observed_checklist == expected_checklist and len(validation_frame) == 15,
        f"expected_cells=15, observed_cells={len(observed_checklist)}",
    )
    manual_status, manual_evidence = _manual_review_result(validation_frame)
    checks.append(
        AcceptanceCheck(
            check_id="E1_BLIND_TEMPLATE_REVIEW",
            status=manual_status,
            evidence=manual_evidence,
        )
    )
    if any(check.status == "FAIL" for check in checks):
        status = "FAIL"
    elif any(check.status == "PENDING_MANUAL_REVIEW" for check in checks):
        status = "PENDING_MANUAL_REVIEW"
    else:
        status = "PASS"
    return AcceptanceReport(
        status=status,
        checks=tuple(checks),
        config_sha256=content_sha256(config),
        state_manifest_sha256=content_sha256(states),
        prompt_corpus_sha256=content_sha256(prompts),
    )


def _checksums_for(paths: Iterable[Path], root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_bytes(path.read_bytes())
        for path in sorted(paths)
    }


def build_stage0(
    parquet_path: str | Path = "data/oracle_logs/phase1_full_v1.parquet",
    jsonl_path: str | Path = "data/oracle_logs/phase1_full_v1.jsonl",
    protocol_path: str | Path = "V2_STUDY_PLAN.md",
    output_dir: str | Path = "prereg/v2/stage0",
    config: V2FrozenConfig | None = None,
    replace_draft: bool = False,
) -> Stage0BuildResult:
    """Build all offline Stage-0 artifacts; no API client is imported."""

    frozen = config or V2FrozenConfig()
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()) and not replace_draft:
        raise FileExistsError(
            f"Refusing to overwrite existing Stage-0 directory: {destination}. "
            "Use a new directory; draft replacement is intentionally not exposed by the CLI."
        )
    destination.mkdir(parents=True, exist_ok=True)
    archive_index = load_archive_prompt_index(jsonl_path)
    selector = load_selector_view(parquet_path, set(archive_index))
    allocated = allocate_all_states(selector, frozen)
    allocated_again = allocate_all_states(selector, frozen)
    reproducible_manifest = content_sha256(
        allocated.to_dict(orient="records")
    ) == content_sha256(allocated_again.to_dict(orient="records"))
    states = state_records_from_allocation(allocated, archive_index)
    selected_keys = {
        (state.scenario_id, state.step, state.agent_id, state.query_index) for state in states
    }
    condition_texts = load_condition_texts(parquet_path, selected_keys)
    config_hash = content_sha256(frozen)
    prompts, reserve_checks = render_frozen_prompts(
        states, archive_index, condition_texts, config_hash
    )

    protocol_source = Path(protocol_path)
    if not protocol_source.is_file():
        raise FileNotFoundError(f"Protocol not found: {protocol_source}")
    shutil.copyfile(protocol_source, destination / "protocol_snapshot.md")
    write_json(destination / "frozen_config.json", frozen)
    write_jsonl(destination / "state_allocation.jsonl", list(states))
    state_frame = _state_frame(states)
    _write_csv(destination / "state_allocation.csv", state_frame)
    state_frame.to_parquet(destination / "state_allocation.parquet", index=False)
    write_jsonl(destination / "prompt_corpus.jsonl", list(prompts))
    write_jsonl(destination / "reserve_render_checks.jsonl", list(reserve_checks))
    validation_frame = _template_validation_frame()
    _write_csv(destination / "template_validation.csv", validation_frame)
    selector_audit = {
        "selector_seed": frozen.selector_seed,
        "pilot_selector_seed": frozen.selector_seed,
        "loaded_columns": list(SELECTOR_COLUMNS),
        "forbidden_columns": sorted(FORBIDDEN_SELECTOR_COLUMNS),
        "input_rows": len(selector),
        "input_scenarios": int(selector["scenario_id"].nunique()),
        "eligible_archived_prompts": len(archive_index),
        "reproducible_manifest": reproducible_manifest,
        "source_parquet_sha256": sha256_bytes(Path(parquet_path).read_bytes()),
        "source_jsonl_sha256": sha256_bytes(Path(jsonl_path).read_bytes()),
    }
    write_json(destination / "selector_audit.json", selector_audit)

    report = verify_stage0_records(
        states,
        prompts,
        reserve_checks,
        frozen,
        validation_frame,
        selector_loaded_columns=SELECTOR_COLUMNS,
        reproducible_manifest=reproducible_manifest,
    )
    write_json(destination / "stage0_acceptance.json", report)
    checksum_paths = [
        path
        for path in destination.rglob("*")
        if path.is_file() and path.name != "stage0_checksums.json"
    ]
    write_json(destination / "stage0_checksums.json", _checksums_for(checksum_paths, destination))
    return Stage0BuildResult(destination, report, states, prompts)


def _read_jsonl_models(path: Path, model: type[StateRecord] | type[PromptRecord] | type[ReserveRenderCheck]) -> tuple[Any, ...]:
    records: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            records.append(model.model_validate_json(line))
    return tuple(records)


def verify_stage0_directory(output_dir: str | Path = "prereg/v2/stage0") -> AcceptanceReport:
    """Independently re-run E1 checks from artifacts, without source data."""

    root = Path(output_dir)
    config = V2FrozenConfig.model_validate_json((root / "frozen_config.json").read_text(encoding="utf-8"))
    states = _read_jsonl_models(root / "state_allocation.jsonl", StateRecord)
    prompts = _read_jsonl_models(root / "prompt_corpus.jsonl", PromptRecord)
    reserve_checks = _read_jsonl_models(root / "reserve_render_checks.jsonl", ReserveRenderCheck)
    validation_frame = pd.read_csv(root / "template_validation.csv", keep_default_na=False)
    selector_audit = json.loads((root / "selector_audit.json").read_text(encoding="utf-8"))
    report = verify_stage0_records(
        states,
        prompts,
        reserve_checks,
        config,
        validation_frame,
        selector_loaded_columns=selector_audit["loaded_columns"],
        reproducible_manifest=bool(selector_audit["reproducible_manifest"]),
    )
    write_json(root / "stage0_acceptance.json", report)
    checksum_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != "stage0_checksums.json"
    ]
    write_json(root / "stage0_checksums.json", _checksums_for(checksum_files, root))
    return report


def build_pilot_schedule(
    stage0_dir: str | Path = "prereg/v2/stage0",
) -> tuple[TaskRecord, ...]:
    """Build the deterministic 200-slot pilot schedule from accepted Stage-0 artifacts."""

    root = Path(stage0_dir)
    acceptance = AcceptanceReport.model_validate_json(
        (root / "stage0_acceptance.json").read_text(encoding="utf-8")
    )
    if acceptance.status != "PASS":
        raise ValueError(f"Stage-0 must PASS before scheduling pilot calls; got {acceptance.status}")
    config = V2FrozenConfig.model_validate_json(
        (root / "frozen_config.json").read_text(encoding="utf-8")
    )
    states = _read_jsonl_models(root / "state_allocation.jsonl", StateRecord)
    prompts = _read_jsonl_models(root / "prompt_corpus.jsonl", PromptRecord)
    pilot_states = sorted(
        (state for state in states if state.allocation == "pilot"),
        key=lambda state: state.state_id,
    )
    if len(pilot_states) != config.pilot_states:
        raise ValueError(f"Expected {config.pilot_states} pilot states, found {len(pilot_states)}")
    prompt_by_key = {(prompt.state_id, prompt.condition): prompt for prompt in prompts}
    config_hash = content_sha256(config)
    rng = np.random.default_rng(config.pilot_schedule_seed)
    state_order = rng.permutation(len(pilot_states)).tolist()
    tasks: list[TaskRecord] = []
    schedule_index = 0
    for dispatch_group, state_position in enumerate(state_order):
        state = pilot_states[int(state_position)]
        cells = [
            (condition, replicate_index)
            for condition in ("authority", "hedge")
            for replicate_index in range(1, config.pilot_replicates_per_side + 1)
        ]
        order = rng.permutation(len(cells)).tolist()
        ordered_cells = [cells[int(index)] for index in order]
        side_sequence = [condition for condition, _ in ordered_cells]
        if side_sequence in (
            ["authority"] * 10 + ["hedge"] * 10,
            ["hedge"] * 10 + ["authority"] * 10,
        ):
            raise RuntimeError("Pilot side schedule unexpectedly failed interleaving guard")
        for condition, replicate_index in ordered_cells:
            prompt = prompt_by_key[(state.state_id, condition)]
            tasks.append(
                TaskRecord(
                    phase="pilot",
                    slot_id=f"pilot:{state.state_id}:{condition}:{replicate_index}",
                    state_id=state.state_id,
                    scenario_id=state.scenario_id,
                    condition=condition,
                    replicate_index=replicate_index,
                    schedule_index=schedule_index,
                    dispatch_group=dispatch_group,
                    prompt_sha256=prompt.prompt_sha256,
                    config_sha256=config_hash,
                    schedule_seed=config.pilot_schedule_seed,
                )
            )
            schedule_index += 1
    if len(tasks) != 200 or len({task.slot_id for task in tasks}) != 200:
        raise RuntimeError("Pilot schedule must contain exactly 200 unique slots")
    return tuple(tasks)


def write_pilot_schedule(
    output_path: str | Path,
    stage0_dir: str | Path = "prereg/v2/stage0",
) -> tuple[TaskRecord, ...]:
    """Freeze a pilot schedule once; existing files are never overwritten."""

    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite frozen pilot schedule: {target}")
    tasks = build_pilot_schedule(stage0_dir)
    write_jsonl(target, list(tasks))
    return tasks


def freeze_pilot_schedule_directory(
    stage0_commit: str,
    output_dir: str | Path = "prereg/v2/pilot",
    stage0_dir: str | Path = "prereg/v2/stage0",
) -> tuple[TaskRecord, ...]:
    """Create the post-Stage-0 pilot schedule acceptance bundle."""

    if not stage0_commit.strip():
        raise ValueError("stage0_commit is required")
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite pilot freeze directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    tasks = build_pilot_schedule(stage0_dir)
    schedule_path = destination / "pilot_schedule.jsonl"
    write_jsonl(schedule_path, list(tasks))
    stage0_root = Path(stage0_dir)
    stage0_acceptance = AcceptanceReport.model_validate_json(
        (stage0_root / "stage0_acceptance.json").read_text(encoding="utf-8")
    )
    config = V2FrozenConfig.model_validate_json(
        (stage0_root / "frozen_config.json").read_text(encoding="utf-8")
    )
    state_counts = Counter(task.state_id for task in tasks)
    side_counts = Counter(task.condition for task in tasks)
    checks = {
        "stage0_pass": stage0_acceptance.status == "PASS",
        "slot_count_200": len(tasks) == 200,
        "unique_slots_200": len({task.slot_id for task in tasks}) == 200,
        "pilot_states_10": len(state_counts) == 10,
        "twenty_slots_per_state": set(state_counts.values()) == {20},
        "side_counts_a100_h100": side_counts == {"authority": 100, "hedge": 100},
        "contiguous_schedule": [task.schedule_index for task in tasks] == list(range(200)),
        "seed_frozen": {task.schedule_seed for task in tasks} == {config.pilot_schedule_seed},
    }
    acceptance = {
        "stage": "E2_PILOT_SCHEDULE",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "stage0_commit": stage0_commit,
        "stage0_state_manifest_sha256": stage0_acceptance.state_manifest_sha256,
        "stage0_prompt_corpus_sha256": stage0_acceptance.prompt_corpus_sha256,
        "config_sha256": content_sha256(config),
        "schedule_seed": config.pilot_schedule_seed,
        "schedule_content_sha256": content_sha256(tasks),
        "schedule_file_sha256": sha256_bytes(schedule_path.read_bytes()),
    }
    write_json(destination / "pilot_schedule_acceptance.json", acceptance)
    checksum_paths = [
        destination / "pilot_schedule.jsonl",
        destination / "pilot_schedule_acceptance.json",
    ]
    write_json(
        destination / "pilot_schedule_checksums.json",
        _checksums_for(checksum_paths, destination),
    )
    if acceptance["status"] != "PASS":
        raise RuntimeError("Pilot schedule freeze failed its acceptance checks")
    return tasks


def build_confirmatory_schedule(
    selected_k: int,
    stage0_dir: str | Path = "prereg/v2/stage0",
) -> tuple[TaskRecord, ...]:
    """Build the seeded post-pilot schedule for the mechanically selected K."""

    if selected_k not in {1, 3, 5, 10}:
        raise ValueError("selected_k must be one of 1, 3, 5, 10")
    root = Path(stage0_dir)
    acceptance = AcceptanceReport.model_validate_json(
        (root / "stage0_acceptance.json").read_text(encoding="utf-8")
    )
    if acceptance.status != "PASS":
        raise ValueError("Stage-0 must PASS before confirmatory scheduling")
    config = V2FrozenConfig.model_validate_json(
        (root / "frozen_config.json").read_text(encoding="utf-8")
    )
    states = _read_jsonl_models(root / "state_allocation.jsonl", StateRecord)
    prompts = _read_jsonl_models(root / "prompt_corpus.jsonl", PromptRecord)
    confirmatory_states = sorted(
        (state for state in states if state.allocation == "confirmatory"),
        key=lambda state: state.state_id,
    )
    if len(confirmatory_states) != config.confirmatory_states:
        raise ValueError("Expected 60 frozen confirmatory states")
    prompt_by_key = {(prompt.state_id, prompt.condition): prompt for prompt in prompts}
    config_hash = content_sha256(config)
    rng = np.random.default_rng(config.confirmatory_schedule_seed)
    state_order = rng.permutation(len(confirmatory_states)).tolist()
    tasks: list[TaskRecord] = []
    for dispatch_group, state_position in enumerate(state_order):
        state = confirmatory_states[int(state_position)]
        cells = [
            (condition, replicate_index)
            for condition in ("authority", "hedge")
            for replicate_index in range(1, selected_k + 1)
        ]
        for cell_position in rng.permutation(len(cells)).tolist():
            condition, replicate_index = cells[int(cell_position)]
            prompt = prompt_by_key[(state.state_id, condition)]
            tasks.append(
                TaskRecord(
                    phase="confirmatory",
                    slot_id=f"confirmatory:{state.state_id}:{condition}:{replicate_index}",
                    state_id=state.state_id,
                    scenario_id=state.scenario_id,
                    condition=condition,
                    replicate_index=replicate_index,
                    schedule_index=len(tasks),
                    dispatch_group=dispatch_group,
                    prompt_sha256=prompt.prompt_sha256,
                    config_sha256=config_hash,
                    schedule_seed=config.confirmatory_schedule_seed,
                )
            )
    expected = config.confirmatory_states * 2 * selected_k
    if len(tasks) != expected or len({task.slot_id for task in tasks}) != expected:
        raise RuntimeError("Confirmatory schedule has an invalid slot count")
    if [task.schedule_index for task in tasks] != list(range(expected)):
        raise RuntimeError("Confirmatory schedule indexes are not contiguous")
    return tuple(tasks)


def freeze_confirmatory_schedule_directory(
    selected_k: int,
    pilot_decision_path: str | Path,
    prelive_commit: str,
    output_dir: str | Path = "prereg/v2/confirmatory",
    stage0_dir: str | Path = "prereg/v2/stage0",
) -> tuple[TaskRecord, ...]:
    """Freeze K and the confirmatory task manifest before any confirmatory call."""

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite confirmatory freeze: {destination}")
    pilot_path = Path(pilot_decision_path)
    pilot_report = json.loads(pilot_path.read_text(encoding="utf-8"))
    decision = pilot_report.get("decision", {})
    if pilot_report.get("status") != "PASS" or decision.get("selected_k") != selected_k:
        raise ValueError("selected_k does not match the passing mechanical pilot decision")
    destination.mkdir(parents=True, exist_ok=True)
    tasks = build_confirmatory_schedule(selected_k, stage0_dir)
    schedule_path = destination / "confirmatory_schedule.jsonl"
    write_jsonl(schedule_path, list(tasks))
    config = V2FrozenConfig.model_validate_json(
        (Path(stage0_dir) / "frozen_config.json").read_text(encoding="utf-8")
    )
    state_counts = Counter(task.state_id for task in tasks)
    side_counts = Counter(task.condition for task in tasks)
    expected = 60 * 2 * selected_k
    checks = {
        "pilot_pass": pilot_report.get("status") == "PASS",
        "k_matches_mechanical_decision": decision.get("selected_k") == selected_k,
        "slot_count": len(tasks) == expected,
        "unique_slots": len({task.slot_id for task in tasks}) == expected,
        "confirmatory_states_60": len(state_counts) == 60,
        "slots_per_state": set(state_counts.values()) == {2 * selected_k},
        "balanced_sides": side_counts == {"authority": 60 * selected_k, "hedge": 60 * selected_k},
        "contiguous_schedule": [task.schedule_index for task in tasks] == list(range(expected)),
        "seed_frozen": {task.schedule_seed for task in tasks} == {config.confirmatory_schedule_seed},
    }
    report = {
        "stage": "E6_CONFIRMATORY_FREEZE",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "selected_k": selected_k,
        "projected_valid_calls": expected,
        "prelive_commit": prelive_commit,
        "pilot_decision_sha256": sha256_bytes(pilot_path.read_bytes()),
        "schedule_file_sha256": sha256_bytes(schedule_path.read_bytes()),
        "schedule_content_sha256": content_sha256(tasks),
        "protocol_deviation": "Confirmatory scheduler code was added after pilot collection but before confirmatory collection; K, states, prompts, seed, and ordering rule were already frozen in Stage-0.",
    }
    write_json(destination / "confirmatory_schedule_acceptance.json", report)
    if report["status"] != "PASS":
        raise RuntimeError("Confirmatory schedule failed its acceptance checks")
    return tasks
