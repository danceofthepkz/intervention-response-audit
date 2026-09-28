"""Strict schemas and deterministic serialization for the v2 freeze builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intervention_response_audit.v2.templates import V2_TEMPLATE_REVISION


AllocationRole = Literal["confirmatory", "pilot", "reserve"]
ConditionSide = Literal["authority", "hedge"]
CredibilityTier = Literal["low", "mid", "high"]
CountBin = Literal["1", "2", "3+"]
AcceptanceStatus = Literal["PASS", "FAIL", "PENDING_MANUAL_REVIEW"]
CollectionPhase = Literal["smoke", "pilot", "confirmatory"]
AttemptStatus = Literal[
    "valid",
    "parse_error",
    "transport_error",
    "integrity_error",
    "budget_exceeded",
]
PilotDecisionCode = Literal[
    "K_1_DETERMINISTIC",
    "K_3",
    "K_5",
    "K_10",
    "HALT_K_INSUFFICIENT",
]
DirectionalLabel = Literal[
    "directional_replication",
    "sign_reversal",
    "direction_unresolved",
]
PracticalLabel = Literal[
    "operationally_meaningful_positive",
    "operationally_meaningful_negative",
    "practical_equivalence",
    "practical_magnitude_unresolved",
]


class StrictFrozenModel(BaseModel):
    """Base model that rejects schema drift and mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class V2FrozenConfig(StrictFrozenModel):
    """All scientific constants needed by the Stage-0 compiler."""

    # Kept for byte-level compatibility with the frozen study artifacts.
    study_id: str = "sociopt_v2_authority_hedge"
    protocol_version: str = "2026-07-16"
    template_revision: str = V2_TEMPLATE_REVISION
    model_snapshot: str = "gpt-5.4-mini-2026-03-17"
    reasoning_effort: str = "none"
    temperature: float = 0.0
    max_output_tokens: int = 300
    response_format: str = "json_object"
    scaffold_id: int = 0
    primary_output: str = "reshare"
    secondary_output: str = "believe"
    selector_seed: int = 20260716
    confirmatory_schedule_seed: int = 20260717
    primary_bootstrap_seed: int = 20260718
    permutation_seed: int = 20260719
    crossfit_seed: int = 20260720
    template_bootstrap_seed: int = 20260721
    pilot_schedule_seed: int = 20260722
    practical_delta: float = 0.03
    k_candidates: tuple[int, ...] = (3, 5, 10)
    pilot_replicates_per_side: int = 10
    confirmatory_states: int = 60
    pilot_states: int = 10
    reserve_states: int = 10
    min_states_per_template_cell: int = 2
    topic_targets: dict[str, int] = Field(
        default_factory=lambda: {
            "club_event": 12,
            "course_material": 12,
            "gossip": 12,
            "internship": 12,
            "policy_change": 12,
        }
    )
    credibility_targets: dict[str, int] = Field(
        default_factory=lambda: {"low": 23, "mid": 16, "high": 21}
    )
    exposure_targets: dict[str, int] = Field(
        default_factory=lambda: {"1": 32, "2": 20, "3+": 8}
    )
    step_targets: dict[str, int] = Field(
        default_factory=lambda: {"1": 15, "2": 16, "3+": 29}
    )
    max_scientific_calls: int = 1400
    max_retry_attempts: int = 14
    max_smoke_attempts: int = 20
    max_live_usd: float = 5.0

    @field_validator("k_candidates")
    @classmethod
    def _validate_k_candidates(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value != (3, 5, 10):
            raise ValueError("The frozen K candidates must be exactly (3, 5, 10)")
        return value


class StateRecord(StrictFrozenModel):
    """One scenario-unique v2 state, with no oracle outcome fields."""

    state_id: str
    scenario_id: str
    agent_id: int
    step: int
    query_index: int
    allocation: AllocationRole
    topic: str
    credibility_tier: CredibilityTier
    exposure_bin: CountBin
    step_bin: CountBin
    template_cell: str
    structured_features: dict[str, int | float | str]
    source_row_sha256: str
    archived_full_prompt_sha256: str
    source_jsonl_line: int


class PromptRecord(StrictFrozenModel):
    """One frozen A/H prompt for a pilot or confirmatory state."""

    state_id: str
    scenario_id: str
    condition: ConditionSide
    template_id: str
    full_prompt: str
    prompt_sha256: str
    outside_message_sha256: str
    message_text: str
    message_start: int
    message_end: int
    message_start_byte: int
    message_end_byte: int
    config_sha256: str


class ReserveRenderCheck(StrictFrozenModel):
    """Proof that a reserve state can be rendered, without freezing call prompts."""

    state_id: str
    scenario_id: str
    authority_prompt_sha256: str
    hedge_prompt_sha256: str
    outside_message_sha256: str
    renderable: bool


class TaskRecord(StrictFrozenModel):
    """One immutable scheduled replicate slot."""

    phase: CollectionPhase
    slot_id: str
    state_id: str
    scenario_id: str
    condition: ConditionSide
    replicate_index: int = Field(ge=1)
    schedule_index: int = Field(ge=0)
    dispatch_group: int = Field(ge=0)
    prompt_sha256: str
    config_sha256: str
    schedule_seed: int

    @model_validator(mode="after")
    def _canonical_slot(self) -> "TaskRecord":
        expected = f"{self.phase}:{self.state_id}:{self.condition}:{self.replicate_index}"
        if self.slot_id != expected:
            raise ValueError(f"Non-canonical slot_id: expected {expected}")
        return self


class ScientificOutput(StrictFrozenModel):
    """Strict four-field transition-rule output."""

    believe: float = Field(ge=0.0, le=1.0)
    reshare: float = Field(ge=0.0, le=1.0)
    attend: float = Field(ge=0.0, le=1.0)
    reason_code: str = Field(min_length=1, max_length=200)


class AttemptRecord(StrictFrozenModel):
    """Append-only evidence for one provider attempt."""

    attempt_id: str
    phase: CollectionPhase
    slot_id: str
    state_id: str
    scenario_id: str
    condition: ConditionSide
    replicate_index: int = Field(ge=1)
    schedule_index: int = Field(ge=0)
    attempt_index: int = Field(ge=1, le=2)
    prompt_sha256: str
    config_sha256: str
    status: AttemptStatus
    accepted: bool
    started_at_utc: str
    completed_at_utc: str
    response_id: str | None = None
    usage_event_id: str | None = None
    response_model_snapshot: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    cached_prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    call_cost_usd: float = Field(default=0.0, ge=0.0)
    latency_ms: float | None = Field(default=None, ge=0.0)
    raw_response: str | None = None
    parsed_output: ScientificOutput | None = None
    error_type: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def _accepted_attempt_is_complete(self) -> "AttemptRecord":
        if self.accepted:
            required = (
                self.status == "valid",
                self.response_id is not None,
                self.usage_event_id is not None,
                self.response_model_snapshot is not None,
                self.parsed_output is not None,
            )
            if not all(required):
                raise ValueError("Accepted attempt lacks valid response provenance")
        if self.status == "valid" and not self.accepted:
            raise ValueError("A valid attempt must be accepted")
        return self


class PilotDecision(StrictFrozenModel):
    """Mechanical post-pilot replicate-count decision."""

    decision_code: PilotDecisionCode
    selected_k: int | None
    deterministic: bool
    valid_slots: int
    state_count: int
    cell_count: int
    q90_v: float
    se_call_by_k: dict[str, float]
    cell_variance_by_key: dict[str, float]
    criterion: float
    near_deterministic: bool

    @model_validator(mode="after")
    def _decision_consistency(self) -> "PilotDecision":
        expected = {
            "K_1_DETERMINISTIC": 1,
            "K_3": 3,
            "K_5": 5,
            "K_10": 10,
            "HALT_K_INSUFFICIENT": None,
        }[self.decision_code]
        if self.selected_k != expected:
            raise ValueError("selected_k does not match decision_code")
        return self


class ConfirmatoryResult(StrictFrozenModel):
    """Frozen primary reshare result and mechanical interpretation labels."""

    selected_k: int
    deterministic: bool
    state_count: int
    estimate: float
    fixed_template_ci: tuple[float, float]
    template_cluster_ci: tuple[float, float]
    directional_label: DirectionalLabel
    practical_label: PracticalLabel
    template_sensitive: bool
    permutation_p_value: float | None
    primary_bootstrap_resamples: int
    template_bootstrap_resamples: int
    permutation_resamples: int
    state_contrasts: dict[str, float]


class AcceptanceCheck(StrictFrozenModel):
    """One machine- or human-verifiable gate condition."""

    check_id: str
    status: AcceptanceStatus
    evidence: str


class AcceptanceReport(StrictFrozenModel):
    """Stage acceptance artifact.  Stage-0 remains pending until blind review."""

    stage: str = "E1_STAGE0"
    status: AcceptanceStatus
    checks: tuple[AcceptanceCheck, ...]
    config_sha256: str
    state_manifest_sha256: str
    prompt_corpus_sha256: str


def canonical_json(value: Any) -> str:
    """Return stable compact JSON for hashing and JSONL records."""

    def jsonable(item: Any) -> Any:
        if isinstance(item, BaseModel):
            return jsonable(item.model_dump(mode="json"))
        if isinstance(item, dict):
            return {str(key): jsonable(subvalue) for key, subvalue in item.items()}
        if isinstance(item, (list, tuple)):
            return [jsonable(subvalue) for subvalue in item]
        if isinstance(item, Path):
            return str(item)
        return item

    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def content_sha256(value: Any) -> str:
    return sha256_text(canonical_json(value))


def write_json(path: str | Path, value: Any) -> None:
    """Write deterministic, human-readable JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    target.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, values: list[BaseModel] | tuple[BaseModel, ...]) -> None:
    """Write deterministic JSONL without timestamps or unstable metadata."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(canonical_json(value) + "\n" for value in values)
    target.write_text(text, encoding="utf-8")
