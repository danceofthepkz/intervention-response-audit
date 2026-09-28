"""Append-only, exact-prompt collection safety shell for v2.

This module contains no live provider adapter.  A provider must satisfy the
small ``CompletionProvider`` protocol, which lets failure behavior be tested
fully offline before any API implementation is authorized.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol

from pydantic import Field, model_validator

from intervention_response_audit.v2.schema import (
    AttemptRecord,
    PromptRecord,
    ScientificOutput,
    StrictFrozenModel,
    TaskRecord,
    V2FrozenConfig,
    canonical_json,
    content_sha256,
    sha256_text,
)


class ProviderResponse(StrictFrozenModel):
    """Provider-independent response provenance required by v2."""

    response_id: str | None
    usage_event_id: str | None
    raw_response: str
    model_snapshot: str
    config_sha256: str
    prompt_tokens: int = Field(ge=0)
    cached_prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    call_cost_usd: float = Field(ge=0.0)
    latency_ms: float = Field(default=0.0, ge=0.0)


class CompletionProvider(Protocol):
    def complete(self, prompt: str) -> ProviderResponse:
        """Return one new usage event, or raise ``ProviderTransportError``."""


class ProviderTransportError(RuntimeError):
    """A transport failure eligible for one exact-prompt retry."""


class CollectorLimits(StrictFrozenModel):
    max_attempts: int = Field(gt=0)
    max_retry_attempts: int = Field(ge=0)
    max_retry_fraction: float = Field(ge=0.0, le=1.0)
    max_cost_usd: float = Field(gt=0.0)
    preflight_cost_reserve_usd: float = Field(default=0.0, ge=0.0)


class CollectionSummary(StrictFrozenModel):
    scheduled_slots: int
    accepted_slots: int
    attempts: int
    retry_attempts: int
    cumulative_cost_usd: float
    resumed_slots: int


class CollectionHalt(RuntimeError):
    """Fail-closed collection stop with a stable protocol code."""

    def __init__(self, code: str, evidence: str) -> None:
        super().__init__(f"{code}: {evidence}")
        self.code = code
        self.evidence = evidence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_scientific_output(raw_response: str) -> ScientificOutput:
    """Parse the exact four-key v2 output schema."""

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid_json: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("response must be one JSON object")
    expected = {"believe", "reshare", "attend", "reason_code"}
    if set(payload) != expected:
        raise ValueError(f"response keys must be exactly {sorted(expected)}")
    try:
        return ScientificOutput.model_validate(payload)
    except Exception as exc:
        raise ValueError(f"invalid_output_schema: {exc}") from exc


def read_attempt_log(path: str | Path) -> tuple[AttemptRecord, ...]:
    target = Path(path)
    if not target.exists():
        return ()
    records: list[AttemptRecord] = []
    with target.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(AttemptRecord.model_validate_json(line))
            except Exception as exc:
                raise CollectionHalt(
                    "HALT_LOG_SCHEMA",
                    f"invalid attempt record at line {line_number}: {exc}",
                ) from exc
    return tuple(records)


def _append_model(path: Path, model: StrictFrozenModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(model) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_halt(
    path: Path | None,
    code: str,
    evidence: str,
    clock: Callable[[], str],
) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"code": code, "evidence": evidence, "halted_at_utc": clock()}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(payload) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _halt(
    code: str,
    evidence: str,
    halt_log_path: Path | None,
    clock: Callable[[], str],
) -> None:
    _append_halt(halt_log_path, code, evidence, clock)
    raise CollectionHalt(code, evidence)


def _attempt_id(task: TaskRecord, attempt_index: int) -> str:
    return sha256_text(
        f"{task.slot_id}|{attempt_index}|{task.prompt_sha256}|{task.config_sha256}"
    )


def _validate_existing_attempts(
    attempts: tuple[AttemptRecord, ...],
    task_by_slot: dict[str, TaskRecord],
    halt_log_path: Path | None,
    clock: Callable[[], str],
) -> None:
    attempt_ids: set[str] = set()
    response_ids: set[str] = set()
    usage_ids: set[str] = set()
    accepted_slots: set[str] = set()
    for attempt in attempts:
        task = task_by_slot.get(attempt.slot_id)
        if task is None:
            _halt("HALT_SCHEDULE_MISMATCH", f"log contains unknown slot {attempt.slot_id}", halt_log_path, clock)
        if (
            attempt.prompt_sha256 != task.prompt_sha256
            or attempt.config_sha256 != task.config_sha256
            or attempt.schedule_index != task.schedule_index
            or attempt.phase != task.phase
            or attempt.state_id != task.state_id
            or attempt.scenario_id != task.scenario_id
            or attempt.condition != task.condition
            or attempt.replicate_index != task.replicate_index
        ):
            _halt("HALT_LOG_INTEGRITY", f"logged provenance differs for {attempt.slot_id}", halt_log_path, clock)
        if attempt.attempt_id != _attempt_id(task, attempt.attempt_index):
            _halt("HALT_LOG_INTEGRITY", f"non-canonical attempt_id for {attempt.slot_id}", halt_log_path, clock)
        if attempt.attempt_id in attempt_ids:
            _halt("HALT_DUPLICATE_ID", f"duplicate attempt_id {attempt.attempt_id}", halt_log_path, clock)
        attempt_ids.add(attempt.attempt_id)
        if attempt.response_id is not None:
            if attempt.response_id in response_ids:
                _halt("HALT_RESPONSE_REUSE", f"duplicate response_id {attempt.response_id}", halt_log_path, clock)
            response_ids.add(attempt.response_id)
        if attempt.usage_event_id is not None:
            if attempt.usage_event_id in usage_ids:
                _halt("HALT_RESPONSE_REUSE", f"duplicate usage_event_id {attempt.usage_event_id}", halt_log_path, clock)
            usage_ids.add(attempt.usage_event_id)
        if attempt.accepted:
            if attempt.slot_id in accepted_slots:
                _halt("HALT_DUPLICATE_ID", f"multiple accepted attempts for {attempt.slot_id}", halt_log_path, clock)
            accepted_slots.add(attempt.slot_id)
    for slot_id in task_by_slot:
        slot_attempts = sorted(
            (attempt for attempt in attempts if attempt.slot_id == slot_id),
            key=lambda attempt: attempt.attempt_index,
        )
        indexes = [attempt.attempt_index for attempt in slot_attempts]
        if indexes not in ([], [1], [1, 2]):
            _halt("HALT_LOG_INTEGRITY", f"invalid attempt sequence {indexes} for {slot_id}", halt_log_path, clock)


def _response_attempt(
    task: TaskRecord,
    attempt_index: int,
    started_at: str,
    completed_at: str,
    response: ProviderResponse,
    status: str,
    accepted: bool,
    parsed_output: ScientificOutput | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> AttemptRecord:
    return AttemptRecord(
        attempt_id=_attempt_id(task, attempt_index),
        phase=task.phase,
        slot_id=task.slot_id,
        state_id=task.state_id,
        scenario_id=task.scenario_id,
        condition=task.condition,
        replicate_index=task.replicate_index,
        schedule_index=task.schedule_index,
        attempt_index=attempt_index,
        prompt_sha256=task.prompt_sha256,
        config_sha256=task.config_sha256,
        status=status,
        accepted=accepted,
        started_at_utc=started_at,
        completed_at_utc=completed_at,
        response_id=response.response_id,
        usage_event_id=response.usage_event_id,
        response_model_snapshot=response.model_snapshot,
        prompt_tokens=response.prompt_tokens,
        cached_prompt_tokens=response.cached_prompt_tokens,
        completion_tokens=response.completion_tokens,
        reasoning_tokens=response.reasoning_tokens,
        call_cost_usd=response.call_cost_usd,
        latency_ms=response.latency_ms,
        raw_response=response.raw_response,
        parsed_output=parsed_output,
        error_type=error_type,
        error_message=error_message,
    )


def collect_tasks(
    tasks: Iterable[TaskRecord],
    prompts: Iterable[PromptRecord],
    provider: CompletionProvider,
    config: V2FrozenConfig,
    attempt_log_path: str | Path,
    halt_log_path: str | Path | None = None,
    limits: CollectorLimits | None = None,
    clock: Callable[[], str] = utc_now,
) -> CollectionSummary:
    """Execute a frozen schedule serially with fail-closed integrity checks."""

    ordered_tasks = tuple(sorted(tasks, key=lambda task: task.schedule_index))
    if not ordered_tasks:
        raise ValueError("Collection schedule is empty")
    if [task.schedule_index for task in ordered_tasks] != list(range(len(ordered_tasks))):
        raise ValueError("schedule_index values must be contiguous from zero")
    if len({task.slot_id for task in ordered_tasks}) != len(ordered_tasks):
        raise ValueError("Schedule contains duplicate slot IDs")
    task_by_slot = {task.slot_id: task for task in ordered_tasks}
    prompt_by_key = {(prompt.state_id, prompt.condition): prompt for prompt in prompts}
    config_hash = content_sha256(config)
    log_path = Path(attempt_log_path)
    halt_path = Path(halt_log_path) if halt_log_path is not None else None
    guard = limits or CollectorLimits(
        max_attempts=len(ordered_tasks) + config.max_retry_attempts,
        max_retry_attempts=config.max_retry_attempts,
        max_retry_fraction=0.01,
        max_cost_usd=config.max_live_usd,
        preflight_cost_reserve_usd=0.001,
    )

    existing = read_attempt_log(log_path)
    _validate_existing_attempts(existing, task_by_slot, halt_path, clock)
    attempts_by_slot: dict[str, list[AttemptRecord]] = {}
    response_ids: set[str] = set()
    usage_ids: set[str] = set()
    for attempt in existing:
        attempts_by_slot.setdefault(attempt.slot_id, []).append(attempt)
        if attempt.response_id is not None:
            response_ids.add(attempt.response_id)
        if attempt.usage_event_id is not None:
            usage_ids.add(attempt.usage_event_id)
    total_attempts = len(existing)
    retry_attempts = sum(attempt.attempt_index == 2 for attempt in existing)
    cumulative_cost = sum(attempt.call_cost_usd for attempt in existing)
    accepted_slots = {attempt.slot_id for attempt in existing if attempt.accepted}
    resumed_slots = len(accepted_slots)
    if total_attempts > guard.max_attempts:
        _halt("HALT_CALL_LIMIT", f"existing log exceeds attempt cap {guard.max_attempts}", halt_path, clock)
    if retry_attempts > guard.max_retry_attempts:
        _halt("HALT_RETRY_LIMIT", f"existing log exceeds retry cap {guard.max_retry_attempts}", halt_path, clock)
    if retry_attempts / len(ordered_tasks) > guard.max_retry_fraction:
        _halt("HALT_RETRY_RATE", "existing log exceeds retry fraction", halt_path, clock)
    if cumulative_cost > guard.max_cost_usd:
        _halt("HALT_BUDGET", f"existing log cost {cumulative_cost:.8f} exceeds cap", halt_path, clock)

    for task in ordered_tasks:
        if task.slot_id in accepted_slots:
            continue
        if task.config_sha256 != config_hash:
            _halt("HALT_CONFIG_MISMATCH", f"task config hash differs for {task.slot_id}", halt_path, clock)
        prompt = prompt_by_key.get((task.state_id, task.condition))
        if prompt is None:
            _halt("HALT_PROMPT_HASH_MISMATCH", f"missing prompt for {task.slot_id}", halt_path, clock)
        if (
            prompt.prompt_sha256 != task.prompt_sha256
            or sha256_text(prompt.full_prompt) != task.prompt_sha256
            or prompt.config_sha256 != task.config_sha256
        ):
            _halt("HALT_PROMPT_HASH_MISMATCH", f"frozen prompt mismatch for {task.slot_id}", halt_path, clock)

        prior = sorted(attempts_by_slot.get(task.slot_id, []), key=lambda item: item.attempt_index)
        if len(prior) > 1:
            _halt("HALT_SLOT_INCOMPLETE", f"slot already has two failed attempts: {task.slot_id}", halt_path, clock)
        if prior and prior[0].status not in {"parse_error", "transport_error"}:
            _halt("HALT_SLOT_INCOMPLETE", f"slot has non-retryable prior failure: {task.slot_id}", halt_path, clock)
        attempt_index = 2 if prior else 1

        while attempt_index <= 2:
            if total_attempts >= guard.max_attempts:
                _halt("HALT_CALL_LIMIT", f"attempt cap {guard.max_attempts} reached", halt_path, clock)
            if attempt_index == 2:
                projected_retries = retry_attempts + 1
                if projected_retries > guard.max_retry_attempts:
                    _halt("HALT_RETRY_LIMIT", f"retry cap {guard.max_retry_attempts} reached", halt_path, clock)
                if projected_retries / len(ordered_tasks) > guard.max_retry_fraction:
                    _halt(
                        "HALT_RETRY_RATE",
                        f"projected retry fraction {projected_retries / len(ordered_tasks):.6f} exceeds {guard.max_retry_fraction}",
                        halt_path,
                        clock,
                    )
                retry_attempts = projected_retries

            if cumulative_cost + guard.preflight_cost_reserve_usd > guard.max_cost_usd:
                _halt(
                    "HALT_BUDGET",
                    "preflight cost reserve would exceed the live-spend cap",
                    halt_path,
                    clock,
                )

            started_at = clock()
            try:
                response = provider.complete(prompt.full_prompt)
            except ProviderTransportError as exc:
                completed_at = clock()
                attempt = AttemptRecord(
                    attempt_id=_attempt_id(task, attempt_index),
                    phase=task.phase,
                    slot_id=task.slot_id,
                    state_id=task.state_id,
                    scenario_id=task.scenario_id,
                    condition=task.condition,
                    replicate_index=task.replicate_index,
                    schedule_index=task.schedule_index,
                    attempt_index=attempt_index,
                    prompt_sha256=task.prompt_sha256,
                    config_sha256=task.config_sha256,
                    status="transport_error",
                    accepted=False,
                    started_at_utc=started_at,
                    completed_at_utc=completed_at,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                _append_model(log_path, attempt)
                total_attempts += 1
                attempts_by_slot.setdefault(task.slot_id, []).append(attempt)
                if attempt_index == 2:
                    _halt("HALT_SLOT_INCOMPLETE", f"two failed attempts for {task.slot_id}", halt_path, clock)
                attempt_index = 2
                continue
            except Exception as exc:
                completed_at = clock()
                attempt = AttemptRecord(
                    attempt_id=_attempt_id(task, attempt_index),
                    phase=task.phase,
                    slot_id=task.slot_id,
                    state_id=task.state_id,
                    scenario_id=task.scenario_id,
                    condition=task.condition,
                    replicate_index=task.replicate_index,
                    schedule_index=task.schedule_index,
                    attempt_index=attempt_index,
                    prompt_sha256=task.prompt_sha256,
                    config_sha256=task.config_sha256,
                    status="transport_error",
                    accepted=False,
                    started_at_utc=started_at,
                    completed_at_utc=completed_at,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                _append_model(log_path, attempt)
                total_attempts += 1
                _halt(
                    "HALT_PROVIDER_UNEXPECTED",
                    f"unexpected provider exception for {task.slot_id}: {type(exc).__name__}",
                    halt_path,
                    clock,
                )

            completed_at = clock()
            total_attempts += 1
            cumulative_cost += response.call_cost_usd
            integrity_code: str | None = None
            integrity_message: str | None = None
            if not response.response_id:
                integrity_code, integrity_message = "HALT_RESPONSE_ID_MISSING", f"missing response ID for {task.slot_id}"
            elif response.response_id in response_ids:
                integrity_code, integrity_message = "HALT_RESPONSE_REUSE", f"duplicate response ID {response.response_id}"
            elif not response.usage_event_id:
                integrity_code, integrity_message = "HALT_USAGE_EVENT_MISSING", f"missing usage event for {task.slot_id}"
            elif response.usage_event_id in usage_ids:
                integrity_code, integrity_message = "HALT_RESPONSE_REUSE", f"duplicate usage event {response.usage_event_id}"
            elif response.model_snapshot != config.model_snapshot or response.config_sha256 != config_hash:
                integrity_code, integrity_message = "HALT_CONFIG_MISMATCH", f"provider configuration differs for {task.slot_id}"
            elif cumulative_cost > guard.max_cost_usd:
                integrity_code, integrity_message = "HALT_BUDGET", f"cumulative cost {cumulative_cost:.8f} exceeds {guard.max_cost_usd:.8f}"

            if response.response_id:
                response_ids.add(response.response_id)
            if response.usage_event_id:
                usage_ids.add(response.usage_event_id)
            if integrity_code is not None:
                status = "budget_exceeded" if integrity_code == "HALT_BUDGET" else "integrity_error"
                attempt = _response_attempt(
                    task,
                    attempt_index,
                    started_at,
                    completed_at,
                    response,
                    status,
                    False,
                    error_type=integrity_code,
                    error_message=integrity_message,
                )
                _append_model(log_path, attempt)
                _halt(integrity_code, str(integrity_message), halt_path, clock)

            try:
                parsed = parse_scientific_output(response.raw_response)
            except ValueError as exc:
                attempt = _response_attempt(
                    task,
                    attempt_index,
                    started_at,
                    completed_at,
                    response,
                    "parse_error",
                    False,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
                _append_model(log_path, attempt)
                attempts_by_slot.setdefault(task.slot_id, []).append(attempt)
                if attempt_index == 2:
                    _halt("HALT_SLOT_INCOMPLETE", f"two failed attempts for {task.slot_id}", halt_path, clock)
                attempt_index = 2
                continue

            attempt = _response_attempt(
                task,
                attempt_index,
                started_at,
                completed_at,
                response,
                "valid",
                True,
                parsed_output=parsed,
            )
            _append_model(log_path, attempt)
            attempts_by_slot.setdefault(task.slot_id, []).append(attempt)
            accepted_slots.add(task.slot_id)
            break

    return CollectionSummary(
        scheduled_slots=len(ordered_tasks),
        accepted_slots=len(accepted_slots),
        attempts=total_attempts,
        retry_attempts=retry_attempts,
        cumulative_cost_usd=cumulative_cost,
        resumed_slots=resumed_slots,
    )
