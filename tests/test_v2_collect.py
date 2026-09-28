import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from intervention_response_audit.v2.collect import (
    CollectionHalt,
    CollectorLimits,
    ProviderResponse,
    ProviderTransportError,
    collect_tasks,
    parse_scientific_output,
    read_attempt_log,
)
from intervention_response_audit.v2.design import build_pilot_schedule, freeze_pilot_schedule_directory
from intervention_response_audit.v2.schema import PromptRecord, V2FrozenConfig, content_sha256


STAGE0 = Path("prereg/v2/stage0")
VALID_RAW = json.dumps(
    {"believe": 0.4, "reshare": 0.2, "attend": 0.5, "reason_code": "mock"}
)


class ScriptedProvider:
    def __init__(self, script):
        self.script = list(script)
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> ProviderResponse:
        self.prompts.append(prompt)
        if not self.script:
            raise AssertionError("Provider received an unscheduled extra call")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def response(
    index: int,
    config: V2FrozenConfig,
    *,
    raw: str = VALID_RAW,
    response_id: str | None = None,
    usage_event_id: str | None = None,
    config_sha256: str | None = None,
    model_snapshot: str | None = None,
    cost: float = 0.001,
) -> ProviderResponse:
    return ProviderResponse(
        response_id=response_id if response_id is not None else f"response-{index}",
        usage_event_id=usage_event_id if usage_event_id is not None else f"usage-{index}",
        raw_response=raw,
        model_snapshot=model_snapshot or config.model_snapshot,
        config_sha256=config_sha256 or content_sha256(config),
        prompt_tokens=100,
        cached_prompt_tokens=0,
        completion_tokens=20,
        reasoning_tokens=0,
        call_cost_usd=cost,
    )


def load_prompts() -> tuple[PromptRecord, ...]:
    return tuple(
        PromptRecord.model_validate_json(line)
        for line in (STAGE0 / "prompt_corpus.jsonl").read_text(encoding="utf-8").splitlines()
    )


def permissive_limits(task_count: int) -> CollectorLimits:
    return CollectorLimits(
        max_attempts=task_count + 20,
        max_retry_attempts=20,
        max_retry_fraction=1.0,
        max_cost_usd=10.0,
    )


def test_pilot_schedule_is_frozen_balanced_and_deterministic() -> None:
    left = build_pilot_schedule(STAGE0)
    right = build_pilot_schedule(STAGE0)
    assert left == right
    assert len(left) == len({task.slot_id for task in left}) == 200
    assert [task.schedule_index for task in left] == list(range(200))
    assert {task.schedule_seed for task in left} == {20260722}
    by_state = defaultdict(list)
    for task in left:
        by_state[task.state_id].append(task)
    assert len(by_state) == 10
    for tasks in by_state.values():
        assert Counter(task.condition for task in tasks) == {"authority": 10, "hedge": 10}
        assert {
            task.replicate_index for task in tasks if task.condition == "authority"
        } == set(range(1, 11))
        assert {
            task.replicate_index for task in tasks if task.condition == "hedge"
        } == set(range(1, 11))


def test_pilot_schedule_freeze_bundle_passes_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "pilot"
    tasks = freeze_pilot_schedule_directory("stage0-test-commit", output, STAGE0)
    acceptance = json.loads((output / "pilot_schedule_acceptance.json").read_text())
    assert len(tasks) == 200
    assert acceptance["status"] == "PASS"
    assert acceptance["stage0_commit"] == "stage0-test-commit"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        freeze_pilot_schedule_directory("stage0-test-commit", output, STAGE0)


def test_parser_rejects_extra_keys_and_out_of_range_probabilities() -> None:
    assert parse_scientific_output(VALID_RAW).reshare == pytest.approx(0.2)
    with pytest.raises(ValueError):
        parse_scientific_output('{"believe":0.1,"reshare":0.2,"attend":0.3,"reason_code":"x","extra":1}')
    with pytest.raises(ValueError):
        parse_scientific_output('{"believe":1.1,"reshare":0.2,"attend":0.3,"reason_code":"x"}')


def test_successful_collection_is_append_only_and_resume_safe(tmp_path: Path) -> None:
    config = V2FrozenConfig()
    tasks = build_pilot_schedule(STAGE0)[:2]
    provider = ScriptedProvider([response(1, config), response(2, config)])
    log = tmp_path / "attempts.jsonl"
    summary = collect_tasks(
        tasks,
        load_prompts(),
        provider,
        config,
        log,
        limits=permissive_limits(len(tasks)),
    )
    assert summary.accepted_slots == summary.attempts == 2
    before = log.read_bytes()

    resumed_provider = ScriptedProvider([])
    resumed = collect_tasks(
        tasks,
        load_prompts(),
        resumed_provider,
        config,
        log,
        limits=permissive_limits(len(tasks)),
    )
    assert resumed.resumed_slots == 2
    assert resumed_provider.prompts == []
    assert log.read_bytes() == before


def test_parse_failure_retries_once_with_identical_prompt(tmp_path: Path) -> None:
    config = V2FrozenConfig()
    tasks = build_pilot_schedule(STAGE0)[:1]
    provider = ScriptedProvider(
        [response(1, config, raw="not-json"), response(2, config)]
    )
    log = tmp_path / "attempts.jsonl"
    summary = collect_tasks(
        tasks,
        load_prompts(),
        provider,
        config,
        log,
        limits=permissive_limits(1),
    )
    attempts = read_attempt_log(log)
    assert [attempt.status for attempt in attempts] == ["parse_error", "valid"]
    assert [attempt.attempt_index for attempt in attempts] == [1, 2]
    assert provider.prompts[0] == provider.prompts[1]
    assert summary.retry_attempts == 1


def test_transport_failure_retries_once_with_identical_prompt(tmp_path: Path) -> None:
    config = V2FrozenConfig()
    tasks = build_pilot_schedule(STAGE0)[:1]
    provider = ScriptedProvider([ProviderTransportError("timeout"), response(2, config)])
    log = tmp_path / "attempts.jsonl"
    collect_tasks(
        tasks,
        load_prompts(),
        provider,
        config,
        log,
        limits=permissive_limits(1),
    )
    assert [item.status for item in read_attempt_log(log)] == ["transport_error", "valid"]
    assert provider.prompts[0] == provider.prompts[1]


def test_resume_after_first_failure_runs_only_the_frozen_retry(tmp_path: Path) -> None:
    config = V2FrozenConfig()
    tasks = build_pilot_schedule(STAGE0)[:1]
    log = tmp_path / "attempts.jsonl"
    with pytest.raises(CollectionHalt, match="HALT_RETRY_RATE"):
        collect_tasks(
            tasks,
            load_prompts(),
            ScriptedProvider([response(1, config, raw="bad")]),
            config,
            log,
            limits=CollectorLimits(
                max_attempts=2,
                max_retry_attempts=1,
                max_retry_fraction=0.0,
                max_cost_usd=10.0,
            ),
        )
    provider = ScriptedProvider([response(2, config)])
    summary = collect_tasks(
        tasks,
        load_prompts(),
        provider,
        config,
        log,
        limits=permissive_limits(1),
    )
    assert summary.accepted_slots == 1
    assert [item.attempt_index for item in read_attempt_log(log)] == [1, 2]
    assert len(provider.prompts) == 1


def test_second_invalid_response_halts_instead_of_dropping_slot(tmp_path: Path) -> None:
    config = V2FrozenConfig()
    tasks = build_pilot_schedule(STAGE0)[:1]
    provider = ScriptedProvider(
        [response(1, config, raw="bad"), response(2, config, raw="also bad")]
    )
    with pytest.raises(CollectionHalt, match="HALT_SLOT_INCOMPLETE") as caught:
        collect_tasks(
            tasks,
            load_prompts(),
            provider,
            config,
            tmp_path / "attempts.jsonl",
            tmp_path / "halts.jsonl",
            permissive_limits(1),
        )
    assert caught.value.code == "HALT_SLOT_INCOMPLETE"
    assert len(read_attempt_log(tmp_path / "attempts.jsonl")) == 2


def test_duplicate_response_id_halts_collection(tmp_path: Path) -> None:
    config = V2FrozenConfig()
    tasks = build_pilot_schedule(STAGE0)[:2]
    provider = ScriptedProvider(
        [
            response(1, config, response_id="same-response"),
            response(2, config, response_id="same-response"),
        ]
    )
    with pytest.raises(CollectionHalt, match="HALT_RESPONSE_REUSE"):
        collect_tasks(
            tasks,
            load_prompts(),
            provider,
            config,
            tmp_path / "attempts.jsonl",
            tmp_path / "halts.jsonl",
            permissive_limits(2),
        )


def test_missing_usage_event_and_config_drift_each_halt(tmp_path: Path) -> None:
    config = V2FrozenConfig()
    tasks = build_pilot_schedule(STAGE0)[:1]
    missing_usage = response(1, config).model_copy(update={"usage_event_id": None})
    with pytest.raises(CollectionHalt, match="HALT_USAGE_EVENT_MISSING"):
        collect_tasks(
            tasks,
            load_prompts(),
            ScriptedProvider([missing_usage]),
            config,
            tmp_path / "usage-attempts.jsonl",
            limits=permissive_limits(1),
        )

    with pytest.raises(CollectionHalt, match="HALT_CONFIG_MISMATCH"):
        collect_tasks(
            tasks,
            load_prompts(),
            ScriptedProvider([response(2, config, config_sha256="wrong")]),
            config,
            tmp_path / "config-attempts.jsonl",
            limits=permissive_limits(1),
        )


def test_prompt_tampering_halts_before_provider_call(tmp_path: Path) -> None:
    config = V2FrozenConfig()
    tasks = build_pilot_schedule(STAGE0)[:1]
    prompts = list(load_prompts())
    target = next(
        prompt
        for prompt in prompts
        if (prompt.state_id, prompt.condition) == (tasks[0].state_id, tasks[0].condition)
    )
    prompts[prompts.index(target)] = target.model_copy(update={"full_prompt": target.full_prompt + " tampered"})
    provider = ScriptedProvider([])
    with pytest.raises(CollectionHalt, match="HALT_PROMPT_HASH_MISMATCH"):
        collect_tasks(
            tasks,
            prompts,
            provider,
            config,
            tmp_path / "attempts.jsonl",
            limits=permissive_limits(1),
        )
    assert provider.prompts == []


def test_unexpected_provider_error_is_logged_and_halts(tmp_path: Path) -> None:
    config = V2FrozenConfig()
    tasks = build_pilot_schedule(STAGE0)[:1]
    log = tmp_path / "attempts.jsonl"
    with pytest.raises(CollectionHalt, match="HALT_PROVIDER_UNEXPECTED"):
        collect_tasks(
            tasks,
            load_prompts(),
            ScriptedProvider([RuntimeError("unexpected")]),
            config,
            log,
            limits=permissive_limits(1),
        )
    attempts = read_attempt_log(log)
    assert len(attempts) == 1
    assert attempts[0].error_type == "RuntimeError"


def test_retry_rate_and_budget_guards_fail_closed(tmp_path: Path) -> None:
    config = V2FrozenConfig()
    tasks = build_pilot_schedule(STAGE0)[:1]
    with pytest.raises(CollectionHalt, match="HALT_RETRY_RATE"):
        collect_tasks(
            tasks,
            load_prompts(),
            ScriptedProvider([response(1, config, raw="bad")]),
            config,
            tmp_path / "retry-attempts.jsonl",
            limits=CollectorLimits(
                max_attempts=2,
                max_retry_attempts=1,
                max_retry_fraction=0.0,
                max_cost_usd=10.0,
            ),
        )

    with pytest.raises(CollectionHalt, match="HALT_BUDGET"):
        collect_tasks(
            tasks,
            load_prompts(),
            ScriptedProvider([response(2, config, cost=1.0)]),
            config,
            tmp_path / "budget-attempts.jsonl",
            limits=CollectorLimits(
                max_attempts=1,
                max_retry_attempts=0,
                max_retry_fraction=0.0,
                max_cost_usd=0.5,
            ),
        )

    preflight_provider = ScriptedProvider([])
    with pytest.raises(CollectionHalt, match="HALT_BUDGET"):
        collect_tasks(
            tasks,
            load_prompts(),
            preflight_provider,
            config,
            tmp_path / "preflight-budget-attempts.jsonl",
            limits=CollectorLimits(
                max_attempts=1,
                max_retry_attempts=0,
                max_retry_fraction=0.0,
                max_cost_usd=0.5,
                preflight_cost_reserve_usd=0.6,
            ),
        )
    assert preflight_provider.prompts == []


def test_call_cap_stops_before_the_next_provider_request(tmp_path: Path) -> None:
    config = V2FrozenConfig()
    tasks = build_pilot_schedule(STAGE0)[:2]
    provider = ScriptedProvider([response(1, config)])
    log = tmp_path / "attempts.jsonl"
    with pytest.raises(CollectionHalt, match="HALT_CALL_LIMIT"):
        collect_tasks(
            tasks,
            load_prompts(),
            provider,
            config,
            log,
            limits=CollectorLimits(
                max_attempts=1,
                max_retry_attempts=0,
                max_retry_fraction=0.0,
                max_cost_usd=10.0,
            ),
        )
    assert len(provider.prompts) == 1
    assert len(read_attempt_log(log)) == 1


def test_resume_rejects_tampered_attempt_identity(tmp_path: Path) -> None:
    config = V2FrozenConfig()
    tasks = build_pilot_schedule(STAGE0)[:1]
    log = tmp_path / "attempts.jsonl"
    collect_tasks(
        tasks,
        load_prompts(),
        ScriptedProvider([response(1, config)]),
        config,
        log,
        limits=permissive_limits(1),
    )
    row = json.loads(log.read_text(encoding="utf-8"))
    row["attempt_id"] = "tampered-attempt-id"
    log.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(CollectionHalt, match="HALT_LOG_INTEGRITY"):
        collect_tasks(
            tasks,
            load_prompts(),
            ScriptedProvider([]),
            config,
            log,
            limits=permissive_limits(1),
        )
