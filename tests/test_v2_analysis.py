import json
import math

import pytest

from intervention_response_audit.v2.analysis import analyze_confirmatory_reshare, analyze_pilot_k
from intervention_response_audit.v2.design import build_pilot_schedule
from intervention_response_audit.v2.schema import AttemptRecord, ScientificOutput, StateRecord, V2FrozenConfig


def pilot_attempts(target_v: float) -> tuple[AttemptRecord, ...]:
    config = V2FrozenConfig()
    tasks = build_pilot_schedule("prereg/v2/stage0")
    amplitude = math.sqrt(target_v * 9.0 / 20.0) if target_v else 0.0
    attempts = []
    for index, task in enumerate(tasks):
        value = 0.5 + (amplitude if task.replicate_index % 2 else -amplitude)
        output = ScientificOutput(
            believe=0.4,
            reshare=value,
            attend=0.5,
            reason_code="synthetic",
        )
        attempts.append(
            AttemptRecord(
                attempt_id=f"attempt-{index}",
                phase="pilot",
                slot_id=task.slot_id,
                state_id=task.state_id,
                scenario_id=task.scenario_id,
                condition=task.condition,
                replicate_index=task.replicate_index,
                schedule_index=task.schedule_index,
                attempt_index=1,
                prompt_sha256=task.prompt_sha256,
                config_sha256=task.config_sha256,
                status="valid",
                accepted=True,
                started_at_utc="2026-07-17T00:00:00Z",
                completed_at_utc="2026-07-17T00:00:01Z",
                response_id=f"response-{index}",
                usage_event_id=f"usage-{index}",
                response_model_snapshot=config.model_snapshot,
                prompt_tokens=100,
                cached_prompt_tokens=0,
                completion_tokens=20,
                reasoning_tokens=0,
                call_cost_usd=0.001,
                raw_response=json.dumps(output.model_dump()),
                parsed_output=output,
            )
        )
    return tuple(attempts)


@pytest.mark.parametrize(
    ("target_v", "decision_code", "selected_k"),
    [
        (0.0, "K_1_DETERMINISTIC", 1),
        (0.005, "K_3", 3),
        (0.012, "K_5", 5),
        (0.020, "K_10", 10),
        (0.040, "HALT_K_INSUFFICIENT", None),
    ],
)
def test_pilot_k_branches_are_mechanical(
    target_v: float, decision_code: str, selected_k: int | None
) -> None:
    decision = analyze_pilot_k(pilot_attempts(target_v))
    assert decision.decision_code == decision_code
    assert decision.selected_k == selected_k
    assert decision.q90_v == pytest.approx(target_v)
    assert decision.criterion == pytest.approx(0.0075)


def test_pilot_k_uses_higher_q90_convention() -> None:
    attempts = list(pilot_attempts(0.005))
    # Raise both cell variances for one state; with 10 states, the frozen
    # "higher" Q90 selects the maximum state-level v.
    target_state = attempts[0].state_id
    high_amplitude = math.sqrt(0.02 * 9.0 / 20.0)
    for index, attempt in enumerate(attempts):
        if attempt.state_id != target_state:
            continue
        value = 0.5 + (high_amplitude if attempt.replicate_index % 2 else -high_amplitude)
        output = attempt.parsed_output.model_copy(update={"reshare": value})
        attempts[index] = attempt.model_copy(update={"parsed_output": output})
    decision = analyze_pilot_k(attempts)
    assert decision.q90_v == pytest.approx(0.02)
    assert decision.decision_code == "K_10"


def test_pilot_k_rejects_missing_slots_and_reused_response_ids() -> None:
    attempts = list(pilot_attempts(0.005))
    with pytest.raises(ValueError, match="exactly 200"):
        analyze_pilot_k(attempts[:-1])
    attempts[1] = attempts[1].model_copy(update={"response_id": attempts[0].response_id})
    with pytest.raises(ValueError, match="response IDs"):
        analyze_pilot_k(attempts)


def confirmatory_states() -> tuple[StateRecord, ...]:
    return tuple(
        StateRecord.model_validate_json(line)
        for line in open("prereg/v2/stage0/state_allocation.jsonl", encoding="utf-8")
    )


def confirmatory_attempts(effects: dict[str, float], k: int) -> tuple[AttemptRecord, ...]:
    attempts = []
    index = 0
    offsets = {1: [0.0], 3: [-0.01, 0.0, 0.01]}[k]
    for state in confirmatory_states():
        if state.allocation != "confirmatory":
            continue
        effect = effects[state.state_id]
        for condition, sign in (("authority", 1.0), ("hedge", -1.0)):
            for replicate_index, offset in enumerate(offsets, start=1):
                value = 0.5 + sign * effect / 2.0 + offset
                output = ScientificOutput(
                    believe=0.4,
                    reshare=value,
                    attend=0.5,
                    reason_code="synthetic",
                )
                attempts.append(
                    AttemptRecord(
                        attempt_id=f"confirm-attempt-{index}",
                        phase="confirmatory",
                        slot_id=f"confirmatory:{state.state_id}:{condition}:{replicate_index}",
                        state_id=state.state_id,
                        scenario_id=state.scenario_id,
                        condition=condition,
                        replicate_index=replicate_index,
                        schedule_index=index,
                        attempt_index=1,
                        prompt_sha256=f"prompt-{state.state_id}-{condition}",
                        config_sha256="synthetic-config",
                        status="valid",
                        accepted=True,
                        started_at_utc="2026-07-17T00:00:00Z",
                        completed_at_utc="2026-07-17T00:00:01Z",
                        response_id=f"confirm-response-{index}",
                        usage_event_id=f"confirm-usage-{index}",
                        response_model_snapshot=V2FrozenConfig().model_snapshot,
                        prompt_tokens=100,
                        cached_prompt_tokens=0,
                        completion_tokens=20,
                        reasoning_tokens=0,
                        call_cost_usd=0.001,
                        raw_response=json.dumps(output.model_dump()),
                        parsed_output=output,
                    )
                )
                index += 1
    return tuple(attempts)


@pytest.mark.parametrize(
    ("effect", "direction", "practical"),
    [
        (0.06, "directional_replication", "operationally_meaningful_positive"),
        (0.01, "directional_replication", "practical_equivalence"),
        (-0.06, "sign_reversal", "operationally_meaningful_negative"),
    ],
)
def test_confirmatory_labels_are_mechanical(
    effect: float, direction: str, practical: str
) -> None:
    states = confirmatory_states()
    effects = {state.state_id: effect for state in states if state.allocation == "confirmatory"}
    result = analyze_confirmatory_reshare(
        confirmatory_attempts(effects, 1),
        states,
        1,
        _primary_resamples=500,
        _template_resamples=500,
    )
    assert result.estimate == pytest.approx(effect)
    assert result.directional_label == direction
    assert result.practical_label == practical
    assert result.permutation_p_value is None
    assert result.permutation_resamples == 0


def test_confirmatory_unresolved_and_stochastic_permutation_branches() -> None:
    states = confirmatory_states()
    selected = [state for state in states if state.allocation == "confirmatory"]
    heterogeneous = {
        state.state_id: (0.20 if index < 30 else -0.20)
        for index, state in enumerate(selected)
    }
    unresolved = analyze_confirmatory_reshare(
        confirmatory_attempts(heterogeneous, 1),
        states,
        1,
        _primary_resamples=1_000,
        _template_resamples=1_000,
    )
    assert unresolved.directional_label == "direction_unresolved"
    assert unresolved.practical_label == "practical_magnitude_unresolved"

    positive = {state.state_id: 0.05 for state in selected}
    stochastic = analyze_confirmatory_reshare(
        confirmatory_attempts(positive, 3),
        states,
        3,
        _primary_resamples=300,
        _template_resamples=300,
        _permutation_resamples=500,
    )
    assert stochastic.deterministic is False
    assert stochastic.permutation_resamples == 500
    assert 0.0 < stochastic.permutation_p_value <= 1.0
