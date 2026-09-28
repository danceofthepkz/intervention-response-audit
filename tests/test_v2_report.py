from intervention_response_audit.v2.report import confirmatory_result_text, pilot_decision_text
from intervention_response_audit.v2.schema import ConfirmatoryResult, PilotDecision


def pilot(code, k, deterministic=False):
    return PilotDecision(
        decision_code=code,
        selected_k=k,
        deterministic=deterministic,
        valid_slots=200,
        state_count=10,
        cell_count=20,
        q90_v=0.001,
        se_call_by_k={"3": 0.002, "5": 0.0015, "10": 0.001},
        cell_variance_by_key={"x": 0.0},
        criterion=0.0075,
        near_deterministic=deterministic,
    )


def result(direction, practical, *, sensitive=False):
    return ConfirmatoryResult(
        selected_k=3,
        deterministic=False,
        state_count=60,
        estimate=0.04,
        fixed_template_ci=(0.031, 0.049),
        template_cluster_ci=(0.025, 0.055),
        directional_label=direction,
        practical_label=practical,
        template_sensitive=sensitive,
        permutation_p_value=0.001,
        primary_bootstrap_resamples=20_000,
        template_bootstrap_resamples=20_000,
        permutation_resamples=100_000,
        state_contrasts={"s": 0.04},
    )


def test_every_pilot_branch_has_frozen_text() -> None:
    branches = [
        ("K_1_DETERMINISTIC", 1),
        ("K_3", 3),
        ("K_5", 5),
        ("K_10", 10),
        ("HALT_K_INSUFFICIENT", None),
    ]
    for code, k in branches:
        assert code in pilot_decision_text(pilot(code, k, code == "K_1_DETERMINISTIC"))


def test_all_direction_and_practical_labels_have_frozen_text() -> None:
    directions = ("directional_replication", "sign_reversal", "direction_unresolved")
    practicals = (
        "operationally_meaningful_positive",
        "operationally_meaningful_negative",
        "practical_equivalence",
        "practical_magnitude_unresolved",
    )
    texts = [confirmatory_result_text(result(d, p)) for d in directions for p in practicals]
    assert len(texts) == 12
    assert all("95% fixed-template bootstrap CI" in text for text in texts)


def test_template_sensitivity_is_explicit() -> None:
    text = confirmatory_result_text(
        result("directional_replication", "operationally_meaningful_positive", sensitive=True)
    )
    assert "template-sensitive" in text
