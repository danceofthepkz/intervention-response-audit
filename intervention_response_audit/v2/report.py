"""Mechanical reports for the frozen v2 decision branches."""

from __future__ import annotations

from intervention_response_audit.v2.schema import ConfirmatoryResult, PilotDecision


def pilot_decision_text(result: PilotDecision) -> str:
    """Render the post-pilot gate without discretionary wording."""

    if result.decision_code == "K_1_DETERMINISTIC":
        decision = "All 20 cells had exactly zero within-prompt variance; freeze K=1."
    elif result.decision_code == "HALT_K_INSUFFICIENT":
        decision = "K=10 does not satisfy the frozen call-noise rule; halt before confirmatory collection."
    else:
        decision = f"The smallest candidate satisfying the frozen call-noise rule is K={result.selected_k}."
    return (
        f"Pilot gate: {result.decision_code}. {decision} "
        f"Q90(v)={result.q90_v:.8f}; criterion={result.criterion:.8f}."
    )


def confirmatory_result_text(result: ConfirmatoryResult, delta: float = 0.03) -> str:
    """Render the pre-specified direction and practical-magnitude interpretation."""

    lo, hi = result.fixed_template_ci
    direction = {
        "directional_replication": "The authority-minus-hedge contrast is positive, replicating the predicted v1 direction.",
        "sign_reversal": "Framing affects reshare probability, but the effect reverses the predicted v1 direction.",
        "direction_unresolved": "The interval does not resolve the direction of the authority-minus-hedge contrast.",
    }[result.directional_label]
    practical = {
        "operationally_meaningful_positive": f"The interval lies above the +{delta:.2f} operational threshold.",
        "operationally_meaningful_negative": f"The interval lies below the -{delta:.2f} operational threshold.",
        "practical_equivalence": f"The interval lies entirely within ±{delta:.2f}, supporting practical equivalence at that scale.",
        "practical_magnitude_unresolved": f"The interval crosses a ±{delta:.2f} boundary, so practical magnitude remains unresolved.",
    }[result.practical_label]
    sensitivity = (
        "The template-cluster sensitivity changes at least one decision label, so the result is template-sensitive."
        if result.template_sensitive
        else "The fixed-template and template-cluster intervals yield the same decision labels."
    )
    return (
        f"Across {result.state_count} frozen states, the signed reshare contrast was "
        f"{result.estimate:.4f} (95% fixed-template bootstrap CI [{lo:.4f}, {hi:.4f}]). "
        f"{direction} {practical} {sensitivity}"
    )
