import numpy as np
import pytest

from intervention_response_audit.v2.secondary import anchor_decomposition, bernoulli_kl, text_blind_regret


def test_text_blind_regret_is_zero_for_equal_sides_and_positive_for_contrast() -> None:
    equal = text_blind_regret(np.array([0.2]), np.array([0.2]))
    contrast = text_blind_regret(np.array([0.8]), np.array([0.2]))
    assert equal[0] == pytest.approx(0.0)
    assert contrast[0] > 0.0


def test_anchor_decomposition_identity_holds() -> None:
    authority = np.array([0.8, 0.4, 0.7])
    hedge = np.array([0.2, 0.3, 0.5])
    anchor = np.array([0.45, 0.5, 0.6])
    result = anchor_decomposition(authority, hedge, anchor)
    assert result["r_total_cf_anchor"] == pytest.approx(
        result["r_text_raw"] + result["r_struct_cf_anchor"]
    )
    assert abs(result["decomposition_error"]) < 1e-12


def test_bernoulli_kl_is_nonnegative() -> None:
    values = bernoulli_kl(np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.5, 0.8]))
    assert np.all(values >= -1e-15)
