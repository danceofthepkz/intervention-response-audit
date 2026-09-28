import math

import numpy as np
import pandas as pd
import pytest

from intervention_response_audit.v2.collect import read_attempt_log
from intervention_response_audit.v2.live import load_states
from intervention_response_audit.v2.reported_quantities import (
    DegenerateMidpointError,
    build_report,
    j6_replicate_split,
    second_order_approximation,
)


def _binary_entropy(probability: float) -> float:
    return -probability * math.log(probability) - (1.0 - probability) * math.log(
        1.0 - probability
    )


def test_j6_split_has_known_analytic_values_and_exact_identity() -> None:
    authority = np.array([[0.25, 0.25, 0.75]])
    hedge = np.array([[0.25, 0.75, 0.75]])
    result = j6_replicate_split(authority, hedge, ["synthetic"])
    row = result["per_state"][0]
    expected_j6 = math.log(2.0) - _binary_entropy(0.25)
    expected_between = math.log(2.0) - _binary_entropy(5.0 / 12.0)
    expected_within = expected_j6 - expected_between
    assert row["j6"] == pytest.approx(expected_j6)
    assert row["between"] == pytest.approx(expected_between)
    assert row["within"] == pytest.approx(expected_within)
    assert abs(row["identity_residual"]) <= result["identity_tolerance"]


def test_second_order_reports_degenerate_midpoint_without_clipping() -> None:
    with pytest.raises(DegenerateMidpointError, match="zero-state"):
        second_order_approximation(
            np.array([0.0, 0.4]),
            np.array([0.0, 0.2]),
            ["zero-state", "ordinary-state"],
        )


@pytest.fixture(scope="module")
def real_report() -> dict:
    return build_report(
        read_attempt_log("results/v2/confirmatory/attempts.jsonl"),
        load_states("prereg/v2/stage0"),
        pd.read_parquet("data/oracle_logs/phase1_full_v1.parquet"),
        selected_k=3,
    )


def test_real_j6_and_second_order_values(real_report: dict) -> None:
    split = real_report["j6_replicate_split"]
    assert split["means"]["j6"] == pytest.approx(0.0125555, abs=5e-8)
    assert split["means"]["between"] == pytest.approx(0.0112111, abs=5e-8)
    assert split["means"]["within"] == pytest.approx(0.00134439, abs=5e-9)
    assert split["maximum_absolute_identity_residual"] <= split["identity_tolerance"]
    assert len(split["per_state"]) == 60

    approximation = real_report["second_order_approximation"]
    assert approximation["mean_approx"] == pytest.approx(0.0110135, abs=5e-8)
    assert approximation["relative_error_of_mean"] == pytest.approx(
        -0.01763, abs=5e-6
    )
    assert approximation["pearson_r"] == pytest.approx(0.999956, abs=5e-7)
    assert approximation["naive_aggregate_plugin"]["approx"] == pytest.approx(
        0.00522, abs=5e-6
    )
    assert len(approximation["per_state"]) == 60


def test_real_aligned_regret_values_and_weighting_labels(real_report: dict) -> None:
    aligned = real_report["aligned_regret"]
    assert aligned["r_global"]["row"]["estimate"] == pytest.approx(0.0368253, abs=5e-8)
    assert aligned["r_global"]["scenario_equal"]["estimate"] == pytest.approx(
        0.0362165, abs=5e-8
    )
    assert aligned["r_total"]["state_side_mean"]["estimate"] == pytest.approx(
        0.0380636, abs=5e-8
    )
    assert aligned["r_total"]["call"]["estimate"] == pytest.approx(
        0.0394080, abs=5e-8
    )
    assert "weighting" in aligned["r_global"]["row"]
    assert "weighting" in aligned["r_global"]["scenario_equal"]
    assert "weighting" in aligned["r_total"]["state_side_mean"]
    assert "weighting" in aligned["r_total"]["call"]
    assert len(aligned["r_global"]["per_scenario"]) == 80
    assert len(aligned["r_total"]["per_state"]) == 60
