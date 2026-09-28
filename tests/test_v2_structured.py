from collections import Counter

import numpy as np
import pandas as pd

from intervention_response_audit.v2.structured import constrained_outer_folds, run_structured_crossfit
from intervention_response_audit.v2.schema import StateRecord


def rows():
    return pd.read_parquet("data/oracle_logs/phase1_full_v1.parquet")


def states():
    return tuple(
        StateRecord.model_validate_json(line)
        for line in open("prereg/v2/stage0/state_allocation.jsonl", encoding="utf-8")
    )


def test_outer_folds_are_deterministic_scenario_level_and_balanced() -> None:
    frame = rows()
    left = constrained_outer_folds(frame)
    right = constrained_outer_folds(frame)
    assert left == right
    assert Counter(left.values()) == {0: 16, 1: 16, 2: 16, 3: 16, 4: 16}
    scenario = frame.groupby("scenario_id").first().reset_index()
    scenario["fold"] = scenario["scenario_id"].map(left)
    for _, group in scenario.groupby("topic"):
        counts = group["fold"].value_counts().reindex(range(5), fill_value=0)
        assert counts.max() - counts.min() <= 1
    for _, group in scenario.groupby("credibility"):
        counts = group["fold"].value_counts().reindex(range(5), fill_value=0)
        assert counts.max() - counts.min() <= 1


def test_soft_label_crossfit_is_complete_honest_and_deterministic() -> None:
    frame = rows()
    first = run_structured_crossfit(frame, states())
    second = run_structured_crossfit(frame, states())
    assert first.feature_count == 27
    assert len(first.predictions) == len(frame)
    assert len(first.anchor_prediction_by_state) == 60
    assert set(first.selected_c_by_outer_fold.values()) <= {0.001, 0.003, 0.01, 0.03, 0.1, 0.3}
    assert np.allclose(
        first.predictions["structured_probability"],
        second.predictions["structured_probability"],
    )
    assert first.r_global_full >= 0.0
