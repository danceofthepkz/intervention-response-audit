import math

import numpy as np
import pandas as pd

from intervention_response_audit.audit import TARGETS, run_audit
from intervention_response_audit.oracle.client import MockLLMClient
from intervention_response_audit.oracle.runner import run_oracle_scenario
from intervention_response_audit.network import generate_campus_network, sample_condition


def _mock_rows(tmp_path) -> pd.DataFrame:
    frames = []
    for i in range(6):
        net = generate_campus_network(30, seed=100 + i)
        cond = sample_condition(np.random.default_rng(200 + i))
        result = run_oracle_scenario(
            net,
            cond,
            seeds={0, 1, 2},
            client=MockLLMClient(),
            T=4,
            rng=np.random.default_rng(300 + i),
            scenario_id=f"scenario_{i}",
            run_id=f"audit_{i}",
            log_dir=tmp_path,
        )
        frames.append(result.to_frame())

    base = frames[0].head(10).copy()
    repeats = []
    for scaffold in range(5):
        repeat = base.copy()
        repeat["scenario_id"] = "noise_item"
        repeat["scaffold_id"] = scaffold
        for target in TARGETS:
            repeat[target] = np.clip(repeat[target] + (scaffold - 2) * 0.01, 0.001, 0.999)
        repeats.append(repeat)

    twin_a = frames[1].head(12).copy()
    twin_b = twin_a.copy()
    twin_a["scenario_id"] = "twins_0_a"
    twin_b["scenario_id"] = "twins_0_b"
    twin_a["twin_group"] = "twins_0"
    twin_b["twin_group"] = "twins_0"
    twin_a["twin_side"] = "a"
    twin_b["twin_side"] = "b"
    twin_b["condition_text"] = "A startup recruiter link is going around, though someone said it might be promo spam."
    for target in TARGETS:
        twin_b[target] = np.clip(twin_b[target] - 0.15, 0.001, 0.999)
    frames.extend(repeats + [twin_a, twin_b])
    return pd.concat(frames, ignore_index=True)


def test_full_audit_runs_and_writes_report(tmp_path) -> None:
    rows = _mock_rows(tmp_path)
    result = run_audit(rows, output_dir=tmp_path / "audit", seed=123)

    assert result.report_path.exists()
    assert result.nll_plot_path.exists()
    assert result.twin_plot_path.exists()
    assert set(result.metrics["target"]) == set(TARGETS)
    assert "Decision:" in result.report_path.read_text(encoding="utf-8")


def _synthetic_logistic_rows(semantic_tweak: bool) -> pd.DataFrame:
    rng = np.random.default_rng(12)
    rows = []
    for scenario in range(8):
        for item in range(18):
            topic = ["course_material", "club_event", "gossip", "internship", "policy_change"][item % 5]
            common = {
                "scenario_id": f"synth_{scenario}",
                "query_index": 1,
                "scaffold_id": item % 5,
                "step": 1 + item % 4,
                "agent_id": item,
                "n_exposures": 1 + item % 3,
                "sum_tie_strength": rng.uniform(0.1, 2.2),
                "max_tie_strength": rng.uniform(0.1, 0.9),
                "same_major_exposers": item % 2,
                "same_club_exposers": (item + 1) % 2,
                "source_diversity": 1 + item % 4,
                "n_endorse": item % 3,
                "n_neutral": (item + 1) % 3,
                "n_doubt": (item + 2) % 3,
                "topic": topic,
                "urgency": rng.uniform(0.1, 0.9),
                "credibility": rng.uniform(0.1, 0.9),
                "controversy": rng.uniform(0.1, 0.9),
                "wording_id": f"{topic}:0:0",
                "condition_text": "ordinary wording",
                "socialness": rng.uniform(0.1, 0.9),
                "trust": rng.uniform(0.1, 0.9),
                "topic_interest": rng.uniform(0.1, 0.9),
                "year": 1 + item % 4,
                "exposer_ids": "[0]",
                "sampled_reshare": False,
                "reason_code": "synthetic",
            }
            z = (
                -0.4
                + 0.45 * common["n_exposures"]
                + 0.65 * common["credibility"]
                - 0.35 * common["trust"]
                + 0.25 * common["topic_interest"] * common["urgency"]
            )
            p = 1.0 / (1.0 + math.exp(-z))
            semantic_delta = 0.0
            if semantic_tweak and item % 4 == 0:
                semantic_delta = 0.32
                common["condition_text"] = "Official staff confirmed update."
            elif semantic_tweak and item % 4 == 1:
                semantic_delta = -0.32
                common["condition_text"] = "This looks like promo spam and is not verified."
            common["believe"] = np.clip(p - 0.08 + semantic_delta, 0.001, 0.999)
            common["reshare"] = np.clip(p - 0.18 + semantic_delta, 0.001, 0.999)
            common["attend"] = 0.2
            rows.append(common)

    base = rows[:80]
    for scaffold in range(5):
        for idx, row in enumerate(base):
            repeat = dict(row)
            repeat["scenario_id"] = "noise"
            repeat["scaffold_id"] = scaffold
            repeat["agent_id"] = idx
            rows.append(repeat)

    for side, delta in [("a", 0.0), ("b", 0.16 if semantic_tweak else 0.0)]:
        for idx, row in enumerate(rows[:8]):
            twin = dict(row)
            twin["scenario_id"] = f"pair_0_{side}"
            twin["twin_group"] = "pair_0"
            twin["twin_side"] = side
            twin["agent_id"] = idx
            twin["believe"] = np.clip(twin["believe"] + delta, 0.001, 0.999)
            twin["reshare"] = np.clip(twin["reshare"] + delta, 0.001, 0.999)
            rows.append(twin)
    return pd.DataFrame(rows)


def test_pure_logistic_mock_fails_audit(tmp_path) -> None:
    result = run_audit(_synthetic_logistic_rows(semantic_tweak=False), output_dir=tmp_path, seed=4)

    assert result.decision == "FAIL"
    assert result.metrics["Gap"].abs().max() < 0.12


def test_semantic_tweak_makes_twin_delta_positive(tmp_path) -> None:
    rows = _synthetic_logistic_rows(semantic_tweak=True).reset_index(drop=True)
    result = run_audit(rows, output_dir=tmp_path, seed=5)

    assert result.decision == "PASS"
    assert result.metrics["twins_oracle_abs_delta"].max() >= 0.10
