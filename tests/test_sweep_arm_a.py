import json

import numpy as np

from intervention_response_audit import cli
from intervention_response_audit.config import resolve_arm_settings
from intervention_response_audit.network import generate_campus_network, sample_condition
from intervention_response_audit.oracle.client import MockLLMClient
from intervention_response_audit.oracle.runner import run_oracle_scenario


def test_arm_a_overlay_changes_only_registered_fields() -> None:
    baseline = resolve_arm_settings("baseline").model_dump()
    arm_a = resolve_arm_settings("armA").model_dump()
    changed = {key for key in baseline if baseline[key] != arm_a[key]}

    assert changed == {"arm_id", "reasoning_effort", "max_tokens", "temperature", "reference_simulator_id"}
    assert baseline["reasoning_effort"] == "none"
    assert arm_a["reasoning_effort"] == "medium"
    assert arm_a["max_tokens"] == 2000
    assert arm_a["temperature"] is None
    assert arm_a["reference_simulator_id"] == "gpt54mini_20260317_feed_v1_rmedium"


def test_arm_b_overlay_changes_only_model_and_registered_metadata() -> None:
    baseline = resolve_arm_settings("baseline").model_dump()
    arm_b = resolve_arm_settings("armB").model_dump()
    changed = {key for key in baseline if baseline[key] != arm_b[key]}

    assert changed == {"arm_id", "model", "reference_simulator_id"}
    assert arm_b["model"] == "gpt-5.4-2026-03-05"
    assert arm_b["reasoning_effort"] == baseline["reasoning_effort"] == "none"
    assert arm_b["temperature"] == baseline["temperature"] == 0.0
    assert arm_b["max_tokens"] == baseline["max_tokens"] == 300
    assert arm_b["reference_simulator_id"] == "gpt54full_20260305_feed_v1"


def test_arm_b_micro_specs_are_prefix_of_same_eighty_scenarios(monkeypatch) -> None:
    networks = [generate_campus_network(20, seed=950 + i) for i in range(5)]
    monkeypatch.setattr(cli, "_load_networks", lambda: networks)
    micro = cli._oracle_scenarios(5, seed=42, arm_id="armB")
    full = cli._oracle_scenarios(80, seed=42, arm_id="armB")

    assert cli._scenario_spec_fingerprints(micro) == cli._scenario_spec_fingerprints(full)[:5]
def test_arm_a_micro_specs_are_prefix_of_same_eighty_scenarios(monkeypatch) -> None:
    networks = [generate_campus_network(20, seed=900 + i) for i in range(5)]
    monkeypatch.setattr(cli, "_load_networks", lambda: networks)
    micro = cli._oracle_scenarios(5, seed=42, arm_id="armA")
    full = cli._oracle_scenarios(80, seed=42, arm_id="armA")

    assert cli._scenario_spec_fingerprints(micro) == cli._scenario_spec_fingerprints(full)[:5]


def test_arm_a_simulator_metadata_is_stamped_in_log_and_rows(tmp_path) -> None:
    net = generate_campus_network(20, seed=910)
    cond = sample_condition(np.random.default_rng(911), topic="internship")
    metadata = {
        "mechanism": "ic",
        "arm_id": "armA",
        "reference_simulator_id": "gpt54mini_20260317_feed_v1_rmedium",
    }
    result = run_oracle_scenario(
        net,
        cond,
        seeds={0, 1, 2},
        client=MockLLMClient(),
        T=2,
        rng=np.random.default_rng(912),
        scenario_id="arm_a_test",
        run_id="arm_a_test",
        log_dir=tmp_path,
        scenario_metadata=metadata,
    )
    records = [json.loads(line) for line in (tmp_path / "arm_a_test.jsonl").read_text().splitlines()]

    assert records
    assert all(record["arm_id"] == "armA" for record in records)
    assert all(record["reference_simulator_id"] == metadata["reference_simulator_id"] for record in records)
    frame = result.to_frame()
    assert set(frame["arm_id"]) == {"armA"}
    assert set(frame["reference_simulator_id"]) == {metadata["reference_simulator_id"]}
