from collections import Counter
from itertools import combinations

import pandas as pd
import pytest

from intervention_response_audit import cli
from intervention_response_audit.network import FRAMING_TYPES, TOPICS, generate_campus_network
from intervention_response_audit.oracle.prompts import SCAFFOLDS
from intervention_response_audit.oracle.runner import make_semantic_twin_set


def test_d1_d2_twins_cover_each_framing_combination_three_times() -> None:
    pairs = make_semantic_twin_set(seed=700, count=18)
    order = {frame: index for index, frame in enumerate(FRAMING_TYPES)}
    realized = Counter(
        tuple(sorted((left.cond.framing_type, right.cond.framing_type), key=order.get))
        for left, right in pairs
    )

    assert len(pairs) == 18
    assert realized == Counter({combo: 3 for combo in combinations(FRAMING_TYPES, 2)})
    for left, right in pairs:
        assert left.net is right.net
        assert left.seeds == right.seeds
        assert left.mechanism == right.mechanism
        assert left.cond.topic == right.cond.topic
        assert left.cond.urgency == right.cond.urgency
        assert left.cond.credibility == right.cond.credibility
        assert left.cond.controversy == right.cond.controversy
        assert left.cond.text != right.cond.text


def test_d3_floor_selection_is_40_stratified_points() -> None:
    rows = []
    for i in range(100):
        rows.append(
            {
                "scenario_id": f"scenario_{i // 5:03d}",
                "step": 1 + i % 3,
                "agent_id": i,
                "query_index": 1,
                "topic": TOPICS[i % len(TOPICS)],
                "n_exposures": 1 + (i % 4),
            }
        )
    frame = pd.DataFrame(rows)
    allocations = cli._floor_stratum_allocations(frame, 40)
    selected = cli._select_floor_points(frame, 40, seed=42)

    assert len(selected) == 40
    assert sum(allocations.values()) == 40
    realized = {
        (str(topic), int(n_exposures)): int(size)
        for (topic, n_exposures), size in selected.groupby(["topic", "n_exposures"]).size().items()
    }
    assert realized == {key: value for key, value in allocations.items() if value}
    prompts = [f"{scaffold}\n\nfixed content" for scaffold in SCAFFOLDS]
    assert len({cli._non_scaffold_prompt(prompt) for prompt in prompts}) == 1


def test_d4_eighty_scenarios_are_topic_balanced(monkeypatch) -> None:
    networks = [generate_campus_network(20, seed=800 + i) for i in range(5)]
    monkeypatch.setattr(cli, "_load_networks", lambda: networks)
    specs = cli._default_scenarios(80, seed=42)

    assert len(specs) == 80
    assert Counter(spec.cond.topic for spec in specs) == Counter({topic: 16 for topic in TOPICS})
    assert {spec.mechanism for spec in specs} <= set(cli.MECHANISMS)


def test_per_combination_twin_report_has_six_groups_of_three() -> None:
    rows = []
    pair_index = 0
    for framing_a, framing_b in combinations(FRAMING_TYPES, 2):
        for _ in range(3):
            for side, framing, value in (("a", framing_a, 0.3), ("b", framing_b, 0.5)):
                rows.append(
                    {
                        "twin_group": f"pair_{pair_index}",
                        "twin_side": side,
                        "step": 1,
                        "agent_id": 1,
                        "query_index": 1,
                        "framing_type": framing,
                        "believe": value,
                        "reshare": value - 0.1,
                    }
                )
            pair_index += 1
    summary = cli._per_combination_twins(pd.DataFrame(rows))

    assert len(summary) == 6
    assert summary["n"].tolist() == [3] * 6
    assert summary["believe"].tolist() == pytest.approx([0.2] * 6)
    assert summary["reshare"].tolist() == pytest.approx([0.2] * 6)
