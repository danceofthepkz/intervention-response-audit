import networkx as nx
import numpy as np

from intervention_response_audit.network import CampusNetwork, CREDIBILITY_TIERS, FRAMED_TEMPLATE_BANK, FRAMING_TYPES, generate_campus_network, render_semantic_twins, sample_condition


def test_generate_campus_network_is_deterministic() -> None:
    a = generate_campus_network(n=50, seed=123)
    b = generate_campus_network(n=50, seed=123)

    assert a.to_json() == b.to_json()


def test_generate_campus_network_shape_and_connectivity() -> None:
    net = generate_campus_network(n=50, seed=7)
    mean_degree = sum(dict(net.G.degree()).values()) / net.G.number_of_nodes()

    assert 5 <= mean_degree <= 12
    assert nx.is_connected(net.G)
    assert set(net.edge_features) == {net.edge_key(u, v) for u, v in net.G.edges()}


def test_campus_network_json_roundtrip() -> None:
    net = generate_campus_network(n=20, seed=11)
    loaded = CampusNetwork.from_json(net.to_json())

    assert loaded.to_json() == net.to_json()


def test_semantic_twins_differ_only_in_text_fields() -> None:
    rng = np.random.default_rng(9)
    condition = sample_condition(rng, topic="internship")
    twins = render_semantic_twins(condition, rng, k=2)

    assert len({t.text for t in twins + [condition]}) == 3
    for twin in twins:
        assert twin.topic == condition.topic
        assert twin.urgency == condition.urgency
        assert twin.credibility == condition.credibility
        assert twin.controversy == condition.controversy
        assert twin.text != condition.text
        assert twin.wording_id != condition.wording_id
        assert twin.framing_type != condition.framing_type


def test_template_bank_covers_credibility_tiers_with_wording_variants() -> None:
    for topic, tiers in FRAMED_TEMPLATE_BANK.items():
        assert set(tiers) == set(CREDIBILITY_TIERS)
        for tier, frames in tiers.items():
            assert set(frames) == set(FRAMING_TYPES), (topic, tier, set(frames))
            assert all(frames[frame]["framing_type"] == frame for frame in FRAMING_TYPES)
