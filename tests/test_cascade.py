import time

import networkx as nx
import numpy as np

from intervention_response_audit import cascade
from intervention_response_audit.cascade import mc_prob_trajectory, run_cc, run_ic, run_lt
from intervention_response_audit.network import CampusNetwork, Condition, EdgeFeatures, Persona, generate_campus_network


def _persona() -> Persona:
    return Persona(
        year=1,
        major="CS",
        clubs=("Robotics",),
        socialness=0.6,
        trust=0.7,
        topic_interests={
            "course_material": 0.8,
            "club_event": 0.8,
            "gossip": 0.8,
            "internship": 0.8,
            "policy_change": 0.8,
        },
    )


def _condition() -> Condition:
    return Condition(
        topic="course_material",
        urgency=0.7,
        credibility=0.8,
        controversy=0.1,
        text="Professor shared the review sheet.",
        wording_id="course_material:0:0",
    )


def _line_net() -> CampusNetwork:
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    personas = {i: _persona() for i in range(3)}
    ef = EdgeFeatures(0.5, True, True, True, frozenset({"classmate", "clubmate"}), "med")
    return CampusNetwork(G, personas, {(0, 1): ef, (1, 2): ef})


def test_mc_seeds_are_one_and_monotone() -> None:
    net = generate_campus_network(n=30, seed=2)
    traj = mc_prob_trajectory(net, "ic", _condition(), seeds={0, 1}, R=40, T=5, seed=3)

    assert np.all(traj[:, [0, 1]] == 1.0)
    assert np.all(np.diff(traj, axis=0) >= -1e-12)


def test_zero_edge_probability_prevents_spread(monkeypatch) -> None:
    monkeypatch.setattr(cascade, "edge_prob", lambda ef, pj, cond: 0.0)
    traj = mc_prob_trajectory(_line_net(), "ic", _condition(), seeds={0}, R=20, T=3, seed=1)

    assert np.all(traj[:, 0] == 1.0)
    assert np.all(traj[:, 1:] == 0.0)


def test_high_edge_probability_spreads_on_connected_graph(monkeypatch) -> None:
    monkeypatch.setattr(cascade, "edge_prob", lambda ef, pj, cond: 0.9)
    net = generate_campus_network(n=50, seed=5)
    traj = mc_prob_trajectory(net, "ic", _condition(), seeds={0}, R=120, T=8, seed=6)

    assert traj[-1].mean() > 0.92


def test_ic_line_graph_matches_hand_computed_expectation(monkeypatch) -> None:
    monkeypatch.setattr(cascade, "edge_prob", lambda ef, pj, cond: 0.3)
    traj = mc_prob_trajectory(_line_net(), "ic", _condition(), seeds={0}, R=8_000, T=2, seed=8)

    assert abs(traj[1, 1] - 0.3) < 0.025
    assert abs(traj[2, 1] - 0.3) < 0.025
    assert abs(traj[2, 2] - 0.09) < 0.02


def test_lt_and_cc_are_monotone() -> None:
    net = generate_campus_network(n=25, seed=10)
    rng = np.random.default_rng(4)

    for history in [
        run_lt(net, _condition(), seeds={0}, T=4, rng=rng),
        run_cc(net, _condition(), seeds={0, 1}, T=4, rng=rng, k=2),
    ]:
        sizes = [len(step) for step in history]
        assert sizes == sorted(sizes)


def test_runtime_single_process_under_acceptance_target() -> None:
    net = generate_campus_network(n=50, seed=12)
    start = time.perf_counter()
    traj = mc_prob_trajectory(net, "ic", _condition(), seeds={0, 1, 2}, R=300, T=8, seed=13)

    assert traj.shape == (9, 50)
    assert time.perf_counter() - start < 10.0
