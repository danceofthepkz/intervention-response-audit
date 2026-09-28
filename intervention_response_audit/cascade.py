"""Mechanistic cascade simulators and Monte Carlo activation trajectories."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np

from intervention_response_audit.network import CampusNetwork, Condition, EdgeFeatures, Persona

Mechanism = Literal["ic", "lt", "cc"]

TOPIC_CHANNEL_TABLE: dict[str, dict[str, tuple[str, ...]]] = {
    "course_material": {
        "boosted": ("same_major", "classmate"),
        "damped": ("cross_major",),
    },
    "club_event": {
        "boosted": ("same_club", "clubmate"),
        "damped": ("non_club",),
    },
    "gossip": {
        "boosted": ("chat_frequency_high",),
        "damped": ("chat_frequency_low",),
    },
    "internship": {
        "boosted": ("weak_cross_major",),
        "damped": (),
    },
    "policy_change": {
        "boosted": ("dorm", "high_tie_strength"),
        "damped": (),
    },
}


def edge_prob(ef: EdgeFeatures, pj: Persona, cond: Condition) -> float:
    """Return condition-modulated transmissibility for one directed exposure."""

    p = 0.05
    if cond.topic == "course_material":
        if ef.same_major:
            p *= 4.0
        if "classmate" in ef.contexts:
            p *= 3.0
        if not ef.same_major:
            p *= 0.5
    elif cond.topic == "club_event":
        if ef.same_club:
            p *= 4.0
        if "clubmate" in ef.contexts:
            p *= 3.0
        if not ef.same_club:
            p *= 0.5
    elif cond.topic == "gossip":
        if ef.chat_frequency == "high":
            p *= 4.0
        if ef.chat_frequency == "low":
            p *= 0.5
    elif cond.topic == "internship":
        if ef.tie_strength < 0.3 and not ef.same_major:
            p *= 2.0
    elif cond.topic == "policy_change":
        if "dorm" in ef.contexts:
            p *= 3.0
        if ef.tie_strength > 0.65:
            p *= 3.0

    topic_interest = pj.topic_interests.get(cond.topic, 0.5)
    p *= 0.5 + cond.urgency
    p *= 0.5 + cond.credibility
    p *= 0.5 + topic_interest
    return float(np.clip(p, 0.001, 0.9))


def _sorted_seeds(seeds: Iterable[int]) -> set[int]:
    return {int(s) for s in seeds}


def _edge_prob_for(net: CampusNetwork, u: int, v: int, cond: Condition) -> float:
    return edge_prob(net.edge_features[net.edge_key(u, v)], net.personas[v], cond)


def run_ic(
    net: CampusNetwork,
    cond: Condition,
    seeds: Iterable[int],
    T: int = 8,
    rng: np.random.Generator | None = None,
) -> list[set[int]]:
    """Run Independent Cascade; returns cumulative active sets length `T + 1`."""

    rng = rng or np.random.default_rng()
    active = _sorted_seeds(seeds)
    frontier = set(active)
    history = [set(active)]
    for _ in range(T):
        new_active: set[int] = set()
        for u in sorted(frontier):
            for v in sorted(net.G.neighbors(u)):
                if v in active:
                    continue
                if rng.random() < _edge_prob_for(net, u, v, cond):
                    new_active.add(v)
        active |= new_active
        frontier = new_active
        history.append(set(active))
    return history


def run_lt(
    net: CampusNetwork,
    cond: Condition,
    seeds: Iterable[int],
    T: int = 8,
    rng: np.random.Generator | None = None,
) -> list[set[int]]:
    """Run Linear Threshold diffusion; returns cumulative active sets length `T + 1`."""

    rng = rng or np.random.default_rng()
    active = _sorted_seeds(seeds)
    thresholds = {i: float(rng.random()) for i in net.G.nodes()}
    weights: dict[tuple[int, int], float] = {}
    for v in net.G.nodes():
        incoming = [(u, _edge_prob_for(net, u, v, cond)) for u in net.G.neighbors(v)]
        total = sum(p for _, p in incoming) or 1.0
        for u, p in incoming:
            weights[(u, v)] = p / total

    history = [set(active)]
    for _ in range(T):
        new_active = {
            v
            for v in net.G.nodes()
            if v not in active
            and sum(weights[(u, v)] for u in net.G.neighbors(v) if u in active) >= thresholds[v]
        }
        active |= new_active
        history.append(set(active))
    return history


def run_cc(
    net: CampusNetwork,
    cond: Condition,
    seeds: Iterable[int],
    T: int = 8,
    rng: np.random.Generator | None = None,
    k: int = 2,
) -> list[set[int]]:
    """Run accumulated complex contagion; returns cumulative active sets length `T + 1`."""

    rng = rng or np.random.default_rng()
    active = _sorted_seeds(seeds)
    frontier = set(active)
    successful_sources: dict[int, set[int]] = {i: set() for i in net.G.nodes()}
    history = [set(active)]
    for _ in range(T):
        for u in sorted(frontier):
            for v in sorted(net.G.neighbors(u)):
                if v not in active and rng.random() < _edge_prob_for(net, u, v, cond):
                    successful_sources[v].add(u)
        new_active = {v for v, sources in successful_sources.items() if v not in active and len(sources) >= k}
        active |= new_active
        frontier = new_active
        history.append(set(active))
    return history


def _history_to_matrix(history: list[set[int]], n: int) -> np.ndarray:
    mat = np.zeros((len(history), n), dtype=float)
    for t, active in enumerate(history):
        mat[t, list(active)] = 1.0
    return mat


def _run_one(args: tuple[CampusNetwork, Mechanism, Condition, tuple[int, ...], int, int]) -> np.ndarray:
    net, mechanism, cond, seeds, T, seed = args
    rng = np.random.default_rng(seed)
    runner: dict[Mechanism, Callable[..., list[set[int]]]] = {
        "ic": run_ic,
        "lt": run_lt,
        "cc": run_cc,
    }
    history = runner[mechanism](net, cond, seeds, T=T, rng=rng)
    return _history_to_matrix(history, net.G.number_of_nodes())


def mc_prob_trajectory(
    net: CampusNetwork,
    mechanism: Mechanism,
    cond: Condition,
    seeds: Iterable[int],
    R: int = 300,
    T: int = 8,
    seed: int = 42,
    processes: int = 1,
) -> np.ndarray:
    """Estimate activation frequencies over rollouts.

    Returns an array with shape `(T + 1, N)`, where `N` is the node count.
    """

    if mechanism not in {"ic", "lt", "cc"}:
        raise ValueError(f"Unknown mechanism: {mechanism}")
    seed_tuple = tuple(sorted(int(s) for s in seeds))
    child_seeds = np.random.default_rng(seed).integers(0, 2**32 - 1, size=R, dtype=np.uint32)
    args = [(net, mechanism, cond, seed_tuple, T, int(s)) for s in child_seeds]
    if processes > 1 and R * net.G.number_of_nodes() > 20_000:
        with mp.Pool(processes=processes) as pool:
            mats = pool.map(_run_one, args)
    else:
        mats = [_run_one(arg) for arg in args]
    return np.mean(mats, axis=0)
