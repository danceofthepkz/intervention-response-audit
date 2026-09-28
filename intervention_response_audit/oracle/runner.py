"""Oracle-in-the-loop diffusion runner and special audit set builders."""

from __future__ import annotations

import json
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_exponential

from intervention_response_audit.config import settings
from intervention_response_audit.network import (
    FRAMED_TEMPLATE_BANK,
    FRAMING_TYPES,
    CampusNetwork,
    Condition,
    generate_campus_network,
    sample_condition,
)
from intervention_response_audit.oracle.client import BudgetExceeded, BudgetGuard, LLMClient, complete_json_logged, load_completed_calls
from intervention_response_audit.oracle.prompts import COMMENT_BANK, build_decision_prompt, build_requery_prompt, sample_share_metadata


@dataclass
class ScenarioResult:
    scenario_id: str
    rows: list[dict[str, Any]]
    heard_history: list[set[int]]
    reshared_history: list[set[int]]
    exposure_script: dict[tuple[int, int], dict[str, Any]]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def to_parquet(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_parquet(path, index=False)


@retry(
    wait=wait_exponential(multiplier=0.2, min=0.2, max=2),
    stop=stop_after_attempt(3),
    retry=retry_if_not_exception_type(BudgetExceeded),
    reraise=True,
)
def _call_with_retry(**kwargs: Any) -> dict[str, Any]:
    return complete_json_logged(**kwargs)


def _scenario_features(
    net: CampusNetwork,
    agent: int,
    exposers: set[int],
    cond: Condition,
    step: int,
) -> dict[str, Any]:
    strengths = [net.edge_features[net.edge_key(agent, j)].tie_strength for j in exposers]
    contexts = set()
    same_major = 0
    same_club = 0
    for j in exposers:
        ef = net.edge_features[net.edge_key(agent, j)]
        contexts |= set(ef.contexts)
        same_major += int(ef.same_major)
        same_club += int(ef.same_club)
    persona = net.personas[agent]
    return {
        "step": step,
        "agent_id": agent,
        "n_exposures": len(exposers),
        "sum_tie_strength": float(sum(strengths)),
        "max_tie_strength": float(max(strengths) if strengths else 0.0),
        "same_major_exposers": same_major,
        "same_club_exposers": same_club,
        "source_diversity": len(contexts),
        "topic": cond.topic,
        "urgency": cond.urgency,
        "credibility": cond.credibility,
        "controversy": cond.controversy,
        "wording_id": cond.wording_id,
        "framing_type": cond.framing_type,
        "condition_text": cond.text,
        "socialness": persona.socialness,
        "trust": persona.trust,
        "topic_interest": persona.topic_interests.get(cond.topic, 0.5),
        "year": persona.year,
    }


def _stance_probs() -> tuple[float, float, float]:
    return (
        settings.stance_endorse_prob,
        settings.stance_neutral_prob,
        settings.stance_doubt_prob,
    )


def _exposure_metadata(exposers: set[int], share_metadata: dict[int, dict[str, str]], topic: str) -> list[dict[str, Any]]:
    used: dict[str, set[str]] = {}
    rows = []
    for exposer in sorted(exposers):
        item = dict(share_metadata[exposer])
        stance = item["stance"]
        used.setdefault(stance, set())
        if item["comment_id"] in used[stance]:
            replacements = [
                {"comment_id": comment_id, "comment": comment}
                for comment_id, comment in COMMENT_BANK[topic][stance]
                if comment_id not in used[stance]
            ]
            if replacements:
                item.update(replacements[0])
            else:
                item["comment_id"] = f"{topic}:{stance}:extra:{exposer}"
                item["comment"] = f"{item['comment']} Adding my own signal here."
        used[stance].add(item["comment_id"])
        rows.append(
            {
                "agent_id": int(exposer),
                "stance": item["stance"],
                "comment_id": item["comment_id"],
                "comment": item["comment"],
            }
        )
    return rows


def _exposure_count_features(metadata: list[dict[str, Any]]) -> dict[str, int]:
    stances = [item["stance"] for item in metadata]
    return {
        "n_endorse": stances.count("endorse"),
        "n_neutral": stances.count("neutral"),
        "n_doubt": stances.count("doubt"),
    }


def run_oracle_scenario(
    net: CampusNetwork,
    cond: Condition,
    seeds: Iterable[int],
    client: LLMClient,
    T: int = 8,
    rng: np.random.Generator | None = None,
    max_queries_per_agent: int = 2,
    scenario_id: str = "scenario_0",
    run_id: str = "mock_run",
    log_dir: str | Path | None = None,
    budget: BudgetGuard | None = None,
    scaffold_id: int = 0,
    exposure_script: dict[tuple[int, int], dict[str, Any]] | None = None,
    stance_probs: tuple[float, float, float] | None = None,
    scenario_metadata: dict[str, Any] | None = None,
) -> ScenarioResult:
    """Run feed-based oracle diffusion and return one row per queried decision."""

    rng = rng or np.random.default_rng(settings.seed)
    log_root = Path(log_dir) if log_dir is not None else settings.data_dir / "oracle_logs"
    log_path = log_root / f"{run_id}.jsonl"
    completed = load_completed_calls(log_path)
    budget = budget or BudgetGuard(
        max_api_calls=settings.max_api_calls,
        max_usd=settings.max_usd,
        input_usd_per_1m=settings.input_usd_per_1m,
        output_usd_per_1m=settings.output_usd_per_1m,
    )

    seed_set = {int(s) for s in seeds}
    heard = set(seed_set)
    reshared = set(seed_set)
    query_counts = {i: 0 for i in net.G.nodes()}
    last_exposures: dict[int, set[int]] = {i: set() for i in net.G.nodes()}
    rows: list[dict[str, Any]] = []
    heard_history = [set(heard)]
    reshared_history = [set(reshared)]
    result_script: dict[tuple[int, int], dict[str, Any]] = {}
    active_stance_probs = stance_probs or _stance_probs()
    share_metadata = {
        seed: sample_share_metadata(cond.topic, rng, active_stance_probs)
        for seed in sorted(seed_set)
    }

    for step in range(1, T + 1):
        candidates: list[tuple[int, set[int]]] = []
        if exposure_script is not None:
            for (script_step, agent), entry in sorted(exposure_script.items()):
                if script_step == step:
                    exposers = {int(x) for x in entry["exposers"]}
                    heard.add(agent)
                    candidates.append((agent, exposers))
        else:
            for agent in sorted(net.G.nodes()):
                if agent in seed_set:
                    continue
                exposers = {j for j in net.G.neighbors(agent) if j in reshared}
                if exposers:
                    heard.add(agent)
                if not exposers or query_counts[agent] >= max_queries_per_agent:
                    continue
                if exposers <= last_exposures[agent]:
                    continue
                candidates.append((agent, exposers))

        newly_reshared: set[int] = set()
        for agent, exposers in candidates:
            previous_exposures = set(last_exposures[agent])
            new_exposures = set(exposers) - previous_exposures
            last_exposures[agent] = set(exposers)
            query_counts[agent] += 1
            script_entry = exposure_script.get((step, agent)) if exposure_script is not None else None
            if script_entry is not None:
                metadata = list(script_entry["exposure_metadata"])
                exposure_details = {
                    int(item["agent_id"]): {
                        "stance": str(item["stance"]),
                        "comment_id": str(item["comment_id"]),
                        "comment": str(item["comment"]),
                    }
                    for item in metadata
                }
            else:
                metadata = _exposure_metadata(exposers, share_metadata, cond.topic)
                exposure_details = {
                    int(item["agent_id"]): {
                        "stance": str(item["stance"]),
                        "comment_id": str(item["comment_id"]),
                        "comment": str(item["comment"]),
                    }
                    for item in metadata
                }
            key = (scenario_id, step, agent, scaffold_id)
            if key in completed:
                parsed = completed[key]
            else:
                if previous_exposures:
                    prompt = build_requery_prompt(
                        net,
                        agent,
                        previous_exposures,
                        new_exposures,
                        cond,
                        scaffold_id=scaffold_id,
                        exposure_details=exposure_details,
                    )
                else:
                    prompt = build_decision_prompt(
                        net,
                        agent,
                        exposers,
                        cond,
                        scaffold_id=scaffold_id,
                        exposure_details=exposure_details,
                    )
                parsed = _call_with_retry(
                    client=client,
                    prompt=prompt,
                    log_path=log_path,
                    budget=budget,
                    scenario_id=scenario_id,
                    step=step,
                    agent_id=agent,
                    scaffold_id=scaffold_id,
                    extra_log_fields={
                        "exposure_metadata": metadata,
                        "framing_type": cond.framing_type,
                        **(scenario_metadata or {}),
                    },
                )

            if script_entry is not None:
                sampled_reshare = bool(script_entry["sampled_reshare"])
            else:
                sampled_reshare = rng.random() < float(parsed["reshare"])
            if sampled_reshare:
                newly_reshared.add(agent)
                if agent not in share_metadata:
                    share_metadata[agent] = sample_share_metadata(cond.topic, rng, active_stance_probs)

            result_script[(step, agent)] = {
                "exposers": sorted(exposers),
                "exposure_metadata": metadata,
                "sampled_reshare": sampled_reshare,
            }
            counts = _exposure_count_features(metadata)

            rows.append(
                {
                    "scenario_id": scenario_id,
                    "query_index": query_counts[agent],
                    "scaffold_id": scaffold_id,
                    "exposer_ids": json.dumps(sorted(exposers)),
                    "exposer_stances": json.dumps([item["stance"] for item in metadata]),
                    "exposer_comment_ids": json.dumps([item["comment_id"] for item in metadata]),
                    "exposure_metadata": json.dumps(metadata, sort_keys=True),
                    "believe": float(parsed["believe"]),
                    "reshare": float(parsed["reshare"]),
                    "attend": float(parsed["attend"]),
                    "reason_code": parsed["reason_code"],
                    "sampled_reshare": sampled_reshare,
                    **counts,
                    **_scenario_features(net, agent, exposers, cond, step),
                    **(scenario_metadata or {}),
                }
            )

        reshared |= newly_reshared
        heard_history.append(set(heard))
        reshared_history.append(set(reshared))

    return ScenarioResult(scenario_id, rows, heard_history, reshared_history, result_script)


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    net: CampusNetwork
    cond: Condition
    seeds: tuple[int, ...]
    scaffold_id: int = 0
    mechanism: str = "oracle"


def run_batch(
    scenarios: list[ScenarioSpec],
    client: LLMClient,
    concurrency: int = 8,
    run_id: str = "batch",
    seed: int = 42,
    log_dir: str | Path | None = None,
) -> list[ScenarioResult]:
    """Run scenarios sequentially for deterministic tests; concurrency is reserved for live runs."""

    results = []
    for idx, scenario in enumerate(scenarios):
        results.append(
            run_oracle_scenario(
                scenario.net,
                scenario.cond,
                scenario.seeds,
                client,
                rng=np.random.default_rng(seed + idx),
                scenario_id=scenario.scenario_id,
                run_id=run_id,
                log_dir=log_dir,
                scaffold_id=scenario.scaffold_id,
            )
        )
    return results


def make_noise_floor_set(seed: int = 42, count: int = 40) -> list[ScenarioSpec]:
    rng = np.random.default_rng(seed)
    specs: list[ScenarioSpec] = []
    for i in range(count):
        net = generate_campus_network(50, seed + i)
        cond = sample_condition(rng)
        seeds = tuple(sorted(rng.choice(list(net.G.nodes()), size=3, replace=False).tolist()))
        for scaffold_id in range(5):
            specs.append(ScenarioSpec(f"noise_{i}", net, cond, seeds, scaffold_id=scaffold_id))
    return specs


def make_semantic_twin_set(seed: int = 100, count: int = 18) -> list[tuple[ScenarioSpec, ScenarioSpec]]:
    """Build uniform cross-framing twins: every framing combination appears equally."""

    framing_pairs = list(combinations(FRAMING_TYPES, 2))
    if count <= 0 or count % len(framing_pairs) != 0:
        raise ValueError(f"Twin count must be a positive multiple of {len(framing_pairs)}")
    rng = np.random.default_rng(seed)
    schedule = framing_pairs * (count // len(framing_pairs))
    rng.shuffle(schedule)
    pairs: list[tuple[ScenarioSpec, ScenarioSpec]] = []
    for i, (framing_a, framing_b) in enumerate(schedule):
        net = generate_campus_network(50, seed + i)
        sampled = sample_condition(rng)
        topic, tier, _ = sampled.wording_id.split(":")
        variant_a = FRAMED_TEMPLATE_BANK[topic][tier][framing_a]
        variant_b = FRAMED_TEMPLATE_BANK[topic][tier][framing_b]
        cond = Condition(
            topic=sampled.topic,
            urgency=sampled.urgency,
            credibility=sampled.credibility,
            controversy=sampled.controversy,
            text=str(variant_a["text"]),
            wording_id=f"{topic}:{tier}:{framing_a}",
            framing_type=framing_a,
        )
        twin = Condition(
            topic=sampled.topic,
            urgency=sampled.urgency,
            credibility=sampled.credibility,
            controversy=sampled.controversy,
            text=str(variant_b["text"]),
            wording_id=f"{topic}:{tier}:{framing_b}",
            framing_type=framing_b,
        )
        seeds = tuple(sorted(rng.choice(list(net.G.nodes()), size=3, replace=False).tolist()))
        mechanism = str(rng.choice(("ic", "lt", "cc")))
        pairs.append(
            (
                ScenarioSpec(f"twin_{i}_a", net, cond, seeds, mechanism=mechanism),
                ScenarioSpec(f"twin_{i}_b", net, twin, seeds, mechanism=mechanism),
            )
        )
    return pairs
