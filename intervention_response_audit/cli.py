"""Command-line entry points for the Intervention Response Audit Phase 1 workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from intervention_response_audit.cascade import TOPIC_CHANNEL_TABLE, mc_prob_trajectory
from intervention_response_audit.config import resolve_arm_settings, settings
from intervention_response_audit.network import FRAMING_TYPES, TOPICS, CampusNetwork, Condition, generate_campus_network, sample_condition
from intervention_response_audit.oracle.client import (
    AnthropicClient,
    BudgetExceeded,
    BudgetGuard,
    MockLLMClient,
    OpenAIClient,
    complete_json_logged,
    load_completed_calls,
)
from intervention_response_audit.oracle.prompts import SCAFFOLDS, build_decision_prompt, sample_share_metadata
from intervention_response_audit.oracle.runner import (
    ScenarioSpec,
    make_noise_floor_set,
    make_semantic_twin_set,
    run_oracle_scenario,
)

FULL_RUN_IDS = {"phase1_full_v1", "phase1_floor_v1", "phase1_twins_v1"}
ARM_A_COLLECTION_IDS = {"armA_full_v1", "armA_floor_v1", "armA_twins_v1"}
ARM_A_RUN_IDS = {"armA_micro_v1", "armA_micro_v2", *ARM_A_COLLECTION_IDS}
ARM_B_COLLECTION_IDS = {"armB_full_v1", "armB_floor_v1", "armB_twins_v1"}
ARM_B_RUN_IDS = {"armB_micro_v1", *ARM_B_COLLECTION_IDS}
FLOOR_DIAGNOSTIC_RUN_IDS = {"diag_floor_base_v1", "diag_floor_armA_v1"}
ARM_PRICING_USD_PER_1M = {
    "baseline": (settings.input_usd_per_1m, settings.output_usd_per_1m),
    "armA": (settings.input_usd_per_1m, settings.output_usd_per_1m),
    "armB": (2.50, 15.00),
}
MECHANISMS = ("ic", "lt", "cc")


def _data_path(*parts: str) -> Path:
    return settings.data_dir.joinpath(*parts)


def _load_networks() -> list[CampusNetwork]:
    paths = sorted(_data_path("networks").glob("network_*.json"))
    return [CampusNetwork.from_json(path) for path in paths]


def _live_cost_to_date() -> float:
    """Return cumulative exact-usage live spend across all local trial logs."""

    total = 0.0
    if not settings.data_dir.exists():
        return total
    for path in settings.data_dir.rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("usage_source") == "api_response":
                total += float(record.get("call_cost_usd", 0.0))
    return total


def _live_cost_for_run_ids(run_ids: set[str]) -> float:
    total = 0.0
    for run_id in run_ids:
        path = _data_path("oracle_logs", f"{run_id}.jsonl")
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("usage_source") == "api_response":
                total += float(record.get("call_cost_usd", 0.0))
    return total


def _live_budget(run_id: str, max_api_calls: int, arm_id: str | None = None) -> BudgetGuard:
    project_cost = _live_cost_to_date()
    if run_id in ARM_A_RUN_IDS:
        non_arm_cost = project_cost - _live_cost_for_run_ids(ARM_A_RUN_IDS)
        effective_max = non_arm_cost + settings.sweep_arm_max_usd
    elif run_id in ARM_B_RUN_IDS:
        non_arm_cost = project_cost - _live_cost_for_run_ids(ARM_B_RUN_IDS)
        effective_max = non_arm_cost + settings.sweep_arm_b_max_usd
    elif run_id in FLOOR_DIAGNOSTIC_RUN_IDS:
        non_diagnostic_cost = project_cost - _live_cost_for_run_ids(FLOOR_DIAGNOSTIC_RUN_IDS)
        effective_max = non_diagnostic_cost + settings.floor_diagnostic_max_usd
    elif run_id in FULL_RUN_IDS:
        non_full_cost = project_cost - _live_cost_for_run_ids(FULL_RUN_IDS)
        effective_max = min(settings.max_usd, non_full_cost + settings.full_run_max_usd)
    else:
        effective_max = settings.trial_max_usd
    input_price, output_price = ARM_PRICING_USD_PER_1M[arm_id or "baseline"]
    return BudgetGuard(
        max_api_calls=max_api_calls,
        max_usd=effective_max,
        input_usd_per_1m=input_price,
        output_usd_per_1m=output_price,
        cost_usd=project_cost,
    )


def _canonical_exposure_script(script: dict[tuple[int, int], dict[str, object]]) -> str:
    rows = [
        {"step": step, "agent_id": agent, **entry}
        for (step, agent), entry in sorted(script.items())
    ]
    return json.dumps(rows, sort_keys=True, separators=(",", ":"))


def _client(args: argparse.Namespace):
    arm = resolve_arm_settings(getattr(args, "arm_id", None))
    if getattr(args, "mock", False):
        return MockLLMClient(semantic_tweak=args.mock_semantic_tweak)
    if args.provider == "anthropic":
        return AnthropicClient(arm.model, temperature=arm.temperature, max_tokens=arm.max_tokens)
    return OpenAIClient(
        arm.model,
        temperature=arm.temperature,
        max_tokens=arm.max_tokens,
        reasoning_effort=arm.reasoning_effort,
        allow_temperature_fallback=arm.arm_id != "armB",
    )


def gen_networks(args: argparse.Namespace) -> None:
    out = _data_path("networks")
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        net = generate_campus_network(args.n, args.seed + i)
        net.to_json(out / f"network_{i:03d}.json")
    print(f"Wrote {args.count} networks to {out}")


def run_cascades(args: argparse.Namespace) -> None:
    networks = _load_networks() or [generate_campus_network(50, args.seed)]
    mechanisms = [m.strip() for m in args.mechanisms.split(",") if m.strip()]
    rng = np.random.default_rng(args.seed)
    rows = []
    for net_idx, net in enumerate(networks):
        cond = sample_condition(rng)
        seeds = tuple(sorted(rng.choice(list(net.G.nodes()), size=args.seed_count, replace=False).tolist()))
        for mechanism in mechanisms:
            traj = mc_prob_trajectory(net, mechanism, cond, seeds, R=args.R, T=args.T, seed=args.seed + net_idx)
            for step in range(traj.shape[0]):
                for node in range(traj.shape[1]):
                    rows.append(
                        {
                            "network_id": net_idx,
                            "mechanism": mechanism,
                            "step": step,
                            "agent_id": node,
                            "activation_prob": float(traj[step, node]),
                            "topic": cond.topic,
                            "wording_id": cond.wording_id,
                        }
                    )
    out = _data_path("cascades")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "cascade_trajectories.parquet", index=False)
    (out / "topic_channel_table.json").write_text(
        json.dumps(TOPIC_CHANNEL_TABLE, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote cascade trajectories to {out}")


def _default_scenarios(count: int, seed: int) -> list[ScenarioSpec]:
    rng = np.random.default_rng(seed)
    networks = _load_networks()
    if count % len(TOPICS) == 0:
        topics = [topic for topic in TOPICS for _ in range(count // len(TOPICS))]
        rng.shuffle(topics)
    else:
        topics = [str(rng.choice(TOPICS)) for _ in range(count)]
    specs = []
    for i, topic in enumerate(topics):
        net = networks[i % len(networks)] if networks else generate_campus_network(50, seed + i)
        cond = sample_condition(rng, topic=topic)
        seeds = tuple(sorted(rng.choice(list(net.G.nodes()), size=3, replace=False).tolist()))
        mechanism = str(rng.choice(MECHANISMS))
        specs.append(ScenarioSpec(f"scenario_{i:03d}", net, cond, seeds, mechanism=mechanism))
    return specs


def _scenario_spec_fingerprints(specs: list[ScenarioSpec]) -> list[dict[str, object]]:
    """Canonical hashes for scenario-definition identity checks."""

    rows = []
    for spec in specs:
        network_json = spec.net.to_json()
        condition = asdict(spec.cond)
        payload = {
            "scenario_id": spec.scenario_id,
            "network_sha256": hashlib.sha256(network_json.encode("utf-8")).hexdigest(),
            "condition": condition,
            "seeds": list(spec.seeds),
            "scaffold_id": spec.scaffold_id,
            "mechanism": spec.mechanism,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        rows.append({**payload, "spec_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()})
    return rows


def _oracle_scenarios(count: int, seed: int, arm_id: str) -> list[ScenarioSpec]:
    if arm_id in {"armA", "armB"}:
        return _default_scenarios(80, seed)[:count]
    return _default_scenarios(count, seed)


def _endorse_demo_scenarios(count: int, seed: int) -> list[ScenarioSpec]:
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(count):
        net = generate_campus_network(50, seed + 500 + i)
        for persona in net.personas.values():
            persona.topic_interests["internship"] = 0.95
        cond = Condition(
            topic="internship",
            urgency=0.85,
            credibility=0.95,
            controversy=0.20,
            text="The official career office says an alum referral window closes tonight.",
            wording_id=f"demo_endorse:{i}:0",
        )
        seed_nodes = tuple(sorted(rng.choice(list(net.G.nodes()), size=5, replace=False).tolist()))
        specs.append(ScenarioSpec(f"demo_endorse_{i:03d}", net, cond, seed_nodes))
    return specs


def _ordered_parallel_map(func, items: list, concurrency: int) -> list:
    """Run independent work concurrently while returning results in input order."""

    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if concurrency == 1 or len(items) <= 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="intervention-audit-oracle") as executor:
        futures = [executor.submit(func, item) for item in items]
        return [future.result() for future in futures]


def run_oracle(args: argparse.Namespace) -> None:
    arm = resolve_arm_settings(getattr(args, "arm_id", None))
    worker_state = threading.local()

    def worker_client():
        if not hasattr(worker_state, "client"):
            worker_state.client = _client(args)
        return worker_state.client
    log_path = _data_path("oracle_logs", f"{args.run_id}.jsonl")
    completed_before = 0
    if log_path.exists():
        completed_before = sum(1 for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip())
    budget = (
        BudgetGuard(max_api_calls=args.max_api_calls, max_usd=settings.max_usd)
        if args.mock
        else _live_budget(args.run_id, args.max_api_calls, arm.arm_id)
    )
    specs: list[ScenarioSpec] = []
    twin_pairs: list[tuple[ScenarioSpec, ScenarioSpec]] = []
    stance_probs_override = None
    if args.demo_regime == "endorse":
        specs = _endorse_demo_scenarios(args.scenarios, args.seed)
        stance_probs_override = (0.8, 0.15, 0.05)
    elif args.twins_only:
        twin_pairs = make_semantic_twin_set(args.seed, count=args.twin_pairs)
    elif args.special_sets:
        specs.extend(make_noise_floor_set(args.seed))
        twin_pairs = make_semantic_twin_set(args.seed + 1000)
    else:
        specs = _oracle_scenarios(args.scenarios, args.seed, arm.arm_id)

    arm_metadata = {
        "arm_id": arm.arm_id,
        "reference_simulator_id": arm.reference_simulator_id,
    }

    try:
        frames = []

        def run_spec(item):
            i, spec = item
            result = run_oracle_scenario(
                spec.net,
                spec.cond,
                spec.seeds,
                worker_client(),
                T=args.T,
                rng=np.random.default_rng(args.seed + i),
                scenario_id=spec.scenario_id,
                run_id=args.run_id,
                scaffold_id=spec.scaffold_id,
                stance_probs=stance_probs_override,
                budget=budget,
                scenario_metadata={"mechanism": spec.mechanism, **arm_metadata},
            )
            frame = result.to_frame()
            if spec.scenario_id.endswith("_a") or spec.scenario_id.endswith("_b"):
                frame["twin_group"] = spec.scenario_id.rsplit("_", 1)[0]
                frame["twin_side"] = spec.scenario_id.rsplit("_", 1)[1]
            return frame

        frames.extend(
            _ordered_parallel_map(run_spec, list(enumerate(specs)), args.concurrency)
        )

        def run_twin_pair(item):
            i, (left, right) = item
            scalar_identity = (
                left.net is right.net
                and left.seeds == right.seeds
                and left.scaffold_id == right.scaffold_id
                and left.cond.topic == right.cond.topic
                and left.cond.urgency == right.cond.urgency
                and left.cond.credibility == right.cond.credibility
                and left.cond.controversy == right.cond.controversy
            )
            first = run_oracle_scenario(
                left.net,
                left.cond,
                left.seeds,
                worker_client(),
                T=args.T,
                rng=np.random.default_rng(args.seed + 10_000 + i),
                scenario_id=left.scenario_id,
                run_id=args.run_id,
                scaffold_id=left.scaffold_id,
                budget=budget,
                scenario_metadata={"mechanism": left.mechanism, **arm_metadata},
            )
            second = run_oracle_scenario(
                right.net,
                right.cond,
                right.seeds,
                worker_client(),
                T=args.T,
                rng=np.random.default_rng(args.seed + 20_000 + i),
                scenario_id=right.scenario_id,
                run_id=args.run_id,
                scaffold_id=right.scaffold_id,
                exposure_script=first.exposure_script,
                budget=budget,
                scenario_metadata={"mechanism": right.mechanism, **arm_metadata},
            )
            script_identity = _canonical_exposure_script(first.exposure_script) == _canonical_exposure_script(
                second.exposure_script
            )
            identity = {
                    "pair": i + 1,
                    "arm_id": arm.arm_id,
                    "reference_simulator_id": arm.reference_simulator_id,
                    "left_scenario_id": left.scenario_id,
                    "right_scenario_id": right.scenario_id,
                    "left_framing_type": left.cond.framing_type,
                    "right_framing_type": right.cond.framing_type,
                    "message_text_differs": left.cond.text != right.cond.text,
                    "fixed_scalars_network_seeds_identical": scalar_identity,
                    "mechanism_identical": left.mechanism == right.mechanism,
                    "exposure_script_stances_comments_identical": script_identity,
                    "pass": (
                        scalar_identity
                        and script_identity
                        and left.mechanism == right.mechanism
                        and left.cond.text != right.cond.text
                    ),
                }
            pair_frames = []
            for result, side in [(first, "a"), (second, "b")]:
                frame = result.to_frame()
                frame["twin_group"] = result.scenario_id.rsplit("_", 1)[0]
                frame["twin_side"] = side
                pair_frames.append(frame)
            return identity, pair_frames

        twin_outputs = _ordered_parallel_map(
            run_twin_pair, list(enumerate(twin_pairs)), args.concurrency
        )
        twin_identity: list[dict[str, object]] = [identity for identity, _ in twin_outputs]
        for _, pair_frames in twin_outputs:
            frames.extend(pair_frames)

        if twin_pairs:
            identity_path = _data_path("oracle_logs", f"{args.run_id}_twin_identity.json")
            identity_path.parent.mkdir(parents=True, exist_ok=True)
            identity_path.write_text(json.dumps(twin_identity, indent=2, sort_keys=True), encoding="utf-8")
            print(
                f"Twin byte-identity checks: {sum(bool(item['pass']) for item in twin_identity)}/"
                f"{len(twin_identity)} PASS; wrote {identity_path}"
            )

        if specs:
            topic_mechanism = Counter((spec.cond.topic, spec.mechanism) for spec in specs)
            design = {
                "run_id": args.run_id,
                "arm_id": arm.arm_id,
                "reference_simulator_id": arm.reference_simulator_id,
                "scenario_count": len(specs),
                "topic_counts": dict(sorted(Counter(spec.cond.topic for spec in specs).items())),
                "topic_mechanism_counts": {
                    f"{topic}|{mechanism}": topic_mechanism.get((topic, mechanism), 0)
                    for topic in TOPICS
                    for mechanism in MECHANISMS
                },
                "scenario_spec_fingerprints": _scenario_spec_fingerprints(specs),
            }
            design_path = _data_path("oracle_logs", f"{args.run_id}_design.json")
            design_path.parent.mkdir(parents=True, exist_ok=True)
            design_path.write_text(json.dumps(design, indent=2, sort_keys=True), encoding="utf-8")
            print(f"Realized topic x mechanism design: {json.dumps(design['topic_mechanism_counts'], sort_keys=True)}")

    except BudgetExceeded as exc:
        raise SystemExit(f"Budget exceeded cleanly: {exc}") from None

    out = _data_path("oracle_logs")
    out.mkdir(parents=True, exist_ok=True)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    merged.to_parquet(out / f"{args.run_id}.parquet", index=False)
    print(f"Wrote oracle rows to {out / f'{args.run_id}.parquet'}")
    skipped = completed_before
    print(f"Resume stats: skipped={skipped} new={budget.calls}")


def _floor_stratum_allocations(rows: pd.DataFrame, count: int) -> dict[tuple[str, int], int]:
    """Largest-remainder allocation matching topic x exposure proportions."""

    sizes = {
        (str(topic), int(n_exposures)): int(size)
        for (topic, n_exposures), size in rows.groupby(["topic", "n_exposures"]).size().items()
    }
    if count <= 0 or count > sum(sizes.values()):
        raise ValueError("Floor item count must be positive and no larger than the source pool")
    raw = {key: count * size / sum(sizes.values()) for key, size in sizes.items()}
    allocated = {key: min(sizes[key], int(np.floor(value))) for key, value in raw.items()}
    remaining = count - sum(allocated.values())
    order = sorted(sizes, key=lambda key: (-(raw[key] - np.floor(raw[key])), key))
    while remaining:
        progressed = False
        for key in order:
            if allocated[key] < sizes[key]:
                allocated[key] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise RuntimeError("Unable to allocate all floor items")
    return allocated


def _select_floor_points(rows: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    allocations = _floor_stratum_allocations(rows, count)
    rng = np.random.default_rng(seed)
    selected = []
    for (topic, n_exposures), n_items in sorted(allocations.items()):
        if n_items == 0:
            continue
        group = rows[(rows["topic"] == topic) & (rows["n_exposures"].astype(int) == n_exposures)]
        group = group.sort_values(["scenario_id", "step", "agent_id", "query_index"])
        positions = sorted(rng.choice(len(group), size=n_items, replace=False).tolist())
        selected.append(group.iloc[positions])
    return pd.concat(selected, ignore_index=True).sort_values(
        ["topic", "n_exposures", "scenario_id", "step", "agent_id"]
    ).reset_index(drop=True)


def _non_scaffold_prompt(prompt: str) -> str:
    parts = prompt.split("\n\n", 1)
    if len(parts) != 2:
        raise ValueError("Prompt does not contain the expected scaffold separator")
    return parts[1]


def run_floor_live(args: argparse.Namespace) -> None:
    """Collect 40 stratified decision points under all five existing scaffolds."""

    source_parquet = _data_path("oracle_logs", f"{args.source_run_id}.parquet")
    source_jsonl = _data_path("oracle_logs", f"{args.source_run_id}.jsonl")
    if not source_parquet.exists() or not source_jsonl.exists():
        raise SystemExit(f"Missing completed source run: {source_parquet} / {source_jsonl}")
    source_rows = pd.read_parquet(source_parquet)
    selected = _select_floor_points(source_rows, args.items, args.seed)
    source_records = {}
    for line in source_jsonl.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        key = (
            str(record["scenario_id"]),
            int(record["step"]),
            int(record["agent_id"]),
            int(record["scaffold_id"]),
        )
        source_records[key] = record

    log_path = _data_path("oracle_logs", f"{args.run_id}.jsonl")
    completed = load_completed_calls(log_path)
    completed_before = len(completed)
    client = OpenAIClient(
        settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        reasoning_effort=settings.reasoning_effort,
    )
    budget = _live_budget(args.run_id, args.items * len(SCAFFOLDS) * 2)
    output_rows = []
    try:
        for item_index, (_, source_row) in enumerate(selected.iterrows()):
            source_key = (
                str(source_row["scenario_id"]),
                int(source_row["step"]),
                int(source_row["agent_id"]),
                int(source_row["scaffold_id"]),
            )
            source_record = source_records[source_key]
            non_scaffold = _non_scaffold_prompt(str(source_record["full_prompt"]))
            floor_id = f"floor_{item_index:03d}"
            for scaffold_id, scaffold in enumerate(SCAFFOLDS):
                prompt = f"{scaffold}\n\n{non_scaffold}"
                key = (floor_id, int(source_row["step"]), int(source_row["agent_id"]), scaffold_id)
                if key in completed:
                    parsed = completed[key]
                else:
                    parsed = complete_json_logged(
                        client=client,
                        prompt=prompt,
                        log_path=log_path,
                        budget=budget,
                        scenario_id=floor_id,
                        step=int(source_row["step"]),
                        agent_id=int(source_row["agent_id"]),
                        scaffold_id=scaffold_id,
                        extra_log_fields={
                            "exposure_metadata": source_record["exposure_metadata"],
                            "framing_type": source_record["framing_type"],
                            "mechanism": source_record["mechanism"],
                            "floor_item_id": floor_id,
                            "source_scenario_id": source_key[0],
                            "source_n_exposures": int(source_row["n_exposures"]),
                            "source_topic": str(source_row["topic"]),
                        },
                    )
                row = source_row.to_dict()
                row.update(
                    {
                        "scenario_id": floor_id,
                        "scaffold_id": scaffold_id,
                        "believe": float(parsed["believe"]),
                        "reshare": float(parsed["reshare"]),
                        "attend": float(parsed["attend"]),
                        "reason_code": str(parsed["reason_code"]),
                        "floor_item_id": floor_id,
                        "source_scenario_id": source_key[0],
                    }
                )
                output_rows.append(row)
    except BudgetExceeded as exc:
        raise SystemExit(f"Budget exceeded cleanly: {exc}") from None

    out = _data_path("oracle_logs")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_parquet(out / f"{args.run_id}.parquet", index=False)
    source_counts = source_rows.groupby(["topic", "n_exposures"]).size()
    selected_counts = selected.groupby(["topic", "n_exposures"]).size()
    manifest = {
        "run_id": args.run_id,
        "source_run_id": args.source_run_id,
        "items": args.items,
        "scaffolds_per_item": len(SCAFFOLDS),
        "records": len(output_rows),
        "source_strata": {f"{topic}|{int(n)}": int(v) for (topic, n), v in source_counts.items()},
        "selected_strata": {f"{topic}|{int(n)}": int(v) for (topic, n), v in selected_counts.items()},
    }
    manifest_path = out / f"{args.run_id}_design.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Floor set: records={len(output_rows)} skipped={completed_before} new_api_calls={budget.calls}")
    print(f"Floor stratification: {json.dumps(manifest['selected_strata'], sort_keys=True)}")
    print(f"Wrote {out / f'{args.run_id}.parquet'} and {manifest_path}")


def run_floor_replay_live(args: argparse.Namespace) -> None:
    """Replay the exact baseline floor prompts under one sweep-arm overlay."""

    arm = resolve_arm_settings(args.arm_id)
    source_jsonl = _data_path("oracle_logs", f"{args.source_run_id}.jsonl")
    source_parquet = _data_path("oracle_logs", f"{args.source_run_id}.parquet")
    if not source_jsonl.exists() or not source_parquet.exists():
        raise SystemExit(f"Missing baseline floor set: {source_jsonl} / {source_parquet}")
    source_records = [json.loads(line) for line in source_jsonl.read_text(encoding="utf-8").splitlines() if line]
    source_frame = pd.read_parquet(source_parquet)
    if len(source_records) != 200 or len(source_frame) != 200:
        raise SystemExit("Baseline floor set must contain exactly 200 records")
    source_rows = {
        (str(row.scenario_id), int(row.step), int(row.agent_id), int(row.scaffold_id)): row
        for row in source_frame.itertuples(index=False)
    }

    log_path = _data_path("oracle_logs", f"{args.run_id}.jsonl")
    completed = load_completed_calls(log_path)
    completed_before = len(completed)
    worker_state = threading.local()

    def worker_client():
        if not hasattr(worker_state, "client"):
            worker_state.client = OpenAIClient(
                arm.model,
                temperature=arm.temperature,
                max_tokens=arm.max_tokens,
                reasoning_effort=arm.reasoning_effort,
                allow_temperature_fallback=arm.arm_id != "armB",
            )
        return worker_state.client

    budget = _live_budget(args.run_id, args.max_api_calls, arm.arm_id)

    def replay_record(source_record):
        key = (
            str(source_record["scenario_id"]),
            int(source_record["step"]),
            int(source_record["agent_id"]),
            int(source_record["scaffold_id"]),
        )
        if key in completed:
            parsed = completed[key]
        else:
            parsed = complete_json_logged(
                client=worker_client(),
                prompt=str(source_record["full_prompt"]),
                log_path=log_path,
                budget=budget,
                scenario_id=key[0],
                step=key[1],
                agent_id=key[2],
                scaffold_id=key[3],
                extra_log_fields={
                    "exposure_metadata": source_record["exposure_metadata"],
                    "framing_type": source_record["framing_type"],
                    "mechanism": source_record["mechanism"],
                    "floor_item_id": source_record["floor_item_id"],
                    "source_scenario_id": source_record["source_scenario_id"],
                    "source_n_exposures": source_record["source_n_exposures"],
                    "source_topic": source_record["source_topic"],
                    "arm_id": arm.arm_id,
                    "reference_simulator_id": arm.reference_simulator_id,
                },
            )
        row = source_rows[key]._asdict()
        row.update(
            {
                "believe": float(parsed["believe"]),
                "reshare": float(parsed["reshare"]),
                "attend": float(parsed["attend"]),
                "reason_code": str(parsed["reason_code"]),
                "arm_id": arm.arm_id,
                "reference_simulator_id": arm.reference_simulator_id,
            }
        )
        return row

    try:
        output_rows = _ordered_parallel_map(replay_record, source_records, args.concurrency)
    except BudgetExceeded as exc:
        raise SystemExit(f"Budget exceeded cleanly: {exc}") from None

    out = _data_path("oracle_logs")
    pd.DataFrame(output_rows).to_parquet(out / f"{args.run_id}.parquet", index=False)
    prompt_hashes = [hashlib.sha256(r["full_prompt"].encode("utf-8")).hexdigest() for r in source_records]
    manifest = {
        "run_id": args.run_id,
        "source_run_id": args.source_run_id,
        "arm_id": arm.arm_id,
        "reference_simulator_id": arm.reference_simulator_id,
        "records": len(output_rows),
        "source_prompt_set_sha256": hashlib.sha256("".join(prompt_hashes).encode("utf-8")).hexdigest(),
        "source_prompts_reused_byte_identically": True,
    }
    manifest_path = out / f"{args.run_id}_design.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Replayed floor set: records={len(output_rows)} skipped={completed_before} new_api_calls={budget.calls}")
    print(f"Wrote {out / f'{args.run_id}.parquet'} and {manifest_path}")


def _load_completed_floor_repeats(path: Path) -> dict[tuple[str, int], dict[str, object]]:
    """Load valid diagnostic repeats keyed by floor item and explicit repeat index."""

    if not path.exists():
        return {}
    completed: dict[tuple[str, int], dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        if record.get("valid") and record.get("parsed_output") is not None:
            key = (str(record["floor_item_id"]), int(record["repeat_index"]))
            if key in completed:
                raise ValueError(f"Duplicate diagnostic repeat key: {key}")
            completed[key] = record
    return completed


def run_floor_repeat_live(args: argparse.Namespace) -> None:
    """Repeat byte-identical scaffold-0 floor prompts under one frozen configuration."""

    authorized = {
        ("phase1_floor_v1", "diag_floor_base_v1", "baseline", 10, 3),
        ("armA_floor_v1", "diag_floor_armA_v1", "armA", 20, 5),
    }
    request = (args.source_run_id, args.run_id, args.arm_id, args.items, args.repeats)
    if request not in authorized:
        raise SystemExit(f"Unauthorized floor diagnostic configuration: {request}")

    arm = resolve_arm_settings(args.arm_id)
    source_jsonl = _data_path("oracle_logs", f"{args.source_run_id}.jsonl")
    source_parquet = _data_path("oracle_logs", f"{args.source_run_id}.parquet")
    if not source_jsonl.exists() or not source_parquet.exists():
        raise SystemExit(f"Missing source floor set: {source_jsonl} / {source_parquet}")

    source_frame = pd.read_parquet(source_parquet)
    scaffold_zero = source_frame[source_frame["scaffold_id"].astype(int) == 0].copy()
    if len(scaffold_zero) != 40:
        raise SystemExit(f"Expected 40 scaffold-0 floor items, found {len(scaffold_zero)}")
    selected = _select_floor_points(scaffold_zero, args.items, args.seed)
    selected_ids = set(selected["floor_item_id"].astype(str))

    source_records = {}
    for line in source_jsonl.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record["scaffold_id"]) == 0 and str(record["floor_item_id"]) in selected_ids:
            source_records[str(record["floor_item_id"])] = record
    if set(source_records) != selected_ids:
        raise SystemExit("Selected floor items are not fully represented in the source JSONL")

    log_path = _data_path("oracle_logs", f"{args.run_id}.jsonl")
    completed = _load_completed_floor_repeats(log_path)
    completed_before = len(completed)
    client = OpenAIClient(
        arm.model,
        temperature=arm.temperature,
        max_tokens=arm.max_tokens,
        reasoning_effort=arm.reasoning_effort,
    )
    budget = _live_budget(args.run_id, args.items * args.repeats, arm.arm_id)
    try:
        for source_row in selected.itertuples(index=False):
            floor_item_id = str(source_row.floor_item_id)
            source_record = source_records[floor_item_id]
            for repeat_index in range(args.repeats):
                key = (floor_item_id, repeat_index)
                if key in completed:
                    continue
                complete_json_logged(
                    client=client,
                    prompt=str(source_record["full_prompt"]),
                    log_path=log_path,
                    budget=budget,
                    scenario_id=str(source_record["scenario_id"]),
                    step=int(source_record["step"]),
                    agent_id=int(source_record["agent_id"]),
                    scaffold_id=0,
                    extra_log_fields={
                        "repeat_index": repeat_index,
                        "floor_item_id": floor_item_id,
                        "source_run_id": args.source_run_id,
                        "source_topic": str(source_row.topic),
                        "source_n_exposures": int(source_row.n_exposures),
                        "exposure_metadata": source_record["exposure_metadata"],
                        "framing_type": source_record["framing_type"],
                        "mechanism": source_record["mechanism"],
                        "arm_id": arm.arm_id,
                        "reference_simulator_id": arm.reference_simulator_id,
                    },
                )
    except BudgetExceeded as exc:
        raise SystemExit(f"Budget exceeded cleanly: {exc}") from None

    completed = _load_completed_floor_repeats(log_path)
    expected_keys = {(item, repeat) for item in selected_ids for repeat in range(args.repeats)}
    if set(completed) != expected_keys:
        raise SystemExit(f"Incomplete diagnostic run: {len(completed)}/{len(expected_keys)} repeats")

    source_by_id = {str(row.floor_item_id): row._asdict() for row in selected.itertuples(index=False)}
    output_rows = []
    for (floor_item_id, repeat_index), record in sorted(completed.items()):
        row = source_by_id[floor_item_id].copy()
        parsed = record["parsed_output"]
        row.update(
            {
                "repeat_index": repeat_index,
                "believe": float(parsed["believe"]),
                "reshare": float(parsed["reshare"]),
                "attend": float(parsed["attend"]),
                "reason_code": str(parsed["reason_code"]),
                "arm_id": arm.arm_id,
                "reference_simulator_id": arm.reference_simulator_id,
            }
        )
        output_rows.append(row)

    out = _data_path("oracle_logs")
    pd.DataFrame(output_rows).to_parquet(out / f"{args.run_id}.parquet", index=False)
    prompt_hashes = {
        item: hashlib.sha256(str(source_records[item]["full_prompt"]).encode("utf-8")).hexdigest()
        for item in sorted(selected_ids)
    }
    manifest = {
        "run_id": args.run_id,
        "source_run_id": args.source_run_id,
        "arm_id": arm.arm_id,
        "reference_simulator_id": arm.reference_simulator_id,
        "items": args.items,
        "repeats_per_item": args.repeats,
        "records": len(output_rows),
        "repeat_index_logged": True,
        "source_prompts_reused_byte_identically": True,
        "selected_prompt_sha256": prompt_hashes,
        "selected_strata": {
            f"{topic}|{int(n)}": int(size)
            for (topic, n), size in selected.groupby(["topic", "n_exposures"]).size().items()
        },
    }
    manifest_path = out / f"{args.run_id}_design.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"Floor repeats: records={len(output_rows)} skipped={completed_before} "
        f"new_api_calls={budget.calls} run_cost=${_live_cost_for_run_ids({args.run_id}):.8f}"
    )
    print(f"Selected strata: {json.dumps(manifest['selected_strata'], sort_keys=True)}")
    print(f"Wrote {out / f'{args.run_id}.parquet'} and {manifest_path}")


def run_floor_decomposition_cmd(args: argparse.Namespace) -> None:
    """Run the pre-registered offline floor decomposition and archive its report."""

    from intervention_response_audit.floor_diagnostic import run_floor_decomposition

    registration_path = _data_path("floor_diag_pre_registration.json")
    if not registration_path.exists():
        raise SystemExit(f"Missing diagnostic pre-registration manifest: {registration_path}")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    protected_hashes = registration["protected_file_sha256"]
    protected_unchanged = all(
        hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        for path, expected in protected_hashes.items()
    )
    baseline_cost = _live_cost_for_run_ids({"diag_floor_base_v1"})
    arm_cost = _live_cost_for_run_ids({"diag_floor_armA_v1"})
    result = run_floor_decomposition(
        baseline_floor=_data_path("oracle_logs", "phase1_floor_v1.parquet"),
        arm_floor=_data_path("oracle_logs", "armA_floor_v1.parquet"),
        baseline_repeats=_data_path("oracle_logs", "diag_floor_base_v1.parquet"),
        arm_repeats=_data_path("oracle_logs", "diag_floor_armA_v1.parquet"),
        output_dir=_data_path("audit", "diag_floor_v1"),
        metadata={
            "pre_registration_commit": registration["commit"],
            "pytest": registration["pytest"],
            "baseline_cost_usd": baseline_cost,
            "arm_cost_usd": arm_cost,
            "total_cost_usd": baseline_cost + arm_cost,
            "protected_files_unchanged": protected_unchanged,
        },
    )
    print(result["report"])


def _verify_sweep_arm_specs(args: argparse.Namespace, arm_id: str) -> None:
    """Verify one sweep arm reuses baseline normal, floor, and twin definitions."""

    specs = _default_scenarios(80, args.seed)
    fingerprints = _scenario_spec_fingerprints(specs)
    baseline_frame = pd.read_parquet(_data_path("oracle_logs", "phase1_full_v1.parquet"))
    baseline_scenarios = baseline_frame.sort_values(["scenario_id", "step", "agent_id"]).groupby("scenario_id").first()
    normal_fields_match = True
    for spec in specs:
        row = baseline_scenarios.loc[spec.scenario_id]
        normal_fields_match &= (
            str(row["topic"]) == spec.cond.topic
            and float(row["urgency"]) == spec.cond.urgency
            and float(row["credibility"]) == spec.cond.credibility
            and float(row["controversy"]) == spec.cond.controversy
            and str(row["wording_id"]) == spec.cond.wording_id
            and str(row["condition_text"]) == spec.cond.text
            and str(row["mechanism"]) == spec.mechanism
        )

    twin_a = make_semantic_twin_set(args.seed, count=18)
    twin_b = make_semantic_twin_set(args.seed, count=18)
    twin_specs_a = [spec for pair in twin_a for spec in pair]
    twin_specs_b = [spec for pair in twin_b for spec in pair]
    twin_fingerprints_a = _scenario_spec_fingerprints(twin_specs_a)
    twin_fingerprints_b = _scenario_spec_fingerprints(twin_specs_b)
    twin_specs_match = twin_fingerprints_a == twin_fingerprints_b

    floor_records = [
        json.loads(line)
        for line in _data_path("oracle_logs", "phase1_floor_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    floor_prompt_hashes = [hashlib.sha256(r["full_prompt"].encode("utf-8")).hexdigest() for r in floor_records]
    manifest = {
        "arm_id": arm_id,
        "normal_scenarios": len(specs),
        "normal_fields_match_baseline_parquet": bool(normal_fields_match),
        "normal_spec_set_sha256": hashlib.sha256(
            "".join(str(row["spec_sha256"]) for row in fingerprints).encode("utf-8")
        ).hexdigest(),
        "floor_records": len(floor_records),
        "floor_prompt_set_sha256": hashlib.sha256("".join(floor_prompt_hashes).encode("utf-8")).hexdigest(),
        "floor_replay_is_byte_exact": len(floor_records) == 200,
        "twin_pairs": len(twin_a),
        "twin_specs_deterministic": twin_specs_match,
        "twin_spec_set_sha256": hashlib.sha256(
            "".join(str(row["spec_sha256"]) for row in twin_fingerprints_a).encode("utf-8")
        ).hexdigest(),
        "pass": bool(normal_fields_match and len(floor_records) == 200 and twin_specs_match),
    }
    path = _data_path("oracle_logs", f"{arm_id}_spec_identity.json")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not manifest["pass"]:
        raise SystemExit(f"{arm_id} scenario identity verification failed")


def verify_arm_a_specs(args: argparse.Namespace) -> None:
    _verify_sweep_arm_specs(args, "armA")


def verify_arm_b_specs(args: argparse.Namespace) -> None:
    _verify_sweep_arm_specs(args, "armB")


def smoke_live(args: argparse.Namespace) -> None:
    """Send exactly one normal decision prompt to the pinned OpenAI snapshot."""

    log_path = _data_path("oracle_logs", "smoke_live.jsonl")
    if log_path.exists() and log_path.stat().st_size:
        raise SystemExit(f"Refusing an accidental second smoke call; existing log: {log_path}")

    rng = np.random.default_rng(args.seed)
    networks = _load_networks()
    net = networks[0] if networks else generate_campus_network(50, args.seed)
    cond = sample_condition(rng)
    agent, exposer = next(iter(sorted(net.G.edges())))
    metadata = sample_share_metadata(
        cond.topic,
        rng,
        (
            settings.stance_endorse_prob,
            settings.stance_neutral_prob,
            settings.stance_doubt_prob,
        ),
    )
    prompt = build_decision_prompt(
        net,
        agent,
        {exposer},
        cond,
        scaffold_id=0,
        exposure_details={exposer: metadata},
    )
    client = OpenAIClient(
        settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        reasoning_effort=settings.reasoning_effort,
    )
    budget = BudgetGuard(
        max_api_calls=2,
        max_usd=settings.trial_max_usd,
        input_usd_per_1m=settings.input_usd_per_1m,
        output_usd_per_1m=settings.output_usd_per_1m,
        cost_usd=_live_cost_to_date(),
    )
    parsed = complete_json_logged(
        client=client,
        prompt=prompt,
        log_path=log_path,
        budget=budget,
        scenario_id="smoke_live",
        step=1,
        agent_id=agent,
        scaffold_id=0,
        extra_log_fields={
            "exposure_metadata": [{"agent_id": exposer, **metadata}],
            "framing_type": cond.framing_type,
        },
    )
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    print(f"output: {json.dumps(parsed, sort_keys=True)}")
    print(
        "tokens in/out: "
        f"{record['token_counts']['prompt']}/{record['token_counts']['completion']}, "
        f"reasoning={record['token_counts']['reasoning']}"
    )
    print(f"latency: {record['latency']:.6f}s, cost: ${record['call_cost_usd']:.8f}")
    print(
        f"temperature_accepted={client.temperature_accepted} "
        f"reasoning_effort={client.reasoning_effort} json_mode={client.response_format['type']}"
    )


def run_canary_live(args: argparse.Namespace) -> None:
    """Run or resume 15 deterministic, single-scaffold live decision prompts."""

    out = _data_path("canary", "baseline_v1")
    log_path = out / "records.jsonl"
    completed = load_completed_calls(log_path)
    client = OpenAIClient(
        settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        reasoning_effort=settings.reasoning_effort,
    )
    budget = BudgetGuard(
        max_api_calls=args.count * 2,
        max_usd=settings.trial_max_usd,
        input_usd_per_1m=settings.input_usd_per_1m,
        output_usd_per_1m=settings.output_usd_per_1m,
        cost_usd=_live_cost_to_date(),
    )
    specs = _default_scenarios(args.count, args.seed)
    skipped = 0
    for i, spec in enumerate(specs):
        scenario_id = f"canary_{i:03d}"
        key_candidates = []
        for exposer in sorted(spec.seeds):
            for agent in sorted(spec.net.G.neighbors(exposer)):
                if agent not in spec.seeds:
                    key_candidates.append((agent, exposer))
        if not key_candidates:
            raise RuntimeError(f"No canary decision point available for {scenario_id}")
        agent, exposer = key_candidates[0]
        key = (scenario_id, 1, agent, 0)
        if key in completed:
            skipped += 1
            continue
        rng = np.random.default_rng(args.seed + i)
        metadata = sample_share_metadata(
            spec.cond.topic,
            rng,
            (
                settings.stance_endorse_prob,
                settings.stance_neutral_prob,
                settings.stance_doubt_prob,
            ),
        )
        prompt = build_decision_prompt(
            spec.net,
            agent,
            {exposer},
            spec.cond,
            scaffold_id=0,
            exposure_details={exposer: metadata},
        )
        complete_json_logged(
            client=client,
            prompt=prompt,
            log_path=log_path,
            budget=budget,
            scenario_id=scenario_id,
            step=1,
            agent_id=agent,
            scaffold_id=0,
            extra_log_fields={
                "exposure_metadata": [{"agent_id": exposer, **metadata}],
                "framing_type": spec.cond.framing_type,
            },
        )
    manifest = {
        "model": settings.model,
        "count": args.count,
        "seed": args.seed,
        "scaffold_id": 0,
        "records_path": str(log_path),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Canary baseline: skipped={skipped} new_api_calls={budget.calls}; wrote {out}")
    print(f"Cumulative live cost: ${budget.cost_usd:.8f}")


def _per_combination_twins(rows: pd.DataFrame) -> pd.DataFrame:
    twin_rows = rows[rows.get("twin_side", pd.Series(index=rows.index, dtype=object)).isin(["a", "b"])].copy()
    key_cols = ["twin_group", "step", "agent_id", "query_index"]
    left = twin_rows[twin_rows["twin_side"] == "a"]
    right = twin_rows[twin_rows["twin_side"] == "b"]
    merged = left.merge(right, on=key_cols, suffixes=("_a", "_b"))
    order = {frame: index for index, frame in enumerate(FRAMING_TYPES)}
    pair_rows = []
    for twin_group, group in merged.groupby("twin_group"):
        framing_a = str(group["framing_type_a"].iloc[0])
        framing_b = str(group["framing_type_b"].iloc[0])
        first, second = sorted((framing_a, framing_b), key=order.get)
        pair_rows.append(
            {
                "twin_group": twin_group,
                "combo": f"{first}-{second}",
                "believe": float(np.mean(np.abs(group["believe_a"] - group["believe_b"]))),
                "reshare": float(np.mean(np.abs(group["reshare_a"] - group["reshare_b"]))),
            }
        )
    pair_frame = pd.DataFrame(pair_rows)
    combo_order = [f"{a}-{b}" for a, b in combinations(FRAMING_TYPES, 2)]
    summary = pair_frame.groupby("combo").agg(n=("twin_group", "nunique"), believe=("believe", "mean"), reshare=("reshare", "mean"))
    return summary.reindex(combo_order).reset_index()


def _plot_per_combination_twins(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(summary))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, summary["believe"], width, label="believe")
    ax.bar(x + width / 2, summary["reshare"], width, label="reshare")
    ax.set_xticks(x, summary["combo"], rotation=25, ha="right")
    ax.set_ylabel("Mean |delta p| across pairs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_audit_cmd(args: argparse.Namespace) -> None:
    from intervention_response_audit.audit import _markdown_table, _scenario_split, run_audit

    if args.run_id:
        run_ids = [item.strip() for item in args.run_id.split(",") if item.strip()]
        paths = [_data_path("oracle_logs", f"{run_id}.parquet") for run_id in run_ids]
    else:
        paths = sorted(_data_path("oracle_logs").glob("*.parquet"))
    if not paths:
        raise SystemExit("No oracle parquet files found. Run run-oracle first.")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing oracle parquet files: {missing}")
    for path in paths:
        print(f"Consuming run_id={path.stem} path={path} mtime={path.stat().st_mtime}")
    rows = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    result = run_audit(rows, output_dir=_data_path("audit"), seed=args.seed, fit_gbm=args.gbm)
    combo_summary = _per_combination_twins(rows)
    combo_plot = _data_path("audit", "twins_by_framing.png")
    _plot_per_combination_twins(combo_summary, combo_plot)
    print(
        result.metrics[
            ["target", "NLL_mech", "NLL_floor", "Gap", "twins_oracle_abs_delta", "twins_mech_abs_delta"]
        ].to_string(index=False)
    )
    _, test_mask = _scenario_split(rows.reset_index(drop=True), args.seed)
    floor_mask = rows.groupby(["scenario_id", "step", "agent_id"])["scaffold_id"].transform("nunique") >= 2
    eval_rows = rows.reset_index(drop=True).loc[test_mask]
    floor_rows = rows.loc[floor_mask.to_numpy()]
    if not floor_rows.empty and not eval_rows.empty:
        print(
            "Comparability floor_vs_eval "
            f"mean_step={floor_rows['step'].mean():.3f}/{eval_rows['step'].mean():.3f} "
            f"mean_n_exposures={floor_rows['n_exposures'].mean():.3f}/{eval_rows['n_exposures'].mean():.3f}"
        )
    print("Per-combination twins (pair-weighted mean |delta p|):")
    print(combo_summary.to_string(index=False))
    report_appendix = [
        "",
        "## Per-combination semantic twins",
        "",
        _markdown_table(combo_summary),
        "",
        f"![Per-combination twins]({combo_plot.name})",
        "",
    ]
    with result.report_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(report_appendix))
    selected_run_ids = set(run_ids if args.run_id else [])
    archive_name = None
    if selected_run_ids == FULL_RUN_IDS:
        archive_name = "phase1_v1"
    elif selected_run_ids == ARM_A_COLLECTION_IDS:
        archive_name = "armA_v1"
    elif selected_run_ids == ARM_B_COLLECTION_IDS:
        archive_name = "armB_v1"
    if archive_name:
        archive = _data_path("audit", archive_name)
        archive.mkdir(parents=True, exist_ok=True)
        for path in (result.report_path, result.nll_plot_path, result.twin_plot_path, combo_plot):
            shutil.copy2(path, archive / path.name)
        shutil.copy2(Path("microbenchmark.json"), archive / "microbenchmark.json")
        print(f"Archived audit artifacts to {archive}")
    print(f"{result.decision}: wrote {result.report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m intervention_response_audit.cli")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("gen-networks")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--seed", type=int, default=settings.seed)
    p.set_defaults(func=gen_networks)

    p = sub.add_parser("run-cascades")
    p.add_argument("--mechanisms", default="ic,lt,cc")
    p.add_argument("--R", type=int, default=300)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--seed", type=int, default=settings.seed)
    p.add_argument("--seed-count", type=int, default=3)
    p.set_defaults(func=run_cascades)

    p = sub.add_parser("run-oracle")
    p.add_argument("--scenarios", type=int, default=40)
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--seed", type=int, default=settings.seed)
    p.add_argument("--run-id", default="mock_run")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--mock-semantic-tweak", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--special-sets", action="store_true")
    p.add_argument("--twins-only", action="store_true")
    p.add_argument("--twin-pairs", type=int, default=18)
    p.add_argument("--demo-regime", choices=("endorse",), default=None)
    p.add_argument("--max-api-calls", type=int, default=settings.max_api_calls)
    p.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    p.add_argument("--model", default=settings.model)
    p.add_argument("--arm-id", choices=("baseline", "armA", "armB"), default=settings.arm_id)
    p.add_argument("--concurrency", type=int, default=8)
    p.set_defaults(func=run_oracle)

    p = sub.add_parser("run-audit")
    p.add_argument("--seed", type=int, default=settings.seed)
    p.add_argument("--gbm", action="store_true")
    p.add_argument("--run-id", default=None)
    p.set_defaults(func=run_audit_cmd)

    p = sub.add_parser("smoke-live")
    p.add_argument("--seed", type=int, default=settings.seed)
    p.set_defaults(func=smoke_live)

    p = sub.add_parser("run-canary-live")
    p.add_argument("--count", type=int, default=15)
    p.add_argument("--seed", type=int, default=settings.seed)
    p.set_defaults(func=run_canary_live)

    p = sub.add_parser("run-floor-live")
    p.add_argument("--source-run-id", default="phase1_full_v1")
    p.add_argument("--run-id", default="phase1_floor_v1")
    p.add_argument("--items", type=int, default=40)
    p.add_argument("--seed", type=int, default=settings.seed)
    p.set_defaults(func=run_floor_live)

    p = sub.add_parser("run-floor-replay-live")
    p.add_argument("--source-run-id", default="phase1_floor_v1")
    p.add_argument("--run-id", default="armA_floor_v1")
    p.add_argument("--arm-id", choices=("armA", "armB"), default="armA")
    p.add_argument("--max-api-calls", type=int, default=400)
    p.add_argument("--concurrency", type=int, default=8)
    p.set_defaults(func=run_floor_replay_live)

    p = sub.add_parser("run-floor-repeat-live")
    p.add_argument("--source-run-id", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--arm-id", choices=("baseline", "armA"), required=True)
    p.add_argument("--items", type=int, required=True)
    p.add_argument("--repeats", type=int, required=True)
    p.add_argument("--seed", type=int, default=settings.seed)
    p.set_defaults(func=run_floor_repeat_live)

    p = sub.add_parser("run-floor-decomposition")
    p.set_defaults(func=run_floor_decomposition_cmd)

    p = sub.add_parser("verify-arm-a-specs")
    p.add_argument("--seed", type=int, default=settings.seed)
    p.set_defaults(func=verify_arm_a_specs)
    p = sub.add_parser("verify-arm-b-specs")
    p.add_argument("--seed", type=int, default=settings.seed)
    p.set_defaults(func=verify_arm_b_specs)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
