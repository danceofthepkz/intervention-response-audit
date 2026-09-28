"""Explicitly authorized live gates for v2 smoke and pilot collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from intervention_response_audit.v2.analysis import analyze_confirmatory_reshare, analyze_pilot_k
from intervention_response_audit.v2.collect import (
    CollectorLimits,
    CompletionProvider,
    collect_tasks,
    read_attempt_log,
)
from intervention_response_audit.v2.integrity import verify_stage0_directory
from intervention_response_audit.v2.openai_provider import OpenAIChatProvider, TokenPricing
from intervention_response_audit.v2.report import confirmatory_result_text, pilot_decision_text
from intervention_response_audit.v2.schema import (
    PromptRecord,
    StateRecord,
    TaskRecord,
    V2FrozenConfig,
    canonical_json,
    write_json,
    write_jsonl,
)


SMOKE_AUTHORIZATION = "I_AUTHORIZE_4_LIVE_SMOKE_CALLS"
PILOT_AUTHORIZATION = "I_AUTHORIZE_200_LIVE_PILOT_CALLS"
CONFIRMATORY_AUTHORIZATION = "I_AUTHORIZE_FROZEN_CONFIRMATORY_CALLS"


def _load_jsonl(path: str | Path, model_type):
    return tuple(
        model_type.model_validate_json(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def load_config(stage0_dir: str | Path) -> V2FrozenConfig:
    return V2FrozenConfig.model_validate_json(
        (Path(stage0_dir) / "frozen_config.json").read_text(encoding="utf-8")
    )


def load_prompts(stage0_dir: str | Path) -> tuple[PromptRecord, ...]:
    return _load_jsonl(Path(stage0_dir) / "prompt_corpus.jsonl", PromptRecord)


def load_pilot_schedule(pilot_dir: str | Path) -> tuple[TaskRecord, ...]:
    return _load_jsonl(Path(pilot_dir) / "pilot_schedule.jsonl", TaskRecord)


def load_confirmatory_schedule(confirmatory_dir: str | Path) -> tuple[TaskRecord, ...]:
    return _load_jsonl(Path(confirmatory_dir) / "confirmatory_schedule.jsonl", TaskRecord)


def load_states(stage0_dir: str | Path) -> tuple[StateRecord, ...]:
    return _load_jsonl(Path(stage0_dir) / "state_allocation.jsonl", StateRecord)


def build_smoke_schedule(pilot_tasks: tuple[TaskRecord, ...]) -> tuple[TaskRecord, ...]:
    """Use two exact repeats per side for one frozen pilot state (four calls)."""

    if len(pilot_tasks) != 200:
        raise ValueError("Smoke schedule requires the frozen 200-slot pilot manifest")
    first_state = pilot_tasks[0].state_id
    by_key = {
        (task.condition, task.replicate_index): task
        for task in pilot_tasks
        if task.state_id == first_state
    }
    order = (("authority", 1), ("hedge", 1), ("authority", 2), ("hedge", 2))
    result = []
    for schedule_index, (condition, replicate_index) in enumerate(order):
        source = by_key[(condition, replicate_index)]
        result.append(
            TaskRecord(
                phase="smoke",
                slot_id=f"smoke:{source.state_id}:{condition}:{replicate_index}",
                state_id=source.state_id,
                scenario_id=source.scenario_id,
                condition=condition,
                replicate_index=replicate_index,
                schedule_index=schedule_index,
                dispatch_group=0,
                prompt_sha256=source.prompt_sha256,
                config_sha256=source.config_sha256,
                schedule_seed=source.schedule_seed,
            )
        )
    return tuple(result)


def freeze_smoke_schedule(
    pilot_dir: str | Path = "prereg/v2/pilot",
    prelive_dir: str | Path = "prereg/v2/prelive",
) -> tuple[TaskRecord, ...]:
    target = Path(prelive_dir) / "smoke_schedule.jsonl"
    if target.exists():
        frozen = _load_jsonl(target, TaskRecord)
        expected = build_smoke_schedule(load_pilot_schedule(pilot_dir))
        if frozen != expected:
            raise RuntimeError("Frozen smoke schedule differs from the pilot manifest")
        return frozen
    tasks = build_smoke_schedule(load_pilot_schedule(pilot_dir))
    write_jsonl(target, tasks)
    return tasks


def run_smoke(
    provider: CompletionProvider,
    *,
    stage0_dir: str | Path = "prereg/v2/stage0",
    pilot_dir: str | Path = "prereg/v2/pilot",
    prelive_dir: str | Path = "prereg/v2/prelive",
    results_dir: str | Path = "results/v2/smoke",
) -> dict:
    """Collect and mechanically assess exactly four non-analytic smoke calls."""

    stage0 = verify_stage0_directory(stage0_dir)
    if stage0.status != "PASS":
        raise RuntimeError(f"Stage-0 integrity is {stage0.status}")
    config = load_config(stage0_dir)
    tasks = freeze_smoke_schedule(pilot_dir, prelive_dir)
    destination = Path(results_dir)
    destination.mkdir(parents=True, exist_ok=True)
    acceptance_path = destination / "smoke_acceptance.json"
    if acceptance_path.exists():
        raise FileExistsError("Refusing to rerun an already assessed live smoke")
    summary = collect_tasks(
        tasks,
        load_prompts(stage0_dir),
        provider,
        config,
        destination / "attempts.jsonl",
        destination / "halts.jsonl",
        CollectorLimits(
            max_attempts=4,
            max_retry_attempts=0,
            max_retry_fraction=0.0,
            max_cost_usd=config.max_live_usd,
            preflight_cost_reserve_usd=0.001,
        ),
    )
    attempts = read_attempt_log(destination / "attempts.jsonl")
    checks = {
        "exactly_four_valid_calls": summary.accepted_slots == summary.attempts == 4,
        "unique_response_ids": len({item.response_id for item in attempts}) == 4,
        "unique_usage_events": len({item.usage_event_id for item in attempts}) == 4,
        "exact_snapshot": all(
            item.config_sha256 == tasks[0].config_sha256
            and item.response_model_snapshot == config.model_snapshot
            for item in attempts
        ),
        "both_conditions_twice": sorted(item.condition for item in attempts)
        == ["authority", "authority", "hedge", "hedge"],
        "positive_latency_logged": all((item.latency_ms or 0.0) > 0.0 for item in attempts),
        "within_live_budget": summary.cumulative_cost_usd <= config.max_live_usd,
    }
    report = {
        "stage": "E4_LIVE_SMOKE",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "summary": summary.model_dump(mode="json"),
        "response_ids": [item.response_id for item in attempts],
    }
    write_json(acceptance_path, report)
    if report["status"] != "PASS":
        raise RuntimeError(f"Live smoke failed: {canonical_json(checks)}")
    return report


def run_pilot(
    provider: CompletionProvider,
    *,
    stage0_dir: str | Path = "prereg/v2/stage0",
    pilot_dir: str | Path = "prereg/v2/pilot",
    smoke_results_dir: str | Path = "results/v2/smoke",
    results_dir: str | Path = "results/v2/pilot",
) -> dict:
    """Run the frozen 200-slot pilot only after a passing live smoke."""

    smoke = json.loads((Path(smoke_results_dir) / "smoke_acceptance.json").read_text())
    if smoke.get("status") != "PASS":
        raise RuntimeError("A passing live smoke is required before the pilot")
    stage0 = verify_stage0_directory(stage0_dir)
    if stage0.status != "PASS":
        raise RuntimeError(f"Stage-0 integrity is {stage0.status}")
    config = load_config(stage0_dir)
    tasks = load_pilot_schedule(pilot_dir)
    destination = Path(results_dir)
    destination.mkdir(parents=True, exist_ok=True)
    decision_path = destination / "pilot_decision.json"
    if decision_path.exists():
        raise FileExistsError("Refusing to overwrite an existing pilot decision")
    remaining_budget = config.max_live_usd - float(smoke["summary"]["cumulative_cost_usd"])
    summary = collect_tasks(
        tasks,
        load_prompts(stage0_dir),
        provider,
        config,
        destination / "attempts.jsonl",
        destination / "halts.jsonl",
        CollectorLimits(
            max_attempts=200 + config.max_retry_attempts,
            max_retry_attempts=config.max_retry_attempts,
            max_retry_fraction=0.01,
            max_cost_usd=remaining_budget,
            preflight_cost_reserve_usd=0.001,
        ),
    )
    decision = analyze_pilot_k(read_attempt_log(destination / "attempts.jsonl"), config)
    report = {
        "stage": "E5_REPEATABILITY_PILOT",
        "status": "HALT" if decision.selected_k is None else "PASS",
        "summary": summary.model_dump(mode="json"),
        "decision": decision.model_dump(mode="json"),
        "decision_text": pilot_decision_text(decision),
    }
    write_json(decision_path, report)
    (destination / "pilot_decision.md").write_text(report["decision_text"] + "\n", encoding="utf-8")
    return report


def run_confirmatory(
    provider: CompletionProvider,
    *,
    stage0_dir: str | Path = "prereg/v2/stage0",
    confirmatory_dir: str | Path = "prereg/v2/confirmatory",
    smoke_results_dir: str | Path = "results/v2/smoke",
    pilot_results_dir: str | Path = "results/v2/pilot",
    results_dir: str | Path = "results/v2/confirmatory",
) -> dict:
    """Collect the frozen confirmatory schedule and run the frozen primary analysis."""

    acceptance = json.loads(
        (Path(confirmatory_dir) / "confirmatory_schedule_acceptance.json").read_text()
    )
    if acceptance.get("status") != "PASS":
        raise RuntimeError("A passing confirmatory freeze is required")
    selected_k = int(acceptance["selected_k"])
    tasks = load_confirmatory_schedule(confirmatory_dir)
    if len(tasks) != 60 * 2 * selected_k:
        raise RuntimeError("Confirmatory manifest count differs from frozen K")
    config = load_config(stage0_dir)
    smoke = json.loads((Path(smoke_results_dir) / "smoke_acceptance.json").read_text())
    pilot = json.loads((Path(pilot_results_dir) / "pilot_decision.json").read_text())
    if smoke.get("status") != "PASS" or pilot.get("status") != "PASS":
        raise RuntimeError("Passing smoke and pilot gates are required")
    spent = float(smoke["summary"]["cumulative_cost_usd"]) + float(
        pilot["summary"]["cumulative_cost_usd"]
    )
    used_retries = int(pilot["summary"]["retry_attempts"])
    remaining_retries = config.max_retry_attempts - used_retries
    destination = Path(results_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "primary_result.json"
    if result_path.exists():
        raise FileExistsError("Refusing to overwrite an existing confirmatory result")
    summary = collect_tasks(
        tasks,
        load_prompts(stage0_dir),
        provider,
        config,
        destination / "attempts.jsonl",
        destination / "halts.jsonl",
        CollectorLimits(
            max_attempts=len(tasks) + remaining_retries,
            max_retry_attempts=remaining_retries,
            max_retry_fraction=0.01,
            max_cost_usd=config.max_live_usd - spent,
            preflight_cost_reserve_usd=0.001,
        ),
    )
    primary = analyze_confirmatory_reshare(
        read_attempt_log(destination / "attempts.jsonl"),
        load_states(stage0_dir),
        selected_k,
        config,
    )
    report = {
        "stage": "E7_CONFIRMATORY_PRIMARY",
        "status": "PASS",
        "summary": summary.model_dump(mode="json"),
        "primary": primary.model_dump(mode="json"),
        "result_text": confirmatory_result_text(primary, config.practical_delta),
    }
    write_json(result_path, report)
    (destination / "primary_result.md").write_text(report["result_text"] + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Intervention Response Audit v2 explicitly authorized live gates")
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--authorize-live", required=True)
    smoke.add_argument("--pricing", type=Path, default=Path("prereg/v2/prelive/pricing.json"))
    pilot = commands.add_parser("pilot")
    pilot.add_argument("--authorize-live", required=True)
    pilot.add_argument("--pricing", type=Path, default=Path("prereg/v2/prelive/pricing.json"))
    confirmatory = commands.add_parser("confirmatory")
    confirmatory.add_argument("--authorize-live", required=True)
    confirmatory.add_argument("--pricing", type=Path, default=Path("prereg/v2/prelive/pricing.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    required = {
        "smoke": SMOKE_AUTHORIZATION,
        "pilot": PILOT_AUTHORIZATION,
        "confirmatory": CONFIRMATORY_AUTHORIZATION,
    }[args.command]
    if args.authorize_live != required:
        raise SystemExit(f"Authorization mismatch; required exact phrase: {required}")
    config = load_config("prereg/v2/stage0")
    pricing = TokenPricing.model_validate_json(args.pricing.read_text(encoding="utf-8"))
    provider = OpenAIChatProvider(config, pricing)
    if args.command == "smoke":
        report = run_smoke(provider)
    elif args.command == "pilot":
        report = run_pilot(provider)
    else:
        report = run_confirmatory(provider)
    print(f"{report['stage']}: {report['status']}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
