"""Command-line entry points for the offline v2 Stage-0 compiler."""

from __future__ import annotations

import argparse
from pathlib import Path

from intervention_response_audit.v2.design import build_stage0, freeze_pilot_schedule_directory
from intervention_response_audit.v2.integrity import verify_stage0_directory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Intervention Response Audit v2 offline experiment tools")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("stage0-build", help="Build frozen Stage-0 artifacts offline")
    build.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/oracle_logs/phase1_full_v1.parquet"),
    )
    build.add_argument(
        "--jsonl",
        type=Path,
        default=Path("data/oracle_logs/phase1_full_v1.jsonl"),
    )
    build.add_argument("--protocol", type=Path, default=Path("V2_STUDY_PLAN.md"))
    build.add_argument("--output-dir", type=Path, default=Path("prereg/v2/stage0"))

    verify = commands.add_parser("stage0-verify", help="Verify frozen Stage-0 artifacts")
    verify.add_argument("--output-dir", type=Path, default=Path("prereg/v2/stage0"))

    pilot_schedule = commands.add_parser(
        "pilot-schedule", help="Freeze the deterministic 200-slot pilot schedule"
    )
    pilot_schedule.add_argument("--stage0-commit", required=True)
    pilot_schedule.add_argument("--stage0-dir", type=Path, default=Path("prereg/v2/stage0"))
    pilot_schedule.add_argument("--output-dir", type=Path, default=Path("prereg/v2/pilot"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "stage0-build":
        report = build_stage0(
            parquet_path=args.parquet,
            jsonl_path=args.jsonl,
            protocol_path=args.protocol,
            output_dir=args.output_dir,
        ).report
    elif args.command == "stage0-verify":
        report = verify_stage0_directory(args.output_dir)
    else:
        tasks = freeze_pilot_schedule_directory(
            stage0_commit=args.stage0_commit,
            output_dir=args.output_dir,
            stage0_dir=args.stage0_dir,
        )
        print(f"E2_PILOT_SCHEDULE: PASS ({len(tasks)} slots)")
        return 0
    print(f"{report.stage}: {report.status}")
    for check in report.checks:
        print(f"  {check.status:>21}  {check.check_id}: {check.evidence}")
    return 2 if report.status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
