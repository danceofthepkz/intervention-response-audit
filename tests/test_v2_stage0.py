import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from intervention_response_audit.v2.design import (
    FORBIDDEN_SELECTOR_COLUMNS,
    SELECTOR_COLUMNS,
    build_stage0,
    verify_stage0_directory,
)
from intervention_response_audit.v2.schema import PromptRecord, V2FrozenConfig, sha256_text
from intervention_response_audit.network import CREDIBILITY_TIERS, FRAMED_TEMPLATE_BANK, TOPICS
from intervention_response_audit.v2.templates import V2_TEMPLATE_BANK, V2_TEMPLATE_REVISION


@pytest.fixture(scope="module")
def stage0_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("v2_stage0")
    result = build_stage0(output_dir=output)
    assert result.report.status == "PENDING_MANUAL_REVIEW"
    return output


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_frozen_config_rejects_unknown_fields_and_scientific_drift() -> None:
    with pytest.raises(ValidationError):
        V2FrozenConfig.model_validate({"unexpected_override": True})
    with pytest.raises(ValidationError):
        V2FrozenConfig(k_candidates=(2, 4, 8))


def test_selector_whitelist_is_output_blind() -> None:
    assert not FORBIDDEN_SELECTOR_COLUMNS.intersection(SELECTOR_COLUMNS)


def test_v2_revision_repairs_low_tier_without_mutating_v1_templates() -> None:
    assert V2FrozenConfig().template_revision == V2_TEMPLATE_REVISION
    for topic in TOPICS:
        assert "no posted confirmation" in FRAMED_TEMPLATE_BANK[topic]["low"]["authority"]["text"]
        assert "confirmed internally" in V2_TEMPLATE_BANK[topic]["low"]["authority"]["text"]
        assert "no official source has confirmed it" in V2_TEMPLATE_BANK[topic]["low"]["hedge"]["text"]
        for tier in CREDIBILITY_TIERS:
            if tier == "low":
                continue
            for side in ("authority", "hedge"):
                assert V2_TEMPLATE_BANK[topic][tier][side]["text"] == FRAMED_TEMPLATE_BANK[topic][tier][side]["text"]


def test_stage0_allocation_has_exact_frozen_margins(stage0_dir: Path) -> None:
    states = _read_jsonl(stage0_dir / "state_allocation.jsonl")
    assert len(states) == 80
    assert len({row["scenario_id"] for row in states}) == 80
    assert Counter(row["allocation"] for row in states) == {
        "confirmatory": 60,
        "pilot": 10,
        "reserve": 10,
    }
    confirmatory = [row for row in states if row["allocation"] == "confirmatory"]
    assert Counter(row["topic"] for row in confirmatory) == {
        "club_event": 12,
        "course_material": 12,
        "gossip": 12,
        "internship": 12,
        "policy_change": 12,
    }
    assert Counter(row["credibility_tier"] for row in confirmatory) == {
        "low": 23,
        "mid": 16,
        "high": 21,
    }
    assert Counter(row["exposure_bin"] for row in confirmatory) == {
        "1": 32,
        "2": 20,
        "3+": 8,
    }
    assert Counter(row["step_bin"] for row in confirmatory) == {
        "1": 15,
        "2": 16,
        "3+": 29,
    }
    cells = Counter(row["template_cell"] for row in confirmatory)
    assert len(cells) == 15
    assert min(cells.values()) >= 2
    assert Counter(row["topic"] for row in states if row["allocation"] == "pilot") == {
        topic: 2 for topic in V2FrozenConfig().topic_targets
    }
    assert Counter(row["topic"] for row in states if row["allocation"] == "reserve") == {
        topic: 2 for topic in V2FrozenConfig().topic_targets
    }


def test_stage0_manifest_contains_no_forbidden_outcomes(stage0_dir: Path) -> None:
    records = _read_jsonl(stage0_dir / "state_allocation.jsonl")
    serialized_keys = set().union(*(record.keys() for record in records))
    feature_keys = set().union(*(record["structured_features"].keys() for record in records))
    assert not FORBIDDEN_SELECTOR_COLUMNS.intersection(serialized_keys)
    assert not FORBIDDEN_SELECTOR_COLUMNS.intersection(feature_keys)


def test_prompt_corpus_has_70_exact_ah_pairs(stage0_dir: Path) -> None:
    prompts = [PromptRecord.model_validate(row) for row in _read_jsonl(stage0_dir / "prompt_corpus.jsonl")]
    assert len(prompts) == 140
    grouped: dict[str, list[PromptRecord]] = defaultdict(list)
    for prompt in prompts:
        grouped[prompt.state_id].append(prompt)
        assert prompt.prompt_sha256 == sha256_text(prompt.full_prompt)
        assert prompt.full_prompt[prompt.message_start:prompt.message_end] == prompt.message_text
        prompt_bytes = prompt.full_prompt.encode("utf-8")
        assert prompt_bytes[prompt.message_start_byte:prompt.message_end_byte] == prompt.message_text.encode("utf-8")
    assert len(grouped) == 70
    for pair in grouped.values():
        assert {item.condition for item in pair} == {"authority", "hedge"}
        assert pair[0].outside_message_sha256 == pair[1].outside_message_sha256
        left_outside = (
            pair[0].full_prompt[:pair[0].message_start],
            pair[0].full_prompt[pair[0].message_end:],
        )
        right_outside = (
            pair[1].full_prompt[:pair[1].message_start],
            pair[1].full_prompt[pair[1].message_end:],
        )
        assert left_outside == right_outside
        assert pair[0].message_text != pair[1].message_text


def test_machine_pass_does_not_bypass_blind_review(stage0_dir: Path) -> None:
    report = verify_stage0_directory(stage0_dir)
    assert report.status == "PENDING_MANUAL_REVIEW"
    pending = [check for check in report.checks if check.status == "PENDING_MANUAL_REVIEW"]
    assert [check.check_id for check in pending] == ["E1_BLIND_TEMPLATE_REVIEW"]
    assert all(check.status != "FAIL" for check in report.checks)


def test_stage0_builder_refuses_to_overwrite_existing_freeze(stage0_dir: Path) -> None:
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_stage0(output_dir=stage0_dir)


def test_completed_blind_checklist_is_required_for_pass(stage0_dir: Path, tmp_path: Path) -> None:
    copied = tmp_path / "reviewed"
    shutil.copytree(stage0_dir, copied)
    checklist_path = copied / "template_validation.csv"
    checklist = pd.read_csv(checklist_path, keep_default_na=False)
    for column in (
        "factual_payload_constant",
        "authority_manipulation_success",
        "hedge_manipulation_success",
    ):
        checklist[column] = "yes"
    checklist["reviewer_id"] = "blind-reviewer-test"
    checklist["reviewed_at_utc"] = "2026-07-16T00:00:00Z"
    checklist.to_csv(checklist_path, index=False, lineterminator="\n")

    report = verify_stage0_directory(copied)
    assert report.status == "PASS"
    assert all(check.status == "PASS" for check in report.checks)


def test_completed_blind_review_with_a_no_is_a_failure(stage0_dir: Path, tmp_path: Path) -> None:
    copied = tmp_path / "review-failed"
    shutil.copytree(stage0_dir, copied)
    checklist_path = copied / "template_validation.csv"
    checklist = pd.read_csv(checklist_path, keep_default_na=False)
    for column in (
        "factual_payload_constant",
        "authority_manipulation_success",
        "hedge_manipulation_success",
    ):
        checklist[column] = "yes"
    checklist.loc[0, "authority_manipulation_success"] = "no"
    checklist.loc[0, "notes"] = "Authority cue is not sufficiently distinct."
    checklist["reviewer_id"] = "blind-reviewer-test"
    checklist["reviewed_at_utc"] = "2026-07-16T00:00:00Z"
    checklist.to_csv(checklist_path, index=False, lineterminator="\n")

    report = verify_stage0_directory(copied)
    assert report.status == "FAIL"
    by_id = {check.check_id: check.status for check in report.checks}
    assert by_id["E1_TEMPLATE_CHECKLIST_INTEGRITY"] == "PASS"
    assert by_id["E1_BLIND_TEMPLATE_REVIEW"] == "FAIL"


def test_prompt_tampering_is_detected(stage0_dir: Path, tmp_path: Path) -> None:
    copied = tmp_path / "tampered"
    shutil.copytree(stage0_dir, copied)
    prompt_path = copied / "prompt_corpus.jsonl"
    rows = _read_jsonl(prompt_path)
    rows[0]["full_prompt"] += " tampered"
    prompt_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    report = verify_stage0_directory(copied)
    assert report.status == "FAIL"
    by_id = {check.check_id: check.status for check in report.checks}
    assert by_id["E1_PROMPT_HASH_INTEGRITY"] == "FAIL"
