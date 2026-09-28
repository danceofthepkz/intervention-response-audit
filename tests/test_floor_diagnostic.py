import json
from argparse import Namespace

import pandas as pd
import pytest

from intervention_response_audit import cli
from intervention_response_audit.floor_diagnostic import decompose_floor
from intervention_response_audit.oracle.client import MockLLMClient


def _floor_frame(arm_shift: float = 0.0) -> pd.DataFrame:
    rows = []
    for item in range(40):
        for scaffold in range(5):
            delta = (scaffold - 2) * 0.02
            rows.append(
                {
                    "floor_item_id": f"floor_{item:03d}",
                    "scenario_id": f"floor_{item:03d}",
                    "step": 1,
                    "agent_id": item,
                    "scaffold_id": scaffold,
                    "topic": f"topic_{item % 5}",
                    "n_exposures": 1 + item % 3,
                    "query_index": 1,
                    "believe": 0.5 + arm_shift + delta,
                    "reshare": 0.3 + arm_shift + delta / 2,
                    "attend": 0.2 + arm_shift + delta / 4,
                }
            )
    return pd.DataFrame(rows)


def _repeat_frame(items: int, variable: bool) -> pd.DataFrame:
    rows = []
    for item in range(items):
        for repeat in range(5 if items == 20 else 3):
            delta = (repeat - 2) * 0.01 if variable else 0.0
            rows.append(
                {
                    "floor_item_id": f"floor_{item:03d}",
                    "repeat_index": repeat,
                    "believe": 0.5 + delta,
                    "reshare": 0.3 + delta / 2,
                    "attend": 0.2 + delta / 4,
                }
            )
    return pd.DataFrame(rows)


def test_repeat_loader_uses_repeat_index_as_part_of_key(tmp_path) -> None:
    path = tmp_path / "repeats.jsonl"
    records = [
        {
            "floor_item_id": "floor_000",
            "repeat_index": repeat,
            "valid": True,
            "parsed_output": {"believe": 0.5, "reshare": 0.3, "attend": 0.2, "reason_code": "x"},
        }
        for repeat in range(3)
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    completed = cli._load_completed_floor_repeats(path)

    assert set(completed) == {("floor_000", 0), ("floor_000", 1), ("floor_000", 2)}


def test_repeat_loader_rejects_duplicate_repeat_keys(tmp_path) -> None:
    path = tmp_path / "repeats.jsonl"
    record = {
        "floor_item_id": "floor_000",
        "repeat_index": 0,
        "valid": True,
        "parsed_output": {"believe": 0.5, "reshare": 0.3, "attend": 0.2, "reason_code": "x"},
    }
    path.write_text("\n".join([json.dumps(record), json.dumps(record)]), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate diagnostic repeat key"):
        cli._load_completed_floor_repeats(path)


def test_floor_decomposition_separates_sampling_and_scaffold_components() -> None:
    result = decompose_floor(
        baseline_floor=_floor_frame(),
        arm_floor=_floor_frame(arm_shift=0.05),
        baseline_repeats=_repeat_frame(10, variable=False),
        arm_repeats=_repeat_frame(20, variable=True),
    )

    assert result["baseline_contrast"]["believe"]["V_within"] == 0.0
    assert result["baseline_contrast"]["believe"]["identical_fraction"] == 1.0
    assert result["variance"]["believe"]["V_within"] > 0.0
    assert result["variance"]["believe"]["V_between"] > result["variance"]["believe"]["V_within"]
    assert 0.0 < result["variance"]["believe"]["sampling_share"] < 1.0
    assert set(result["topic_floors"]["baseline"]) == {f"topic_{i}" for i in range(5)}


def test_floor_repeat_collection_logs_thirty_unique_repeat_keys(tmp_path, monkeypatch) -> None:
    log_dir = tmp_path / "oracle_logs"
    log_dir.mkdir(parents=True)
    frame = _floor_frame()
    frame.to_parquet(log_dir / "phase1_floor_v1.parquet", index=False)
    records = []
    for row in frame.itertuples(index=False):
        records.append(
            {
                "scenario_id": row.scenario_id,
                "step": row.step,
                "agent_id": row.agent_id,
                "scaffold_id": row.scaffold_id,
                "floor_item_id": row.floor_item_id,
                "full_prompt": "Think like this student. Return only JSON.\n\nWhat you see: a campus message.",
                "exposure_metadata": [],
                "framing_type": "authority",
                "mechanism": "ic",
            }
        )
    (log_dir / "phase1_floor_v1.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    monkeypatch.setattr(cli.settings, "data_dir", tmp_path)
    monkeypatch.setattr(cli, "OpenAIClient", lambda *args, **kwargs: MockLLMClient())
    args = Namespace(
        source_run_id="phase1_floor_v1",
        run_id="diag_floor_base_v1",
        arm_id="baseline",
        items=10,
        repeats=3,
        seed=42,
    )

    cli.run_floor_repeat_live(args)

    completed = cli._load_completed_floor_repeats(log_dir / "diag_floor_base_v1.jsonl")
    assert len(completed) == 30
    assert len(pd.read_parquet(log_dir / "diag_floor_base_v1.parquet")) == 30
    manifest = json.loads((log_dir / "diag_floor_base_v1_design.json").read_text())
    assert manifest["repeat_index_logged"] is True
    assert manifest["source_prompts_reused_byte_identically"] is True
