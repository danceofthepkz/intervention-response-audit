import json
from pathlib import Path

from intervention_response_audit.v2.collect import ProviderResponse
from intervention_response_audit.v2.design import build_confirmatory_schedule
from intervention_response_audit.v2.live import build_smoke_schedule, load_pilot_schedule, run_smoke
from intervention_response_audit.v2.schema import V2FrozenConfig, content_sha256


class UniqueProvider:
    def __init__(self):
        self.index = 0

    def complete(self, prompt):
        self.index += 1
        return ProviderResponse(
            response_id=f"response-{self.index}",
            usage_event_id=f"usage-{self.index}",
            raw_response=json.dumps(
                {"believe": 0.4, "reshare": 0.2, "attend": 0.1, "reason_code": "mock"}
            ),
            model_snapshot=V2FrozenConfig().model_snapshot,
            config_sha256=content_sha256(V2FrozenConfig()),
            prompt_tokens=100,
            cached_prompt_tokens=0,
            completion_tokens=20,
            reasoning_tokens=0,
            call_cost_usd=0.001,
            latency_ms=1.0,
        )


def test_smoke_schedule_is_four_interleaved_calls_on_one_frozen_state() -> None:
    tasks = build_smoke_schedule(load_pilot_schedule("prereg/v2/pilot"))
    assert len(tasks) == 4
    assert [task.schedule_index for task in tasks] == list(range(4))
    assert [task.condition for task in tasks] == ["authority", "hedge", "authority", "hedge"]
    assert len({task.state_id for task in tasks}) == 1


def test_offline_smoke_passes_all_mechanical_gates(tmp_path: Path) -> None:
    report = run_smoke(
        UniqueProvider(),
        prelive_dir=tmp_path / "prelive",
        results_dir=tmp_path / "smoke",
    )
    assert report["status"] == "PASS"
    assert all(report["checks"].values())
    assert len((tmp_path / "smoke" / "attempts.jsonl").read_text().splitlines()) == 4


def test_k3_confirmatory_schedule_is_frozen_balanced_and_interleaved() -> None:
    tasks = build_confirmatory_schedule(3)
    assert len(tasks) == len({task.slot_id for task in tasks}) == 360
    assert [task.schedule_index for task in tasks] == list(range(360))
    assert len({task.state_id for task in tasks}) == 60
    assert sum(task.condition == "authority" for task in tasks) == 180
    assert sum(task.condition == "hedge" for task in tasks) == 180
    assert {task.schedule_seed for task in tasks} == {20260717}
    assert [task.condition for task in tasks] != ["authority"] * 180 + ["hedge"] * 180
