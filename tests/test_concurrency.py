import json
import threading
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from intervention_response_audit import cli
from intervention_response_audit.oracle.client import BudgetGuard, append_jsonl


def _run_args(run_id: str, concurrency: int, twins: bool = False) -> Namespace:
    return Namespace(
        arm_id="baseline",
        mock=True,
        mock_semantic_tweak=True,
        provider="openai",
        model="unused-by-pinned-arm",
        max_api_calls=5000,
        demo_regime=None,
        twins_only=twins,
        twin_pairs=6,
        special_sets=False,
        scenarios=6,
        T=4,
        seed=42,
        run_id=run_id,
        concurrency=concurrency,
    )


def test_ordered_parallel_map_uses_multiple_worker_threads() -> None:
    gate = threading.Barrier(4)

    def work(value: int) -> tuple[int, int]:
        gate.wait(timeout=2)
        time.sleep(0.01)
        return value, threading.get_ident()

    results = cli._ordered_parallel_map(work, list(range(8)), concurrency=4)

    assert [value for value, _ in results] == list(range(8))
    assert len({thread_id for _, thread_id in results}) >= 4


def test_budget_guard_charges_are_thread_safe() -> None:
    budget = BudgetGuard(
        max_api_calls=1000,
        max_usd=10.0,
        input_usd_per_1m=1.0,
        output_usd_per_1m=1.0,
    )
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(lambda _: budget.charge(1000, 1000), range(1000)))

    assert budget.calls == 1000
    assert budget.cost_usd == pytest.approx(2.0)


def test_arm_b_budget_uses_full_model_prices_and_ten_dollar_revised_cap(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.settings, "data_dir", tmp_path)

    budget = cli._live_budget("armB_micro_v1", 100, arm_id="armB")

    assert budget.input_usd_per_1m == 2.50
    assert budget.output_usd_per_1m == 15.00
    assert budget.max_usd == 10.0


def test_jsonl_appends_are_thread_safe(tmp_path) -> None:
    path = tmp_path / "parallel.jsonl"
    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(lambda i: append_jsonl(path, {"index": i}), range(500)))

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 500
    assert {record["index"] for record in records} == set(range(500))


def test_parallel_normal_run_matches_serial_run(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.settings, "data_dir", tmp_path)
    cli.run_oracle(_run_args("serial", concurrency=1))
    cli.run_oracle(_run_args("parallel", concurrency=4))

    serial = pd.read_parquet(tmp_path / "oracle_logs" / "serial.parquet")
    parallel = pd.read_parquet(tmp_path / "oracle_logs" / "parallel.parquet")
    compare = [
        "scenario_id",
        "step",
        "agent_id",
        "query_index",
        "believe",
        "reshare",
        "attend",
        "sampled_reshare",
        "exposure_metadata",
    ]
    pd.testing.assert_frame_equal(serial[compare], parallel[compare])
    records = [json.loads(line) for line in (tmp_path / "oracle_logs" / "parallel.jsonl").read_text().splitlines()]
    keys = {(r["scenario_id"], r["step"], r["agent_id"], r["scaffold_id"]) for r in records}
    assert len(keys) == len(records)


def test_parallel_twins_preserve_pair_identity(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli.settings, "data_dir", tmp_path)
    cli.run_oracle(_run_args("parallel_twins", concurrency=4, twins=True))

    identity = json.loads((tmp_path / "oracle_logs" / "parallel_twins_twin_identity.json").read_text())
    assert len(identity) == 6
    assert all(item["pass"] for item in identity)
    frame = pd.read_parquet(tmp_path / "oracle_logs" / "parallel_twins.parquet")
    assert frame["twin_group"].nunique() == 6
    assert set(frame["twin_side"]) == {"a", "b"}
