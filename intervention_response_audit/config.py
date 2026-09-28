"""Central configuration for Intervention Response Audit Phase 1."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Settings(BaseModel):
    """All project knobs that should be stable across CLI entry points."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    seed: int = 42
    data_dir: Path = Path("data")
    max_api_calls: int = 5000
    max_usd: float = 8.0
    trial_max_usd: float = 1.0
    full_run_max_usd: float = 3.5
    sweep_arm_max_usd: float = 15.0
    sweep_arm_b_max_usd: float = 10.0
    floor_diagnostic_max_usd: float = 0.60
    arm_id: str = "baseline"
    reference_simulator_id: str = "gpt54mini_20260317_feed_v1_rnone"
    model: str = "gpt-5.4-mini-2026-03-17"
    temperature: float | None = 0.0
    max_tokens: int = 300
    reasoning_effort: str = "none"
    input_usd_per_1m: float = 0.75
    output_usd_per_1m: float = 4.50
    pass_gap_nats: float = 0.05
    pass_twin_delta: float = 0.10
    stance_endorse_prob: float = 0.5
    stance_neutral_prob: float = 0.3
    stance_doubt_prob: float = 0.2
    mock_semantic_tweak: bool = True
    mock_tweak_strength: float = 8.0
    mock_latency_seconds: float = 0.0
    mock_bad_json_rate: float = 0.0


settings = Settings()


ARM_OVERLAYS: dict[str, dict[str, object]] = {
    "baseline": {},
    "armA": {
        "reasoning_effort": "medium",
        "max_tokens": 2000,
        "temperature": None,
        "reference_simulator_id": "gpt54mini_20260317_feed_v1_rmedium",
    },
    "armB": {
        "model": "gpt-5.4-2026-03-05",
        "reference_simulator_id": "gpt54full_20260305_feed_v1",
    },
}


def resolve_arm_settings(arm_id: str | None = None) -> Settings:
    """Apply one named sweep overlay without changing the frozen baseline."""

    selected = arm_id or settings.arm_id
    if selected not in ARM_OVERLAYS:
        raise ValueError(f"Unknown sweep arm: {selected}")
    return settings.model_copy(update={"arm_id": selected, **ARM_OVERLAYS[selected]})
