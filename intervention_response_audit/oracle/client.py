"""Provider wrappers, mock client, strict parsing, logging, and budget guard."""

from __future__ import annotations

import json
import math
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from intervention_response_audit.config import settings


class LLMClient(Protocol):
    model: str
    temperature: float
    max_tokens: int

    def complete(self, prompt: str) -> str:
        """Return a raw model completion."""


class BudgetExceeded(RuntimeError):
    """Raised when API call or cost caps would be exceeded."""


@dataclass
class BudgetGuard:
    max_api_calls: int = 5000
    max_usd: float = 8.0
    input_usd_per_1m: float = 0.0
    output_usd_per_1m: float = 0.0
    calls: int = 0
    cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def charge(self, prompt_tokens: int, output_tokens: int) -> float:
        cost = (prompt_tokens / 1_000_000) * self.input_usd_per_1m
        cost += (output_tokens / 1_000_000) * self.output_usd_per_1m
        with self._lock:
            next_calls = self.calls + 1
            if next_calls > self.max_api_calls:
                raise BudgetExceeded(f"API call cap exceeded: {next_calls}>{self.max_api_calls}")
            if self.cost_usd + cost > self.max_usd:
                raise BudgetExceeded(f"API budget exceeded: ${self.cost_usd + cost:.4f}>${self.max_usd:.4f}")
            self.calls = next_calls
            self.cost_usd += cost
        return cost


class MockLLMClient:
    """Deterministic mock oracle with a hidden logistic rule and semantic keyword tweak."""

    def __init__(
        self,
        model: str = "mock-logistic-semantic",
        temperature: float = 0.0,
        max_tokens: int = 300,
        semantic_tweak: bool | None = None,
        tweak_strength: float | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.semantic_tweak = settings.mock_semantic_tweak if semantic_tweak is None else semantic_tweak
        self.tweak_strength = settings.mock_tweak_strength if tweak_strength is None else tweak_strength
        self._bad_json_rng = random.Random(settings.seed)

    def complete(self, prompt: str) -> str:
        if settings.mock_latency_seconds > 0:
            time.sleep(settings.mock_latency_seconds)
        lower = prompt.lower()
        content_lower = _content_regions(prompt).lower()
        exposure_count = max(1, len(re.findall(r"(?m)^- ", prompt)))
        socialness = 0.0
        if "highly social" in lower:
            socialness = 1.0
        elif "friendly but selective" in lower:
            socialness = 0.45
        elif "small circle" in lower:
            socialness = -0.65
        skepticism = 1.0 if "skeptical" in lower else 0.35 if "some evidence" in lower else -0.15
        topic_interest = 1.0 if "actively keep up" in lower else 0.0 if "occasionally follow" in lower else -0.9
        controversy = 0.8 if any(cue in content_lower for cue in ("rumor", "reported", "write-up", "petition", "spam")) else -0.15
        credibility = 0.0
        contributions: dict[str, float] = {
            "social_pressure": min(exposure_count, 4) * 0.80,
            "skeptical_default": -1.15 * max(skepticism, 0.0),
            "topic_indifference": -1.10 if topic_interest < 0 else 0.0,
            "trusted_source_boost": 0.0,
            "doubt_from_peers": 0.0,
        }
        base = -1.85 + contributions["social_pressure"]
        base += 0.55 if "direct message" in lower else 0.0
        base += 0.65 if "someone you know well" in lower or "a close friend" in lower else 0.0
        base += 0.70 * topic_interest
        base += 0.55 * credibility
        base += contributions["skeptical_default"] + contributions["topic_indifference"]

        positive_cues = (
            "professor",
            "staff confirmed",
            "official",
            "i checked",
            "this is real",
            "verified",
            "source-backed",
            "worth paying attention",
        )
        negative_cues = (
            "not verified",
            "promo",
            "spam",
            "no idea if this is true",
            "wait for a better source",
        )
        social_proof_cues = (
            "people are sharing",
            "half my floor",
            "everyone",
            "group chat",
            "talking about it",
        )
        positive_hits = sum(_cue_present(content_lower, cue) for cue in positive_cues)
        negative_hits = sum(_cue_present(content_lower, cue) for cue in negative_cues)
        social_proof_hits = sum(_cue_present(content_lower, cue) for cue in social_proof_cues)
        believe_semantic = 0.0
        reshare_semantic = 0.0
        if self.semantic_tweak:
            believe_semantic = self.tweak_strength * (1.05 * positive_hits - 1.00 * negative_hits)
            reshare_semantic = self.tweak_strength * (
                0.62 * positive_hits + 0.55 * social_proof_hits - 0.92 * negative_hits
            )
            contributions["trusted_source_boost"] = self.tweak_strength * positive_hits
            contributions["doubt_from_peers"] = -self.tweak_strength * negative_hits

        believe_logit = base + believe_semantic
        reshare_logit = (
            base
            - 0.95
            + 0.95 * socialness
            + 0.40 * controversy
            - 0.95 * max(skepticism, 0.0) * max(controversy, 0.0)
            + 0.22 * min(exposure_count, 3)
            + reshare_semantic
        )

        believe = _sigmoid(believe_logit)
        reshare = _sigmoid(reshare_logit)
        attend = _sigmoid(believe_logit - 0.15) if ("club" in lower or "policy" in lower or "meeting" in lower) else 0.05
        reason = max(contributions.items(), key=lambda item: abs(item[1]))[0]
        response = json.dumps(
            {
                "believe": round(believe, 4),
                "reshare": round(reshare, 4),
                "attend": round(attend, 4),
                "reason_code": reason,
            }
        )
        is_repair_retry = "your last response could not be parsed" in lower
        if (
            settings.mock_bad_json_rate > 0
            and not is_repair_retry
            and self._bad_json_rng.random() < settings.mock_bad_json_rate
        ):
            return response[:-1]
        return response


class OpenAIClient:
    """Official OpenAI SDK client with exact per-response usage capture."""

    def __init__(
        self,
        model: str,
        temperature: float | None = 0.0,
        max_tokens: int = 300,
        reasoning_effort: str = "none",
        allow_temperature_fallback: bool = True,
    ) -> None:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing from the environment")
        self.model = model
        self.temperature = temperature
        self.effective_temperature: float | str = "model_default" if temperature is None else temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.allow_temperature_fallback = allow_temperature_fallback
        self.response_format = {"type": "json_object"}
        self.usage_events: list[dict[str, Any]] = []
        self.temperature_accepted: bool | None = None
        self.temperature_fallback_detail: str | None = None
        self._client = OpenAI(api_key=api_key)

    def complete(self, prompt: str) -> str:
        from openai import BadRequestError

        params: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "response_format": self.response_format,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        try:
            response = self._client.chat.completions.create(**params)
            self.temperature_accepted = True if self.temperature is not None else None
        except BadRequestError as exc:
            detail = str(exc)
            if "temperature" not in detail.lower() or not self.allow_temperature_fallback:
                raise
            self.temperature_accepted = False
            self.temperature_fallback_detail = detail
            self.effective_temperature = "model_default"
            params.pop("temperature")
            response = self._client.chat.completions.create(**params)

        usage = response.usage
        completion_details = getattr(usage, "completion_tokens_details", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        self.usage_events.append(
            {
                "response_id": response.id,
                "response_model": response.model,
                "temperature_parameter_sent": "temperature" in params,
                "prompt_tokens": int(usage.prompt_tokens or 0),
                "completion_tokens": int(usage.completion_tokens or 0),
                "total_tokens": int(usage.total_tokens or 0),
                "reasoning_tokens": int(getattr(completion_details, "reasoning_tokens", 0) or 0),
                "cached_tokens": int(getattr(prompt_details, "cached_tokens", 0) or 0),
            }
        )
        return response.choices[0].message.content or ""


class AnthropicClient:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 300) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        import anthropic

        response = anthropic.Anthropic().messages.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if getattr(block, "type", "") == "text")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _cue_present(text: str, cue: str) -> bool:
    if cue.isalpha():
        return re.search(rf"\b{re.escape(cue)}\b", text) is not None
    return cue in text


def _content_regions(prompt: str) -> str:
    quoted_messages = re.findall(r'They quote the message: "([^"]*)"', prompt)
    comments = re.findall(r'added: "([^"]*)"', prompt)
    return "\n".join(quoted_messages + comments)


def _count_tokens(text: str) -> int:
    return max(1, len(text.split()))


def parse_oracle_json(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    required = ("believe", "reshare", "attend", "reason_code")
    if not isinstance(data, dict) or set(data) != set(required):
        raise ValueError("response missing required keys")
    parsed: dict[str, Any] = {}
    for key in ("believe", "reshare", "attend"):
        value = float(data[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} outside [0,1]")
        parsed[key] = value
    parsed["reason_code"] = str(data["reason_code"])
    return parsed


_PATH_LOCKS_GUARD = threading.Lock()
_PATH_LOCKS: dict[str, threading.RLock] = {}


def _path_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with _path_lock(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def load_completed_calls(path: Path) -> dict[tuple[str, int, int, int], dict[str, Any]]:
    with _path_lock(path):
        if not path.exists():
            return {}
        completed: dict[tuple[str, int, int, int], dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("valid") and record.get("parsed_output") is not None:
                key = (
                    str(record["scenario_id"]),
                    int(record["step"]),
                    int(record["agent_id"]),
                    int(record["scaffold_id"]),
                )
                completed[key] = record["parsed_output"]
        return completed


def complete_json_logged(
    client: LLMClient,
    prompt: str,
    log_path: Path,
    budget: BudgetGuard,
    scenario_id: str,
    step: int,
    agent_id: int,
    scaffold_id: int,
    extra_log_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call the client, parse strict JSON with one repair retry, and append a log record."""

    start = time.perf_counter()
    usage_events = getattr(client, "usage_events", None)
    usage_start = len(usage_events) if isinstance(usage_events, list) else None
    raw = client.complete(prompt)
    repair_error: str | None = None
    first_attempt_valid = False
    try:
        parsed = parse_oracle_json(raw)
        valid = True
        first_attempt_valid = True
    except Exception as exc:
        repair_error = str(exc)
        repair_prompt = f"{prompt}\n\nYour last response could not be parsed: {repair_error}. Return only valid JSON."
        raw = client.complete(repair_prompt)
        try:
            parsed = parse_oracle_json(raw)
            valid = True
        except Exception as second_exc:
            parsed = None
            repair_error = str(second_exc)
            valid = False

    new_usage = usage_events[usage_start:] if usage_start is not None else []
    if new_usage:
        prompt_tokens = sum(int(event["prompt_tokens"]) for event in new_usage)
        output_tokens = sum(int(event["completion_tokens"]) for event in new_usage)
        reasoning_tokens = sum(int(event["reasoning_tokens"]) for event in new_usage)
        cached_tokens = sum(int(event["cached_tokens"]) for event in new_usage)
        charged = sum(
            budget.charge(int(event["prompt_tokens"]), int(event["completion_tokens"]))
            for event in new_usage
        )
        usage_source = "api_response"
    else:
        prompt_tokens = _count_tokens(prompt)
        output_tokens = _count_tokens(raw)
        reasoning_tokens = 0
        cached_tokens = 0
        charged = budget.charge(prompt_tokens, output_tokens)
        usage_source = "estimated"
    params: dict[str, Any] = {"temperature": client.temperature, "max_tokens": client.max_tokens}
    if hasattr(client, "reasoning_effort"):
        params["reasoning_effort"] = getattr(client, "reasoning_effort")
    if hasattr(client, "response_format"):
        params["response_format"] = getattr(client, "response_format")
    if hasattr(client, "temperature_accepted"):
        params["temperature_accepted"] = getattr(client, "temperature_accepted")
        params["temperature_fallback_detail"] = getattr(client, "temperature_fallback_detail", None)
        params["effective_temperature"] = getattr(client, "effective_temperature", client.temperature)
        params["temperature_parameter_sent"] = client.temperature is not None

    record = {
        "scenario_id": scenario_id,
        "step": step,
        "agent_id": agent_id,
        "scaffold_id": scaffold_id,
        "full_prompt": prompt,
        "raw_response": raw,
        "parsed_output": parsed,
        "valid": valid,
        "first_attempt_valid": first_attempt_valid,
        "repaired": not first_attempt_valid,
        "repair_error": repair_error,
        "model": client.model,
        "params": params,
        "latency": time.perf_counter() - start,
        "token_counts": {
            "prompt": prompt_tokens,
            "completion": output_tokens,
            "reasoning": reasoning_tokens,
            "cached_prompt": cached_tokens,
        },
        "usage_source": usage_source,
        "api_usage_events": new_usage,
        "call_cost_usd": charged,
        "cumulative_cost_usd": budget.cost_usd,
        "cumulative_calls": budget.calls,
    }
    if extra_log_fields:
        record.update(extra_log_fields)
    append_jsonl(log_path, record)
    if parsed is None:
        raise ValueError(f"Oracle returned invalid JSON after repair: {repair_error}")
    return parsed
