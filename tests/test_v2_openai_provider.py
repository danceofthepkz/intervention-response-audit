from types import SimpleNamespace

import httpx
import pytest
from openai import APITimeoutError

from intervention_response_audit.v2.collect import ProviderTransportError
from intervention_response_audit.v2.openai_provider import (
    OpenAIChatProvider,
    ProviderConfigurationError,
    TokenPricing,
    token_cost_usd,
)
from intervention_response_audit.v2.schema import V2FrozenConfig, content_sha256


class FakeCompletions:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def fake_client(result=None, error=None):
    completions = FakeCompletions(result, error)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def pricing() -> TokenPricing:
    return TokenPricing(
        input_usd_per_1m=0.75,
        cached_input_usd_per_1m=0.075,
        output_usd_per_1m=4.50,
        source_url="https://developers.openai.com/api/docs/models/gpt-5.4-mini",
        verified_at_utc="2026-07-17T00:00:00Z",
    )


def response(*, content='{"believe":0.4,"reshare":0.2,"attend":0.1,"reason_code":"x"}'):
    return SimpleNamespace(
        id="chatcmpl-test-1",
        model="gpt-5.4-mini-2026-03-17",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=1000,
            completion_tokens=100,
            prompt_tokens_details=SimpleNamespace(cached_tokens=400),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=30),
        ),
    )


def test_adapter_sends_exact_frozen_request_and_captures_usage() -> None:
    config = V2FrozenConfig()
    client, completions = fake_client(response())
    result = OpenAIChatProvider(config, pricing(), client=client).complete("frozen prompt")

    assert completions.calls == [
        {
            "model": "gpt-5.4-mini-2026-03-17",
            "messages": [{"role": "user", "content": "frozen prompt"}],
            "temperature": 0.0,
            "max_completion_tokens": 300,
            "reasoning_effort": "none",
            "response_format": {"type": "json_object"},
            "n": 1,
            "stream": False,
            "store": False,
        }
    ]
    assert result.response_id == "chatcmpl-test-1"
    assert result.usage_event_id == "chat_completion:chatcmpl-test-1"
    assert result.model_snapshot == config.model_snapshot
    assert result.config_sha256 == content_sha256(config)
    assert result.cached_prompt_tokens == 400
    assert result.reasoning_tokens == 30
    assert result.latency_ms >= 0.0
    assert result.call_cost_usd == pytest.approx((600 * 0.75 + 400 * 0.075 + 100 * 4.5) / 1e6)


def test_cost_does_not_double_count_cached_tokens() -> None:
    assert token_cost_usd(1000, 400, 100, pricing()) == pytest.approx(0.00093)
    with pytest.raises(ValueError, match="exceed"):
        token_cost_usd(10, 11, 1, pricing())


def test_retryable_sdk_error_maps_to_exact_prompt_retry_class() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    client, completions = fake_client(error=APITimeoutError(request=request))
    with pytest.raises(ProviderTransportError, match="APITimeoutError"):
        OpenAIChatProvider(V2FrozenConfig(), pricing(), client=client).complete("same")
    assert len(completions.calls) == 1


def test_non_retryable_error_propagates_without_parameter_fallback() -> None:
    client, completions = fake_client(error=RuntimeError("bad frozen parameter"))
    with pytest.raises(RuntimeError, match="bad frozen parameter"):
        OpenAIChatProvider(V2FrozenConfig(), pricing(), client=client).complete("same")
    assert len(completions.calls) == 1
    assert completions.calls[0]["temperature"] == 0.0


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda item: setattr(item, "id", None), "response ID"),
        (lambda item: setattr(item, "choices", []), "exactly one choice"),
        (lambda item: setattr(item.choices[0].message, "content", ""), "text content"),
        (lambda item: setattr(item, "usage", None), "token usage"),
    ],
)
def test_incomplete_provider_response_halts(mutator, message) -> None:
    item = response()
    mutator(item)
    client, _ = fake_client(item)
    with pytest.raises(ProviderConfigurationError, match=message):
        OpenAIChatProvider(V2FrozenConfig(), pricing(), client=client).complete("same")
