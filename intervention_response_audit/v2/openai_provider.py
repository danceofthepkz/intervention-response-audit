"""Fail-closed OpenAI adapter for the frozen v2 Chat Completions request."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from pydantic import Field

from intervention_response_audit.v2.collect import ProviderResponse, ProviderTransportError
from intervention_response_audit.v2.schema import StrictFrozenModel, V2FrozenConfig, content_sha256


class TokenPricing(StrictFrozenModel):
    """Standard API token prices, frozen immediately before live collection."""

    input_usd_per_1m: float = Field(gt=0)
    cached_input_usd_per_1m: float = Field(ge=0)
    output_usd_per_1m: float = Field(gt=0)
    source_url: str
    verified_at_utc: str


class ProviderConfigurationError(RuntimeError):
    """A non-retryable request/response mismatch that must halt collection."""


def token_cost_usd(
    prompt_tokens: int,
    cached_prompt_tokens: int,
    completion_tokens: int,
    pricing: TokenPricing,
) -> float:
    """Price one response without double-counting cached input tokens."""

    if min(prompt_tokens, cached_prompt_tokens, completion_tokens) < 0:
        raise ValueError("Token counts cannot be negative")
    if cached_prompt_tokens > prompt_tokens:
        raise ValueError("Cached prompt tokens cannot exceed prompt tokens")
    uncached_prompt_tokens = prompt_tokens - cached_prompt_tokens
    return (
        uncached_prompt_tokens * pricing.input_usd_per_1m
        + cached_prompt_tokens * pricing.cached_input_usd_per_1m
        + completion_tokens * pricing.output_usd_per_1m
    ) / 1_000_000


class OpenAIChatProvider:
    """Issue exactly one frozen request; never substitute parameters or models."""

    def __init__(
        self,
        config: V2FrozenConfig,
        pricing: TokenPricing,
        *,
        client: Any | None = None,
    ) -> None:
        if config.response_format != "json_object":
            raise ValueError("v2 OpenAI adapter only supports frozen json_object mode")
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.config = config
        self.pricing = pricing
        self._client = client

    def complete(self, prompt: str) -> ProviderResponse:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        started = perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.config.model_snapshot,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_output_tokens,
                reasoning_effort=self.config.reasoning_effort,
                response_format={"type": "json_object"},
                n=1,
                stream=False,
                store=False,
            )
        except (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError) as exc:
            raise ProviderTransportError(f"{type(exc).__name__}: {exc}") from exc
        latency_ms = (perf_counter() - started) * 1000.0

        if not getattr(response, "id", None):
            raise ProviderConfigurationError("OpenAI response omitted its response ID")
        choices = getattr(response, "choices", None)
        if not choices or len(choices) != 1:
            raise ProviderConfigurationError("OpenAI response did not contain exactly one choice")
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if not isinstance(content, str) or not content:
            raise ProviderConfigurationError("OpenAI response omitted text content")
        usage = getattr(response, "usage", None)
        if usage is None:
            raise ProviderConfigurationError("OpenAI response omitted token usage")

        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        cached_prompt_tokens = int(getattr(prompt_details, "cached_tokens", 0) or 0)
        reasoning_tokens = int(getattr(completion_details, "reasoning_tokens", 0) or 0)

        return ProviderResponse(
            response_id=str(response.id),
            usage_event_id=f"chat_completion:{response.id}",
            raw_response=content,
            model_snapshot=str(getattr(response, "model", "")),
            config_sha256=content_sha256(self.config),
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            call_cost_usd=token_cost_usd(
                prompt_tokens,
                cached_prompt_tokens,
                completion_tokens,
                self.pricing,
            ),
            latency_ms=latency_ms,
        )
