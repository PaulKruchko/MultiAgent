"""Anthropic Messages adapter: pure mappings against recorded payloads, plus a fake streaming client."""

from __future__ import annotations

import inspect
from datetime import date
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message as AnthropicMessage
from conftest import load_provider_fixture

from maf.config import price_for
from maf.providers.base import (
    CompletionRequest,
    Message,
    ProviderError,
    ProviderRefusal,
    StructuredOutputError,
    estimate_request_tokens,
)
from maf.providers.claude_provider import DEFAULT_EFFORT, ClaudeProvider
from maf.types import Usage

ON = date(2026, 9, 28)
MODEL = "claude-opus-5-5"
SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}, "issues": {"type": "array", "items": {"type": "string"}}},
    "required": ["verdict", "issues"],
    "additionalProperties": False,
}


def _req(**kw: Any) -> CompletionRequest:
    return CompletionRequest.simple(MODEL, "Critique the draft.", max_output_tokens=64_000, **kw)


def _cost(usage: Usage) -> float:
    p = price_for(MODEL, ON)
    uncached = usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens
    return (
        uncached * p.input_per_mtok
        + usage.cached_input_tokens * p.cached_input_per_mtok
        + usage.cache_write_tokens * (p.cache_write_per_mtok or p.input_per_mtok)
        + usage.output_tokens * p.output_per_mtok
    ) / 1e6


class FakeStream:
    def __init__(self, final: Any, snapshot: Any = None) -> None:
        self._final = final
        self._snapshot = snapshot

    @property
    def current_message_snapshot(self) -> Any:
        assert self._snapshot is not None
        return self._snapshot

    def get_final_message(self) -> Any:
        if isinstance(self._final, Exception):
            raise self._final
        return self._final


class FakeStreamManager:
    def __init__(self, stream: FakeStream | Exception) -> None:
        self._stream = stream
        self.closed = False

    def __enter__(self) -> FakeStream:
        if isinstance(self._stream, Exception):
            raise self._stream
        return self._stream

    def __exit__(self, *exc: object) -> None:
        self.closed = True


class FakeMessages:
    def __init__(self, stream: FakeStream | Exception) -> None:
        self.stream_obj = stream
        self.calls: list[dict[str, Any]] = []
        self.managers: list[FakeStreamManager] = []

    def stream(self, **kwargs: Any) -> FakeStreamManager:
        self.calls.append(kwargs)
        manager = FakeStreamManager(self.stream_obj)
        self.managers.append(manager)
        return manager

    def create(self, **kwargs: Any) -> Any:  # pragma: no cover - must never be used
        raise AssertionError("ClaudeProvider must stream, not call messages.create")


class FakeClient:
    def __init__(self, stream: FakeStream | Exception) -> None:
        self.messages = FakeMessages(stream)


def _message(name: str) -> AnthropicMessage:
    return AnthropicMessage.model_validate(load_provider_fixture(name))


def _provider(stream: FakeStream | Exception) -> tuple[ClaudeProvider, FakeClient]:
    client = FakeClient(stream)
    return ClaudeProvider(client, today=lambda: ON), client


# --- build_params -------------------------------------------------------------------------------


def test_build_params_minimal_sets_adaptive_thinking_and_explicit_effort() -> None:
    assert ClaudeProvider.build_params(_req()) == {
        "model": MODEL,
        "max_tokens": 64_000,
        "messages": [{"role": "user", "content": "Critique the draft."}],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": DEFAULT_EFFORT},
    }
    assert DEFAULT_EFFORT == "high"


def test_build_params_full() -> None:
    req = CompletionRequest(
        model=MODEL,
        system="You are the editor.",
        messages=(
            Message(role="user", content="a"),
            Message(role="assistant", content="b"),
            Message(role="user", content="c"),
        ),
        max_output_tokens=8000,
        effort="max",
        json_schema=SCHEMA,
    )
    params = ClaudeProvider.build_params(req)
    assert params["system"] == [{"type": "text", "text": "You are the editor.", "cache_control": {"type": "ephemeral"}}]
    assert params["output_config"] == {"effort": "max", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert [m["role"] for m in params["messages"]] == ["user", "assistant", "user"]
    for forbidden in ("temperature", "top_p", "top_k", "tool_choice", "output_format"):
        assert forbidden not in params


def test_build_params_matches_sdk_signature() -> None:
    from anthropic.resources.messages import Messages

    accepted = set(inspect.signature(Messages.stream).parameters)
    assert set(ClaudeProvider.build_params(_req(system="s", json_schema=SCHEMA))) <= accepted


def test_build_params_rejects_prefill() -> None:
    req = CompletionRequest(
        model=MODEL,
        messages=(Message(role="user", content="a"), Message(role="assistant", content="{")),
        max_output_tokens=10,
    )
    with pytest.raises(ProviderError, match="prefill"):
        ClaudeProvider.build_params(req)


# --- to_result ----------------------------------------------------------------------------------


def test_to_result_normalizes_cache_usage_and_skips_thinking() -> None:
    result = ClaudeProvider.to_result(load_provider_fixture("anthropic_text"), _req(), on=ON)
    assert result.text == "## Summary\n\nThe draft is sound.\n\n## Issues\n\nNone."
    assert result.usage == Usage(
        input_tokens=1200 + 8000 + 2500, cached_input_tokens=8000, cache_write_tokens=2500, output_tokens=4100
    )
    assert result.cost_usd == pytest.approx(_cost(result.usage))
    assert result.provider == "anthropic"
    assert result.stop_reason == "end_turn"
    assert result.response_id == "msg_01XyZ7pQ2nV8rT4kLm9sA3bC"
    assert result.parsed is None


def test_to_result_structured() -> None:
    result = ClaudeProvider.to_result(_message("anthropic_json"), _req(json_schema=SCHEMA), on=ON)
    assert result.parsed == {"verdict": "accept", "issues": []}


def test_to_result_refusal_raises_with_cost() -> None:
    with pytest.raises(ProviderRefusal) as info:
        ClaudeProvider.to_result(load_provider_fixture("anthropic_refusal"), _req(), on=ON)
    assert info.value.cost_usd == pytest.approx(_cost(Usage(input_tokens=400, output_tokens=5)))
    assert info.value.provider == "anthropic"


def test_to_result_max_tokens_with_schema() -> None:
    with pytest.raises(StructuredOutputError, match="max_tokens") as info:
        ClaudeProvider.to_result(load_provider_fixture("anthropic_max_tokens"), _req(json_schema=SCHEMA), on=ON)
    assert info.value.cost_usd > 1.0  # 64k output tokens were billed


def test_to_result_max_tokens_without_schema_returns_text() -> None:
    result = ClaudeProvider.to_result(load_provider_fixture("anthropic_max_tokens"), _req(), on=ON)
    assert result.stop_reason == "max_tokens"


# --- complete -----------------------------------------------------------------------------------


def test_complete_streams_and_returns_final_message() -> None:
    provider, client = _provider(FakeStream(_message("anthropic_json")))
    req = _req(json_schema=SCHEMA, system="sys")
    result = provider.complete(req)
    assert client.messages.calls == [ClaudeProvider.build_params(req)]
    assert client.messages.managers[0].closed
    assert result.parsed == {"verdict": "accept", "issues": []}


def test_complete_mid_stream_failure_records_partial_spend() -> None:
    """After message_delta (stop_reason known) the snapshot's output_tokens are exact."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    overloaded = anthropic.InternalServerError("overloaded", response=httpx2.Response(529, request=request), body=None)
    snapshot = _message("anthropic_json").model_copy(
        update={"usage": anthropic.types.Usage(input_tokens=3000, output_tokens=200)}
    )
    provider, _ = _provider(FakeStream(overloaded, snapshot=snapshot))
    with pytest.raises(ProviderError, match="529") as info:
        provider.complete(_req())
    assert info.value.retryable
    assert info.value.cost_usd == pytest.approx(_cost(Usage(input_tokens=3000, output_tokens=200)))


def test_mid_stream_failure_before_message_delta_charges_max_output() -> None:
    """Before message_delta the snapshot still holds message_start's ~1 output token; charge the full output."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    dropped = anthropic.APIConnectionError(request=request)
    snapshot = _message("anthropic_json").model_copy(
        update={"usage": anthropic.types.Usage(input_tokens=3000, output_tokens=1), "stop_reason": None}
    )
    provider, _ = _provider(FakeStream(dropped, snapshot=snapshot))
    with pytest.raises(ProviderError) as info:
        provider.complete(_req())
    assert info.value.cost_usd == pytest.approx(_cost(Usage(input_tokens=3000, output_tokens=64_000)))


def test_mid_stream_sse_error_on_200_is_retryable() -> None:
    """An SSE ``error`` event after a 200 surfaces as APIStatusError(status_code=200) in anthropic 1.9."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    sse_error = anthropic.APIStatusError(
        "Overloaded", response=httpx2.Response(200, request=request), body={"type": "error", "error": {"type": "overloaded_error"}}
    )
    snapshot = _message("anthropic_json").model_copy(
        update={"usage": anthropic.types.Usage(input_tokens=3000, output_tokens=1), "stop_reason": None}
    )
    provider, _ = _provider(FakeStream(sse_error, snapshot=snapshot))
    with pytest.raises(ProviderError, match="200") as info:
        provider.complete(_req())
    assert info.value.retryable is True
    assert info.value.cost_usd > 0


def test_timeout_before_stream_start_charges_worst_case() -> None:
    timeout = anthropic.APITimeoutError(request=httpx2.Request("POST", "https://x"))
    provider, _ = _provider(timeout)
    with pytest.raises(ProviderError) as info:
        provider.complete(_req())
    assert info.value.retryable is True
    assert info.value.cost_usd == pytest.approx(provider.worst_case_cost(_req()))


@pytest.mark.parametrize(
    ("exc", "retryable"),
    [
        (
            anthropic.RateLimitError(
                "rl", response=httpx2.Response(429, request=httpx2.Request("POST", "https://x")), body=None
            ),
            True,
        ),
        (
            anthropic.BadRequestError(
                "bad", response=httpx2.Response(400, request=httpx2.Request("POST", "https://x")), body=None
            ),
            False,
        ),
        (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x")), True),
        (RuntimeError("?"), False),
    ],
)
def test_complete_wraps_errors_raised_before_streaming(exc: Exception, retryable: bool) -> None:
    provider, _ = _provider(exc)
    with pytest.raises(ProviderError) as info:
        provider.complete(_req())
    assert info.value.retryable is retryable
    assert info.value.cost_usd == 0.0


def test_complete_refuses_unpriced_model_before_calling() -> None:
    provider, client = _provider(FakeStream(_message("anthropic_text")))
    with pytest.raises(ProviderError, match="unpriced"):
        provider.complete(CompletionRequest.simple("claude-2", "x", max_output_tokens=10))
    assert client.messages.calls == []


def test_missing_api_key() -> None:
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        ClaudeProvider(today=lambda: ON).complete(_req())


def test_lazy_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    provider = ClaudeProvider(timeout_s=30.0)
    client = provider._get_client()
    assert isinstance(client, anthropic.Anthropic)
    assert client.max_retries == 0
    assert provider._get_client() is client


def test_worst_case_cost_assumes_no_cache_discount() -> None:
    req = _req(system="x" * 9000)
    p = price_for(MODEL, ON)
    rate = max(p.input_per_mtok, p.cache_write_per_mtok or 0.0)
    expected = estimate_request_tokens(req) * rate / 1e6 + 64_000 * p.output_per_mtok / 1e6
    assert ClaudeProvider(today=lambda: ON).worst_case_cost(req) == pytest.approx(expected)
