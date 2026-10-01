"""OpenAI Responses adapter: pure mappings against recorded payloads, plus a fake client for ``complete``."""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx2
import openai
import pytest
from conftest import load_provider_fixture
from openai.types.responses import Response

from maf.config import price_for
from maf.providers.base import CompletionRequest, Message, ProviderError, ProviderRefusal, StructuredOutputError
from maf.providers.openai_provider import OpenAIProvider
from maf.types import Usage

ON = date(2026, 9, 28)
MODEL = "gpt-6-sol"
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"mode": {"type": "string"}, "summary": {"type": "string"}},
    "required": ["mode", "summary"],
    "additionalProperties": False,
}


def _req(**kw: Any) -> CompletionRequest:
    return CompletionRequest.simple(MODEL, "Plan the allocator.", max_output_tokens=16_000, **kw)


def _cost(usage: Usage) -> float:
    p = price_for(MODEL, ON)
    uncached = usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens
    write_rate = p.cache_write_per_mtok or p.input_per_mtok
    return (
        uncached * p.input_per_mtok
        + usage.cached_input_tokens * p.cached_input_per_mtok
        + usage.cache_write_tokens * write_rate
        + usage.output_tokens * p.output_per_mtok
    ) / 1e6


class FakeResponses:
    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class FakeClient:
    def __init__(self, reply: Any) -> None:
        self.responses = FakeResponses(reply)


def _provider(reply: Any) -> tuple[OpenAIProvider, FakeClient]:
    client = FakeClient(reply)
    return OpenAIProvider(client, today=lambda: ON), client


# --- build_params -------------------------------------------------------------------------------


def test_build_params_minimal() -> None:
    assert OpenAIProvider.build_params(_req()) == {
        "model": MODEL,
        "input": [{"role": "user", "content": "Plan the allocator."}],
        "max_output_tokens": 16_000,
        "store": False,
    }


def test_build_params_full() -> None:
    req = CompletionRequest(
        model=MODEL,
        system="You are the strategist.",
        messages=(
            Message(role="user", content="a"),
            Message(role="assistant", content="b"),
            Message(role="user", content="c"),
        ),
        max_output_tokens=500,
        effort="xhigh",
        json_schema=SCHEMA,
        schema_name="triage",
        web_search=True,  # ignored by OpenAI
    )
    params = OpenAIProvider.build_params(req)
    assert params["instructions"] == "You are the strategist."
    assert params["input"] == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ]
    assert params["reasoning"] == {"effort": "xhigh"}
    assert params["text"] == {"format": {"type": "json_schema", "name": "triage", "schema": SCHEMA, "strict": True}}
    assert "tools" not in params


def test_build_params_matches_sdk_signature() -> None:
    import inspect

    from openai.resources.responses import Responses

    accepted = set(inspect.signature(Responses.create).parameters)
    req = _req(system="s", effort="high", json_schema=SCHEMA)
    assert set(OpenAIProvider.build_params(req)) <= accepted


# --- to_result ----------------------------------------------------------------------------------


def test_to_result_text_fixture() -> None:
    result = OpenAIProvider.to_result(load_provider_fixture("openai_text"), _req(), on=ON)
    assert result.text == "## Summary\n\nThree options were considered."
    assert result.parsed is None
    assert result.usage == Usage(
        input_tokens=12450, cached_input_tokens=4096, cache_write_tokens=1024, output_tokens=3120, reasoning_tokens=2048
    )
    assert result.cost_usd == pytest.approx(_cost(result.usage))
    assert result.provider == "openai"
    assert result.model == MODEL  # priced and reported at the pinned id, not the dated snapshot
    assert result.response_id == "resp_68d9a1c0f2b4819a"
    assert result.stop_reason == "completed"
    assert result.raw["model"] == "gpt-6-sol-2026-08-14"


def test_to_result_accepts_sdk_object() -> None:
    response = Response.model_validate(load_provider_fixture("openai_text"))
    assert OpenAIProvider.to_result(response, _req(), on=ON).text.startswith("## Summary")


def test_to_result_structured() -> None:
    result = OpenAIProvider.to_result(load_provider_fixture("openai_json"), _req(json_schema=SCHEMA), on=ON)
    assert result.parsed == {"mode": "code", "summary": "Build an O(1) allocator."}
    assert result.usage.reasoning_tokens == 700


def test_to_result_structured_schema_mismatch() -> None:
    schema = {
        **SCHEMA,
        "required": ["mode", "summary", "risks"],
        "properties": {**SCHEMA["properties"], "risks": {"type": "array"}},
    }
    with pytest.raises(StructuredOutputError, match="risks"):
        OpenAIProvider.to_result(load_provider_fixture("openai_json"), _req(json_schema=schema), on=ON)


def test_to_result_incomplete_with_schema_raises_with_cost() -> None:
    with pytest.raises(StructuredOutputError, match="max_output_tokens") as info:
        OpenAIProvider.to_result(load_provider_fixture("openai_incomplete"), _req(json_schema=SCHEMA), on=ON)
    assert info.value.cost_usd == pytest.approx(_cost(Usage(input_tokens=2000, output_tokens=16000)))


def test_to_result_incomplete_without_schema_returns_truncated_text() -> None:
    result = OpenAIProvider.to_result(load_provider_fixture("openai_incomplete"), _req(), on=ON)
    assert result.stop_reason == "max_output_tokens"
    assert result.text == '{"mode": "co'


def test_to_result_refusal() -> None:
    with pytest.raises(ProviderRefusal, match="can't help") as info:
        OpenAIProvider.to_result(load_provider_fixture("openai_refusal"), _req(), on=ON)
    assert info.value.cost_usd > 0


def test_to_result_content_filter_is_refusal() -> None:
    payload = load_provider_fixture("openai_incomplete")
    payload["incomplete_details"] = {"reason": "content_filter"}
    with pytest.raises(ProviderRefusal):
        OpenAIProvider.to_result(payload, _req(), on=ON)


def test_to_result_failed() -> None:
    with pytest.raises(ProviderError, match="server_error") as info:
        OpenAIProvider.to_result(load_provider_fixture("openai_failed"), _req(), on=ON)
    assert type(info.value) is ProviderError
    assert info.value.cost_usd == pytest.approx(_cost(Usage(input_tokens=500)))


# --- complete -----------------------------------------------------------------------------------


def test_complete_sends_build_params_and_maps_result() -> None:
    provider, client = _provider(Response.model_validate(load_provider_fixture("openai_json")))
    req = _req(json_schema=SCHEMA, effort="high")
    result = provider.complete(req)
    assert client.responses.calls == [OpenAIProvider.build_params(req)]
    assert result.parsed == {"mode": "code", "summary": "Build an O(1) allocator."}


@pytest.mark.parametrize(
    ("exc", "retryable", "fragment"),
    [
        (
            openai.RateLimitError(
                "slow down", response=httpx2.Response(429, request=httpx2.Request("POST", "https://x")), body=None
            ),
            True,
            "429",
        ),
        (
            openai.BadRequestError(
                "bad schema", response=httpx2.Response(400, request=httpx2.Request("POST", "https://x")), body=None
            ),
            False,
            "400",
        ),
        (
            openai.InternalServerError(
                "boom", response=httpx2.Response(503, request=httpx2.Request("POST", "https://x")), body=None
            ),
            True,
            "503",
        ),
        (ValueError("weird"), False, "ValueError"),
    ],
)
def test_complete_wraps_sdk_errors(exc: Exception, retryable: bool, fragment: str) -> None:
    provider, _ = _provider(exc)
    with pytest.raises(ProviderError, match=fragment) as info:
        provider.complete(_req())
    assert info.value.retryable is retryable
    assert info.value.cost_usd == 0.0
    assert info.value.__cause__ is exc


def test_timeout_is_charged_worst_case_and_connect_failure_is_free() -> None:
    """A read timeout may still be generated and billed server-side; a connect failure never reached it."""
    request = httpx2.Request("POST", "https://x")
    timeout = openai.APITimeoutError(request=request)
    timeout.__cause__ = httpx2.ReadTimeout("read timed out")
    provider, _ = _provider(timeout)
    with pytest.raises(ProviderError) as info:
        provider.complete(_req())
    assert info.value.retryable is True
    assert info.value.cost_usd == pytest.approx(provider.worst_case_cost(_req()))

    refused = openai.APIConnectionError(request=request)
    refused.__cause__ = httpx2.ConnectError("connection refused")
    provider, _ = _provider(refused)
    with pytest.raises(ProviderError) as info:
        provider.complete(_req())
    assert info.value.retryable is True and info.value.cost_usd == 0.0


def test_sdk_retries_are_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    client = OpenAIProvider()._get_client()
    assert client.max_retries == 0


def test_complete_refuses_unpriced_model_before_calling() -> None:
    provider, client = _provider(Response.model_validate(load_provider_fixture("openai_text")))
    with pytest.raises(ProviderError, match="unpriced"):
        provider.complete(CompletionRequest.simple("gpt-unknown", "x", max_output_tokens=10))
    assert client.responses.calls == []


def test_missing_api_key_raises_provider_error() -> None:
    provider = OpenAIProvider(today=lambda: ON)  # conftest strips OPENAI_API_KEY
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        provider.complete(_req())


def test_lazy_client_is_built_from_env_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    provider = OpenAIProvider(timeout_s=12.0, today=lambda: ON)
    client = provider._get_client()
    assert isinstance(client, openai.OpenAI)
    assert client.timeout == 12.0
    assert provider._get_client() is client


def test_worst_case_cost() -> None:
    provider = OpenAIProvider(today=lambda: ON)
    req = _req(system="s" * 3000)
    p = price_for(MODEL, ON)
    from maf.providers.base import estimate_request_tokens

    input_rate = max(p.input_per_mtok, p.cache_write_per_mtok or 0.0)
    expected = estimate_request_tokens(req) * input_rate / 1e6 + 16_000 * p.output_per_mtok / 1e6
    assert provider.worst_case_cost(req) == pytest.approx(expected)
    assert provider.worst_case_cost(req, on=ON) == pytest.approx(expected)
