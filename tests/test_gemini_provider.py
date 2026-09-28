"""Gemini adapter: pure mappings against recorded payloads, plus a fake client (models + files)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import load_provider_fixture
from google.genai import errors, types

from maf.config import price_for
from maf.providers.base import (
    Attachment,
    Citation,
    CompletionRequest,
    Message,
    ProviderError,
    ProviderRefusal,
    StructuredOutputError,
    estimate_request_tokens,
)
from maf.providers.gemini_provider import GeminiProvider
from maf.types import Usage

ON = date(2026, 9, 28)
MODEL = "gemini-3.8-flash"
SCHEMA = {
    "type": "object",
    "properties": {
        "sources": {"type": "array", "items": {"type": "string"}},
        "facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["sources", "facts"],
}


def _req(**kw: Any) -> CompletionRequest:
    return CompletionRequest.simple(MODEL, "Ingest the sources.", max_output_tokens=32_000, **kw)


def _cost(usage: Usage, on: date = ON) -> float:
    p = price_for(MODEL, on)
    uncached = usage.input_tokens - usage.cached_input_tokens - usage.cache_write_tokens
    return (
        uncached * p.input_per_mtok
        + usage.cached_input_tokens * p.cached_input_per_mtok
        + usage.cache_write_tokens * (p.cache_write_per_mtok or p.input_per_mtok)
        + usage.output_tokens * p.output_per_mtok
    ) / 1e6 + usage.search_queries * p.search_query_usd


def _response(name: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse.model_validate(load_provider_fixture(name))


class FakeModels:
    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, *, model: str, contents: Any, config: types.GenerateContentConfig) -> Any:
        self.calls.append({"model": model, "contents": contents, "config": config})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeFiles:
    def __init__(self, states: list[str] | None = None, fail_upload: Exception | None = None) -> None:
        self.states = states or ["ACTIVE"]
        self.fail_upload = fail_upload
        self.uploaded: list[tuple[str, Any]] = []
        self.gets: list[str] = []
        self.deleted: list[str] = []

    def _file(self, name: str, state: str) -> types.File:
        return types.File(
            name=name,
            uri=f"https://generativelanguage.googleapis.com/v1beta/{name}",
            mime_type="application/pdf",
            state=types.FileState(state),
        )

    def upload(self, *, file: str, config: Any = None) -> types.File:
        if self.fail_upload:
            raise self.fail_upload
        self.uploaded.append((file, config))
        return self._file(f"files/f{len(self.uploaded)}", self.states[0])

    def get(self, *, name: str) -> types.File:
        self.gets.append(name)
        index = min(len(self.gets), len(self.states) - 1)
        return self._file(name, self.states[index])

    def delete(self, *, name: str) -> None:
        self.deleted.append(name)


def _provider(replies: list[Any], files: FakeFiles | None = None) -> tuple[GeminiProvider, SimpleNamespace]:
    client = SimpleNamespace(models=FakeModels(replies), files=files or FakeFiles())
    return GeminiProvider(client, today=lambda: ON, sleep=lambda _s: None), client


# --- build_config / build_contents --------------------------------------------------------------


NO_AFC = {"automatic_function_calling": {"disable": True}}


def test_build_config_minimal() -> None:
    assert GeminiProvider.build_config(_req()) == {"max_output_tokens": 32_000, **NO_AFC}


@pytest.mark.parametrize("schema_in_prompt", [False, True])
def test_build_config_disables_automatic_function_calling(schema_in_prompt: bool) -> None:
    from google.genai import _extra_utils

    req = _req(json_schema=SCHEMA, web_search=True)
    sdk = types.GenerateContentConfig(**GeminiProvider.build_config(req, schema_in_prompt=schema_in_prompt))
    assert sdk.automatic_function_calling == types.AutomaticFunctionCallingConfig(disable=True)
    assert _extra_utils.should_disable_afc(sdk) is True


def test_generate_content_logs_no_afc_warning(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    """Through the real SDK ``Models.generate_content`` (only the HTTP layer is stubbed): one request, no warning."""
    from google import genai
    from google.genai import models

    monkeypatch.setattr(models.Models, "_logged_afc_warning", False)
    client = genai.Client(api_key="not-a-real-key")
    sent: list[Any] = []

    def fake_generate(*, model: str, contents: Any, config: Any) -> types.GenerateContentResponse:
        sent.append(config)
        return _response("gemini_json")

    monkeypatch.setattr(client.models, "_generate_content", fake_generate)
    provider = GeminiProvider(client, today=lambda: ON)
    with caplog.at_level("INFO", logger="google_genai"):
        result = provider.complete(_req(json_schema=SCHEMA))
    assert result.parsed is not None
    assert len(sent) == 1
    assert not [r for r in caplog.records if "automatic function calling" in r.getMessage().lower()]
    assert not [r for r in caplog.records if "AFC" in r.getMessage()]


def test_build_config_full_is_accepted_by_sdk() -> None:
    req = _req(system="You ingest.", json_schema=SCHEMA, web_search=True, effort="xhigh")
    config = GeminiProvider.build_config(req)
    assert config == {
        "max_output_tokens": 32_000,
        "system_instruction": "You ingest.",
        "response_mime_type": "application/json",
        "response_json_schema": SCHEMA,
        "tools": [{"google_search": {}}],
        "thinking_config": {"thinking_level": "HIGH"},
        **NO_AFC,
    }
    sdk = types.GenerateContentConfig(**config)
    assert sdk.tools is not None and isinstance(sdk.tools[0], types.Tool) and sdk.tools[0].google_search is not None
    assert sdk.thinking_config is not None and sdk.thinking_config.thinking_level == types.ThinkingLevel.HIGH


@pytest.mark.parametrize(("effort", "level"), [("low", "LOW"), ("medium", "MEDIUM"), ("high", "HIGH"), ("max", "HIGH")])
def test_effort_maps_to_thinking_level(effort: str, level: str) -> None:
    assert GeminiProvider.build_config(_req(effort=effort))["thinking_config"] == {"thinking_level": level}


def test_build_config_schema_in_prompt_fallback() -> None:
    config = GeminiProvider.build_config(
        _req(system="Base.", json_schema=SCHEMA, web_search=True), schema_in_prompt=True
    )
    assert "response_json_schema" not in config and "response_mime_type" not in config
    assert config["system_instruction"].startswith("Base.\n\nRespond with a single JSON object")
    assert '"sources"' in config["system_instruction"]


def test_build_contents_maps_roles_and_places_files_in_last_user_turn() -> None:
    req = CompletionRequest(
        model=MODEL,
        messages=(
            Message(role="user", content="a"),
            Message(role="assistant", content="b"),
            Message(role="user", content="c"),
        ),
        max_output_tokens=10,
    )
    contents = GeminiProvider.build_contents(req, [("https://f/1", "application/pdf")])
    assert contents == [
        {"role": "user", "parts": [{"text": "a"}]},
        {"role": "model", "parts": [{"text": "b"}]},
        {
            "role": "user",
            "parts": [{"file_data": {"file_uri": "https://f/1", "mime_type": "application/pdf"}}, {"text": "c"}],
        },
    ]
    assert all(types.Content.model_validate(c) for c in contents)


# --- to_result ----------------------------------------------------------------------------------


def test_to_result_search_fixture() -> None:
    result = GeminiProvider.to_result(load_provider_fixture("gemini_search"), _req(web_search=True), on=ON)
    assert result.text == "## Summary\n\nITER targets Q=10 at 500 MW fusion power."  # thought part skipped
    assert result.usage == Usage(
        input_tokens=150_000 + 4_200, output_tokens=9_000 + 3_000, reasoning_tokens=3_000, search_queries=2
    )
    assert result.cost_usd == pytest.approx(_cost(result.usage))
    assert result.citations == (
        Citation(title="iter.org", uri="https://www.iter.org/mach"),
        Citation(title="Nuclear Fusion", uri="https://doi.org/10.1088/0029-5515/47/6/S01"),
    )
    assert result.stop_reason == "STOP"
    assert result.response_id == "Yx3ZaK2pLpGz"
    assert result.provider == "gemini"


def test_to_result_prices_by_call_date() -> None:
    later = date(2027, 1, 2)
    result = GeminiProvider.to_result(load_provider_fixture("gemini_search"), _req(), on=later)
    assert result.cost_usd == pytest.approx(_cost(result.usage, later))
    assert result.cost_usd > GeminiProvider.to_result(load_provider_fixture("gemini_search"), _req(), on=ON).cost_usd


def test_to_result_structured_with_cache() -> None:
    result = GeminiProvider.to_result(_response("gemini_json"), _req(json_schema=SCHEMA), on=ON)
    assert result.parsed == {"sources": ["a.pdf"], "facts": ["x"]}
    assert result.usage.cached_input_tokens == 1000
    assert result.usage.output_tokens == 1000


@pytest.mark.parametrize("name", ["gemini_safety", "gemini_blocked"])
def test_to_result_refusals(name: str) -> None:
    with pytest.raises(ProviderRefusal) as info:
        GeminiProvider.to_result(load_provider_fixture(name), _req(), on=ON)
    assert info.value.cost_usd == pytest.approx(_cost(Usage(input_tokens=700)))


def test_to_result_max_tokens_with_schema() -> None:
    with pytest.raises(StructuredOutputError, match="MAX_TOKENS"):
        GeminiProvider.to_result(load_provider_fixture("gemini_max_tokens"), _req(json_schema=SCHEMA), on=ON)


def test_to_result_no_candidates() -> None:
    with pytest.raises(ProviderError, match="no candidates"):
        GeminiProvider.to_result({"usage_metadata": {"prompt_token_count": 3}}, _req(), on=ON)


# --- complete -----------------------------------------------------------------------------------


def test_complete_passes_sdk_config() -> None:
    provider, client = _provider([_response("gemini_search")])
    req = _req(web_search=True, system="sys")
    result = provider.complete(req)
    call = client.models.calls[0]
    assert call["model"] == MODEL
    assert call["contents"] == GeminiProvider.build_contents(req)
    assert call["config"] == types.GenerateContentConfig(**GeminiProvider.build_config(req))
    assert len(result.citations) == 2


def test_complete_uploads_polls_and_deletes_attachments(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7")
    files = FakeFiles(states=["PROCESSING", "PROCESSING", "ACTIVE"])
    provider, client = _provider([_response("gemini_json")], files)
    provider.complete(_req(json_schema=SCHEMA, attachments=(Attachment(path=pdf),)))
    assert files.uploaded == [(str(pdf), {"mime_type": "application/pdf"})]
    assert files.gets == ["files/f1", "files/f1"]
    assert files.deleted == ["files/f1"]
    parts = client.models.calls[0]["contents"][-1]["parts"]
    assert parts[0] == {
        "file_data": {
            "file_uri": "https://generativelanguage.googleapis.com/v1beta/files/f1",
            "mime_type": "application/pdf",
        }
    }


def test_complete_file_processing_failure_still_deletes(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    files = FakeFiles(states=["PROCESSING", "FAILED"])
    provider, client = _provider([], files)
    with pytest.raises(ProviderError, match="processing failed"):
        provider.complete(_req(attachments=(Attachment(path=video),)))
    assert files.deleted == ["files/f1"]
    assert client.models.calls == []


def test_complete_missing_attachment(tmp_path: Path) -> None:
    provider, _ = _provider([])
    with pytest.raises(ProviderError, match="not found"):
        provider.complete(_req(attachments=(Attachment(path=tmp_path / "nope.pdf"),)))


def test_complete_deletes_uploads_when_generation_fails(tmp_path: Path) -> None:
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF")
    files = FakeFiles()
    provider, _ = _provider(
        [errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}})], files
    )
    with pytest.raises(ProviderError, match="503") as info:
        provider.complete(_req(attachments=(Attachment(path=pdf),)))
    assert info.value.retryable
    assert files.deleted == ["files/f1"]


def test_schema_with_search_falls_back_to_prompt_and_remembers() -> None:
    rejection = errors.ClientError(
        400,
        {
            "error": {
                "code": 400,
                "message": "Tool use with a response mime type: 'application/json' is unsupported",
                "status": "INVALID_ARGUMENT",
            }
        },
    )
    provider, client = _provider([rejection, _response("gemini_json"), _response("gemini_json")])
    req = _req(json_schema=SCHEMA, web_search=True)
    assert provider.complete(req).parsed == {"sources": ["a.pdf"], "facts": ["x"]}
    first, second = client.models.calls[0]["config"], client.models.calls[1]["config"]
    assert first.response_json_schema == SCHEMA
    assert second.response_json_schema is None and "JSON Schema" in second.system_instruction
    provider.complete(req)  # later calls go straight to the fallback
    assert len(client.models.calls) == 3
    assert client.models.calls[2]["config"].response_json_schema is None


def test_client_error_without_combination_is_not_retried() -> None:
    rejection = errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}})
    provider, client = _provider([rejection])
    with pytest.raises(ProviderError, match="400") as info:
        provider.complete(_req(json_schema=SCHEMA))
    assert not info.value.retryable
    assert len(client.models.calls) == 1


def test_unrelated_400_with_search_and_schema_does_not_fall_back() -> None:
    rejection = errors.ClientError(
        400, {"error": {"code": 400, "message": "Request payload size exceeds the limit", "status": "INVALID_ARGUMENT"}}
    )
    provider, client = _provider([rejection])
    with pytest.raises(ProviderError, match="400"):
        provider.complete(_req(json_schema=SCHEMA, web_search=True))
    assert len(client.models.calls) == 1
    assert provider._schema_with_search_rejected == set()


def test_read_timeout_is_charged_and_retryable() -> None:
    import httpx

    provider, _ = _provider([httpx.ReadTimeout("timed out")])
    with pytest.raises(ProviderError) as info:
        provider.complete(_req())
    assert info.value.retryable is True
    assert info.value.cost_usd == pytest.approx(provider.worst_case_cost(_req()))


def test_connect_error_is_free() -> None:
    import httpx

    provider, _ = _provider([httpx.ConnectError("refused")])
    with pytest.raises(ProviderError) as info:
        provider.complete(_req())
    assert info.value.retryable is True and info.value.cost_usd == 0.0


def test_missing_api_key() -> None:
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        GeminiProvider(today=lambda: ON).complete(_req())


def test_lazy_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from google import genai

    monkeypatch.setenv("GEMINI_API_KEY", "test-not-real")
    provider = GeminiProvider(timeout_s=90.0)
    client = provider._get_client()
    assert isinstance(client, genai.Client)
    assert client._api_client._http_options.retry_options.attempts == 1  # no SDK retries of billed calls
    assert provider._get_client() is client


def test_worst_case_cost_includes_search_fees_only_with_search() -> None:
    provider = GeminiProvider(today=lambda: ON)
    p = price_for(MODEL, ON)
    plain = _req()
    base = estimate_request_tokens(plain) * p.input_per_mtok / 1e6 + 32_000 * p.output_per_mtok / 1e6
    assert provider.worst_case_cost(plain) == pytest.approx(base)
    searching = _req(web_search=True, max_search_queries=10)
    assert provider.worst_case_cost(searching) == pytest.approx(base + 10 * p.search_query_usd)
