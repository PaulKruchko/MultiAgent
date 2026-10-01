"""Shared provider helpers, fixture validity against the SDK types, and ``build_providers``."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from maf.config import Settings, UnknownModelPrice, price_for
from maf.providers import (
    ClaudeCodeProvider,
    ClaudeProvider,
    CompletionRequest,
    GeminiProvider,
    OpenAIProvider,
    Provider,
    Providers,
    StructuredOutputError,
    build_providers,
)
from maf.providers.base import (
    IMAGE_TOKENS,
    MESSAGE_OVERHEAD_TOKENS,
    PDF_TOKENS_PER_PAGE,
    Attachment,
    ProviderError,
    SandboxUnavailable,
    attachment_mime_type,
    estimate_attachment_tokens,
    estimate_request_tokens,
    estimate_text_tokens,
    get_field,
    is_retryable_status,
    parse_json_output,
    token_worst_case,
)
from maf.providers.claude_code import max_tmpdir_bytes

FIXTURES = Path(__file__).parent / "fixtures" / "providers"
ON = date(2026, 9, 28)


def _req(prompt: str = "hello", **kw: Any) -> CompletionRequest:
    return CompletionRequest.simple("claude-opus-5-5", prompt, max_output_tokens=1000, **kw)


# --- token_worst_case -------------------------------------------------------------------------


def test_token_worst_case_prices_input_at_max_of_input_and_cache_write() -> None:
    price = price_for("claude-opus-5-5", ON)
    rate = max(price.input_per_mtok, price.cache_write_per_mtok or 0.0)
    got = token_worst_case("claude-opus-5-5", 10_000, 2_000, on=ON)
    assert got == pytest.approx(10_000 * rate / 1e6 + 2_000 * price.output_per_mtok / 1e6)


def test_token_worst_case_adds_search_fees_and_follows_price_dates() -> None:
    old = price_for("gemini-3.8-flash", ON)
    new = price_for("gemini-3.8-flash", date(2027, 1, 1))
    for on, price in ((ON, old), (date(2027, 1, 1), new)):
        got = token_worst_case("gemini-3.8-flash", 1_000_000, 100_000, search_queries=20, on=on)
        expected = price.input_per_mtok + 0.1 * price.output_per_mtok + 20 * price.search_query_usd
        assert got == pytest.approx(expected)
    assert new.input_per_mtok > old.input_per_mtok


def test_token_worst_case_unknown_model_is_refused() -> None:
    with pytest.raises(UnknownModelPrice):
        token_worst_case("gpt-2", 1, 1, on=ON)


# --- estimates --------------------------------------------------------------------------------


def test_estimate_text_tokens_uses_three_chars_per_token() -> None:
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("abc") == 1
    assert estimate_text_tokens("abcd") == 2


def test_estimate_request_tokens_counts_system_messages_and_schema() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    req = _req("x" * 300, system="y" * 30, json_schema=schema)
    expected = 10 + 100 + MESSAGE_OVERHEAD_TOKENS + estimate_text_tokens(json.dumps(schema))
    assert estimate_request_tokens(req) == expected


def test_attachment_estimates(tmp_path: Path) -> None:
    png = tmp_path / "plot.png"
    png.write_bytes(b"\x89PNG" + b"0" * 5000)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\n" + b"<< /Type /Page >>\n" * 7 + b"<< /Type /Pages /Count 7 >>\n")
    opaque_pdf = tmp_path / "compressed.pdf"
    opaque_pdf.write_bytes(b"%PDF-1.7\n".ljust(100_000, b"\x00"))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00" * 250_000)  # 2 Mbit -> at most 8 s at 250 kbps
    notes = tmp_path / "notes.csv"
    notes.write_text("a,b\n" * 300)

    assert estimate_attachment_tokens(Attachment(path=png)) == IMAGE_TOKENS
    assert estimate_attachment_tokens(Attachment(path=pdf)) == 7 * PDF_TOKENS_PER_PAGE
    assert estimate_attachment_tokens(Attachment(path=opaque_pdf)) == 5 * PDF_TOKENS_PER_PAGE
    assert estimate_attachment_tokens(Attachment(path=video)) == 8 * 300
    assert estimate_attachment_tokens(Attachment(path=notes)) == 400
    assert estimate_attachment_tokens(Attachment(path=tmp_path / "missing.png")) == 0
    req = _req("", attachments=(Attachment(path=png), Attachment(path=pdf)))
    assert estimate_request_tokens(req) == MESSAGE_OVERHEAD_TOKENS + IMAGE_TOKENS + 7 * PDF_TOKENS_PER_PAGE


def test_attachment_mime_type() -> None:
    assert attachment_mime_type(Attachment(path=Path("a.pdf"))) == "application/pdf"
    assert attachment_mime_type(Attachment(path=Path("a.bin"), mime_type="image/png")) == "image/png"
    assert attachment_mime_type(Attachment(path=Path("noext"))) == "application/octet-stream"


def test_url_context_is_an_opt_in_request_field() -> None:
    assert _req().url_context is False
    request = _req(web_search=True, url_context=True)
    assert request.url_context is True
    assert estimate_request_tokens(request) == estimate_request_tokens(_req())  # the provider prices fetched pages


# --- structured output -------------------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {"mode": {"type": "string", "enum": ["code", "prose"]}},
    "required": ["mode"],
    "additionalProperties": False,
}


def test_parse_json_output_accepts_plain_and_fenced() -> None:
    assert parse_json_output('{"mode": "code"}', SCHEMA, provider="openai", cost_usd=0.1) == {"mode": "code"}
    fenced = '```json\n{"mode": "prose"}\n```'
    assert parse_json_output(fenced, SCHEMA, provider="openai", cost_usd=0.1) == {"mode": "prose"}


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("not json", "not valid JSON"),
        ("[1, 2]", "expected a JSON object"),
        ('{"mode": "poetry"}', "does not match schema at mode"),
        ('{"mode": "code", "x": 1}', "does not match schema"),
        ("{}", "does not match schema at <root>"),
    ],
)
def test_parse_json_output_failures_carry_cost(text: str, fragment: str) -> None:
    with pytest.raises(StructuredOutputError, match=fragment) as info:
        parse_json_output(text, SCHEMA, provider="gemini", cost_usd=0.25)
    assert info.value.cost_usd == 0.25
    assert info.value.provider == "gemini"
    assert not info.value.retryable


def test_parse_json_output_invalid_schema() -> None:
    with pytest.raises(StructuredOutputError, match="invalid JSON schema"):
        parse_json_output("{}", {"type": 12}, provider="openai", cost_usd=0)


# --- small helpers ---------------------------------------------------------------------------


def test_sandbox_unavailable_is_a_never_retryable_provider_error() -> None:
    exc = SandboxUnavailable("bridge sockets", provider="claude_code", cost_usd=1.25)
    assert isinstance(exc, ProviderError) and not isinstance(exc, StructuredOutputError)
    assert (exc.provider, exc.cost_usd, exc.retryable, str(exc)) == ("claude_code", 1.25, False, "bridge sockets")
    assert SandboxUnavailable("x", provider="claude_code").cost_usd == 0.0
    with pytest.raises(TypeError):
        SandboxUnavailable("x", provider="claude_code", retryable=True)  # type: ignore[call-arg]


def test_get_field_handles_dicts_objects_and_none() -> None:
    class Obj:
        a = 1
        b = None

    assert get_field({"a": 1}, "a") == 1
    assert get_field({"a": None}, "a", 5) == 5
    assert get_field(Obj(), "a") == 1
    assert get_field(Obj(), "b", 7) == 7
    assert get_field(None, "a", 3) == 3


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(400, False), (401, False), (408, True), (429, True), (500, True), (529, True), (None, False)],
)
def test_is_retryable_status(status: int | None, retryable: bool) -> None:
    assert is_retryable_status(status) is retryable


# --- fixtures are genuine SDK payloads ---------------------------------------------------------


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.stem)
def test_fixture_validates_against_sdk_type(path: Path) -> None:
    payload = json.loads(path.read_text())
    prefix = path.stem.split("_")[0]
    if prefix == "openai":
        from openai.types.responses import Response

        Response.model_validate(payload)
    elif prefix == "anthropic":
        from anthropic.types import Message

        Message.model_validate(payload)
    elif prefix == "gemini":
        from google.genai import types

        types.GenerateContentResponse.model_validate(payload)  # extra="forbid": catches typos
    else:
        from maf.providers.claude_code import ClaudeCodeProvider

        assert payload["type"] == "result"
        ClaudeCodeProvider.parse_output(json.dumps(payload))


# --- build_providers ---------------------------------------------------------------------------


def test_build_providers_constructs_all_adapters_without_network(settings: Settings, tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "run-1"
    providers = build_providers(settings, workspace)
    assert isinstance(providers, Providers)
    assert isinstance(providers.chatgpt, OpenAIProvider)
    assert isinstance(providers.gemini, GeminiProvider)
    assert isinstance(providers.claude, ClaudeProvider)
    assert isinstance(providers.claude_code, ClaudeCodeProvider)
    for provider in (providers.chatgpt, providers.gemini, providers.claude, providers.claude_code):
        assert isinstance(provider, Provider)
    assert [p.agent for p in (providers.chatgpt, providers.gemini, providers.claude, providers.claude_code)] == [
        "chatgpt",
        "gemini",
        "claude",
        "claude",
    ]
    assert providers.for_role("claude_code") is providers.claude_code
    code = providers.claude_code
    assert isinstance(code, ClaudeCodeProvider)
    assert code.workspace == workspace
    assert code.executable == settings.claude_executable
    assert code.allowed_tools == settings.claude_code_tools
    assert code.timeout_s == settings.claude_code_timeout_s
    assert code.sandbox_verified is False
    assert len(str(code.tmpdir)) <= max_tmpdir_bytes() and not code.tmpdir.is_relative_to(workspace)
    assert code.build_env()["TMPDIR"] == str(code.tmpdir)
    assert code.bash_timeout_s == settings.bash_timeout_s == 0.75 * settings.claude_code_timeout_s
    assert code.build_env()["BASH_MAX_TIMEOUT_MS"] == str(int(settings.bash_timeout_s * 1000))
    assert not workspace.exists()  # construction has no side effects


def test_build_providers_puts_project_python_first_on_path(tmp_path: Path) -> None:
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    settings = Settings(
        vault_path=tmp_path / "v", workspaces_path=tmp_path / "w", python_executable=venv_bin / "python"
    )
    code = build_providers(settings, tmp_path / "w" / "r").claude_code
    assert isinstance(code, ClaudeCodeProvider)
    assert code.build_env()["PATH"].split(":")[0] == str(venv_bin)


def test_build_providers_passes_the_tmp_base_through(tmp_path: Path) -> None:
    settings = Settings(vault_path=tmp_path / "v", workspaces_path=tmp_path / "w", claude_code_tmp_base=Path("/var/t"))
    code = build_providers(settings, tmp_path / "w" / "r").claude_code
    assert isinstance(code, ClaudeCodeProvider)
    assert code.tmp_base == Path("/var/t") and code.tmpdir.parent == Path("/var/t")
    assert code.build_env()["TMPDIR"] == str(code.tmpdir)


def test_sandbox_unavailable_is_exported_as_a_provider_error() -> None:
    from maf import providers

    assert providers.SandboxUnavailable is SandboxUnavailable
    assert issubclass(providers.SandboxUnavailable, providers.ProviderError)
