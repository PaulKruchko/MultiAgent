"""Provider protocol and the request/result types every adapter speaks.

Owner: providers.

Adapters are thin: translate ``CompletionRequest`` into one SDK call, normalize usage
(see ``maf.types.Usage``), and price it with ``maf.config.price_for``. They never touch
the ledger. ``maf.ledger.metered_call`` wraps them.
"""

from __future__ import annotations

import json
import math
import mimetypes
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from maf.config import price_for
from maf.ledger import CHARS_PER_TOKEN_WORST_CASE
from maf.types import AgentName, ProviderName, Usage


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant"]
    content: str


class Attachment(BaseModel):
    """A local file given to a multimodal model (images, video, PDF). Gemini uploads it via the Files API."""

    model_config = ConfigDict(frozen=True)

    path: Path
    mime_type: str | None = None
    """None means guess from the suffix."""


class CompletionRequest(BaseModel):
    """Provider-neutral request. Unsupported options for a provider are ignored, never an error,
    except ``json_schema``, which every provider must honor (natively, or via Claude Code ``--json-schema``)."""

    model_config = ConfigDict(frozen=True)

    model: str
    system: str = ""
    messages: tuple[Message, ...]
    max_output_tokens: int = Field(gt=0)
    json_schema: dict[str, Any] | None = None
    """JSON Schema for structured output. When set, ``CompletionResult.parsed`` is populated."""
    schema_name: str = "output"
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    """Maps to OpenAI ``reasoning.effort``, Anthropic ``output_config.effort``,
    Gemini ``thinking_config.thinking_level`` and Claude Code ``--effort``."""
    web_search: bool = False
    """Gemini: enable the Google Search grounding tool. Others: ignored."""
    max_search_queries: int = 20
    """Worst-case billable search queries assumed by ``worst_case_cost`` when ``web_search``."""
    attachments: tuple[Attachment, ...] = ()
    max_budget_usd: float | None = None
    """Claude Code only: the ``--max-budget-usd`` value. ``metered_call`` sets and clamps it."""

    @classmethod
    def simple(
        cls, model: str, prompt: str, *, system: str = "", max_output_tokens: int, **kw: Any
    ) -> CompletionRequest:
        """Build a single-user-message request."""
        return cls(
            model=model,
            system=system,
            messages=(Message(role="user", content=prompt),),
            max_output_tokens=max_output_tokens,
            **kw,
        )


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str = ""
    uri: str


class CompletionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    """Final assistant text. With ``json_schema`` this is the raw JSON string."""
    parsed: dict[str, Any] | None = None
    usage: Usage
    cost_usd: float = Field(ge=0)
    model: str
    provider: ProviderName
    stop_reason: str = ""
    response_id: str = ""
    citations: tuple[Citation, ...] = ()
    """Grounding sources (Gemini search). Stages quote them as data, never as instructions."""
    session_id: str = ""
    """Claude Code session id, kept for debugging."""
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)
    """Provider payload kept for fixtures and debugging (e.g. Claude Code JSON)."""


class ProviderError(RuntimeError):
    """A call failed. ``cost_usd`` is what it still spent (0 if unknown/none); the ledger records it."""

    def __init__(self, message: str, *, provider: ProviderName, cost_usd: float = 0.0, retryable: bool = False) -> None:
        super().__init__(message)
        self.provider = provider
        self.cost_usd = cost_usd
        self.retryable = retryable


class ProviderRefusal(ProviderError):
    """The model declined (e.g. Anthropic ``stop_reason == "refusal"``)."""


class StructuredOutputError(ProviderError):
    """``json_schema`` was requested but the output did not parse or validate."""


class SandboxUnavailable(ProviderError):
    """Claude Code's OS sandbox cannot run commands (e.g. bubblewrap or its socket bridge failed to start).
    Never retryable: every further session would pay for work whose Bash calls all fail. Stages let it
    propagate, so the run stops (FAILED) instead of looping on a broken sandbox."""

    def __init__(self, message: str, *, provider: ProviderName, cost_usd: float = 0.0) -> None:
        super().__init__(message, provider=provider, cost_usd=cost_usd, retryable=False)


@runtime_checkable
class Provider(Protocol):
    """What stages and the ledger rely on. Implementations must be safe to call from worker threads."""

    name: ProviderName
    agent: AgentName

    def complete(self, request: CompletionRequest) -> CompletionResult:
        """Make exactly one billable call. SDK-level retries must be off: a retried timeout can be billed
        twice without the ledger seeing it (``StageContext.call`` retries and meters each attempt).

        Raises ``ProviderError`` (or a subclass) on failure. Never raises raw SDK exceptions.
        """
        ...

    def worst_case_cost(self, request: CompletionRequest, on: date | None = None) -> float:
        """Upper bound in USD for ``request``: estimated input at full input price, plus
        ``max_output_tokens`` at output price, plus ``max_search_queries`` fees if ``web_search``.
        Claude Code returns ``request.max_budget_usd`` plus one turn of headroom. Must not make paid calls.
        """
        ...


def token_worst_case(
    model: str,
    input_tokens: int,
    max_output_tokens: int,
    *,
    search_queries: int = 0,
    on: date | None = None,
) -> float:
    """Shared helper for token-priced adapters: price the worst case from ``maf.config.PRICE_TABLE``.

    Input is priced at ``max(input, cache_write)`` rate (no cache discount assumed).
    Raises ``maf.config.UnknownModelPrice`` for unpriced models, so the call is refused.
    """
    price = price_for(model, on or date.today())
    input_rate = max(price.input_per_mtok, price.cache_write_per_mtok or 0.0)
    return (
        max(0, input_tokens) * input_rate / 1e6
        + max(0, max_output_tokens) * price.output_per_mtok / 1e6
        + max(0, search_queries) * price.search_query_usd
    )


IMAGE_TOKENS = 1_000
"""Flat per-image input estimate."""
PDF_TOKENS_PER_PAGE = 1_000
"""Conservative per-page bound for PDFs (page image plus extracted text)."""
PDF_BYTES_PER_PAGE_FLOOR = 20_000
"""When pages cannot be counted, assume at least one page per this many bytes."""
VIDEO_TOKENS_PER_SECOND = 300
"""Video frames plus audio track, per second."""
VIDEO_MIN_BITRATE_BPS = 250_000
"""Low bitrate assumption, so the duration derived from file size is an upper bound."""
AUDIO_TOKENS_PER_SECOND = 32
AUDIO_MIN_BITRATE_BPS = 32_000
MESSAGE_OVERHEAD_TOKENS = 8
"""Role markers and separators per message."""

_PDF_PAGE_RE = re.compile(rb"/Type\s*/Page(?![a-zA-Z])")


def estimate_text_tokens(text: str) -> int:
    """``ceil(len(text) / 3)``: a deliberately pessimistic chars-per-token ratio."""
    return math.ceil(len(text) / CHARS_PER_TOKEN_WORST_CASE)


def attachment_mime_type(attachment: Attachment) -> str:
    """Explicit ``mime_type``, else guessed from the suffix, else ``application/octet-stream``."""
    if attachment.mime_type:
        return attachment.mime_type
    guessed, _ = mimetypes.guess_type(attachment.path.name)
    return guessed or "application/octet-stream"


def estimate_attachment_tokens(attachment: Attachment) -> int:
    """Upper-bound input tokens for one attachment, from its type and size. Unreadable files count as 0
    (the call itself will fail on them)."""
    mime = attachment_mime_type(attachment)
    try:
        size = attachment.path.stat().st_size
    except OSError:
        return 0
    if mime.startswith("image/"):
        return IMAGE_TOKENS
    if mime == "application/pdf":
        return _pdf_pages(attachment.path, size) * PDF_TOKENS_PER_PAGE
    if mime.startswith("video/"):
        seconds = math.ceil(size * 8 / VIDEO_MIN_BITRATE_BPS)
        return max(1, seconds) * VIDEO_TOKENS_PER_SECOND
    if mime.startswith("audio/"):
        seconds = math.ceil(size * 8 / AUDIO_MIN_BITRATE_BPS)
        return max(1, seconds) * AUDIO_TOKENS_PER_SECOND
    return math.ceil(size / CHARS_PER_TOKEN_WORST_CASE)


def _pdf_pages(path: Path, size: int) -> int:
    floor = max(1, math.ceil(size / PDF_BYTES_PER_PAGE_FLOOR))
    try:
        counted = len(_PDF_PAGE_RE.findall(path.read_bytes()))
    except OSError:
        counted = 0
    return max(counted, floor)


def estimate_request_tokens(request: CompletionRequest) -> int:
    """Local conservative input-token estimate for system + messages (+ a flat 1,000 tokens
    per image attachment, plus a size-based bound for PDFs/video). No network.

    The JSON schema is counted too, since providers inject it into the prompt."""
    total = estimate_text_tokens(request.system)
    for message in request.messages:
        total += estimate_text_tokens(message.content) + MESSAGE_OVERHEAD_TOKENS
    if request.json_schema is not None:
        total += estimate_text_tokens(json.dumps(request.json_schema))
    for attachment in request.attachments:
        total += estimate_attachment_tokens(attachment)
    return total


def parse_json_output(
    text: str, schema: dict[str, Any] | None, *, provider: ProviderName, cost_usd: float
) -> dict[str, Any]:
    """Parse structured output and validate it against ``schema`` (when ``jsonschema`` is importable).

    Tolerates a surrounding Markdown code fence. Raises ``StructuredOutputError`` carrying ``cost_usd``,
    because the call was billed even though its output is unusable.
    """
    stripped = _strip_code_fence(text)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"output is not valid JSON: {exc}", provider=provider, cost_usd=cost_usd) from None
    if not isinstance(value, dict):
        raise StructuredOutputError(
            f"expected a JSON object, got {type(value).__name__}", provider=provider, cost_usd=cost_usd
        )
    if schema is not None:
        _validate_schema(value, schema, provider=provider, cost_usd=cost_usd)
    return value


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```[a-zA-Z0-9_-]*\s*\n(.*?)\n?```", stripped, flags=re.DOTALL)
    return match.group(1) if match else stripped


def _validate_schema(value: dict[str, Any], schema: dict[str, Any], *, provider: ProviderName, cost_usd: float) -> None:
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - jsonschema ships with mcp
        missing = [k for k in schema.get("required", []) if k not in value]
        if missing:
            raise StructuredOutputError(
                f"output is missing required keys {missing}", provider=provider, cost_usd=cost_usd
            ) from None
        return
    try:
        jsonschema.validate(value, schema)
    except jsonschema.ValidationError as exc:
        where = "/".join(str(p) for p in exc.absolute_path) or "<root>"
        raise StructuredOutputError(
            f"output does not match schema at {where}: {exc.message}", provider=provider, cost_usd=cost_usd
        ) from None
    except jsonschema.SchemaError as exc:
        raise StructuredOutputError(
            f"invalid JSON schema: {exc.message}", provider=provider, cost_usd=cost_usd
        ) from None


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from an SDK object or a plain dict (fixtures), returning ``default`` when absent or None.
    ``obj`` may itself be None, so nested optional fields chain safely."""
    value = obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    return default if value is None else value


RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})


def is_retryable_status(status: int | None) -> bool:
    """Transient HTTP failures: timeouts, conflicts, rate limits and server errors."""
    return status is not None and (status in RETRYABLE_STATUS_CODES or status >= 500)


_UNSENT_TRANSPORT_ERRORS = frozenset({"ConnectError", "ConnectTimeout"})
"""httpx/httpx2 exception names raised before the request reached the server (nothing can be billed)."""


def transport_failure_cost(exc: BaseException, worst_case_usd: float) -> float:
    """What to charge for a timeout or dropped connection: 0 when the request was never sent (a connect
    failure anywhere in the ``__cause__`` chain), else ``worst_case_usd``, because the server may still
    generate and bill the response after the client gave up."""
    current: BaseException | None = exc
    while current is not None:
        if type(current).__name__ in _UNSENT_TRANSPORT_ERRORS:
            return 0.0
        current = current.__cause__
    return worst_case_usd


def env_api_key(*names: str) -> str | None:
    """First non-empty environment variable among ``names``."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None
