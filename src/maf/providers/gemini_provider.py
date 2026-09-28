"""Gemini through google-genai (>=2.25): large-context ingestion, search grounding, multimodal.

Owner: providers.

SDK facts, verified by introspecting google-genai 2.25.0:

- ``client.models.generate_content(model=, contents=, config=types.GenerateContentConfig(...))``.
  Config fields used: ``system_instruction``, ``max_output_tokens``, ``tools``,
  ``response_mime_type="application/json"`` + ``response_json_schema`` (a plain dict schema),
  ``thinking_config=types.ThinkingConfig(thinking_level=...)``,
  ``automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)`` (fields ``disable``,
  ``maximum_remote_calls``, ``ignore_call_history``). The adapter passes no Python-callable tools, so AFC never
  ran a function; left enabled it only routed each call through the SDK's AFC loop (still one request) and
  logged "Direct use of automatic function calling (AFC) in Models.generate_content is not recommended".
- Search grounding: ``tools=[types.Tool(google_search=types.GoogleSearch())]``. Sources are in
  ``response.candidates[0].grounding_metadata.grounding_chunks[i].web.{uri,title}`` and
  queries in ``.web_search_queries`` (one billable query each).
- Files: ``client.files.upload(file=path, config=types.UploadFileConfig(mime_type=...))``
  returns ``types.File`` (``name``, ``uri``, ``mime_type``, ``state``). Poll ``client.files.get(name=)``
  until ``state`` is ACTIVE (video/PDF processing), then pass the File (or
  ``types.Part.from_uri(file_uri=, mime_type=)``) in ``contents``. Delete after the call.
- ``response.usage_metadata``: ``prompt_token_count``, ``cached_content_token_count``,
  ``candidates_token_count``, ``thoughts_token_count``, ``tool_use_prompt_token_count``.
  Normalize: input = prompt + tool_use_prompt; output = candidates + thoughts.
- ``client.models.count_tokens`` is free but a network call. ``worst_case_cost`` uses the local estimate.
- ``client.interactions`` exists (stateful API). Not used: Python owns state.

If the model rejects ``google_search`` combined with ``response_json_schema`` (a 400 naming both), the
adapter drops the schema, asks for JSON in the prompt, and validates locally (``StructuredOutputError`` on
failure).
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable, Sequence
from datetime import date
from typing import Any

from maf.config import ModelPrice, UnknownModelPrice, price_for
from maf.providers.base import (
    Attachment,
    Citation,
    CompletionRequest,
    CompletionResult,
    ProviderError,
    ProviderRefusal,
    StructuredOutputError,
    attachment_mime_type,
    env_api_key,
    estimate_request_tokens,
    get_field,
    is_retryable_status,
    parse_json_output,
    token_worst_case,
    transport_failure_cost,
)
from maf.types import AgentName, ProviderName, Usage

PROVIDER: ProviderName = "gemini"

THINKING_LEVELS: dict[str, str] = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH", "xhigh": "HIGH", "max": "HIGH"}
"""``CompletionRequest.effort`` to ``ThinkingLevel``; Gemini tops out at HIGH."""

REFUSAL_FINISH_REASONS = frozenset(
    {
        "SAFETY",
        "RECITATION",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
    }
)
FILE_POLL_INTERVAL_S = 2.0

UploadedFile = tuple[str, str]
"""``(file_uri, mime_type)`` of an ACTIVE Files API upload."""


class GeminiProvider:
    name: ProviderName = PROVIDER
    agent: AgentName = "gemini"

    def __init__(
        self,
        client: Any | None = None,
        *,
        timeout_s: float = 600.0,
        retry_attempts: int = 1,
        today: Callable[[], date] = date.today,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """``client`` is a ``google.genai.Client`` or test double. None means build from ``GEMINI_API_KEY`` lazily.

        ``retry_attempts`` defaults to 1 (no SDK retries): genai's retry predicate includes read timeouts, which
        would re-send possibly billed generations unseen by the ledger. ``StageContext.call`` retries instead."""
        self._client = client
        self._timeout_s = timeout_s
        self._retry_attempts = retry_attempts
        self._today = today
        self._sleep = sleep
        self._lock = threading.Lock()
        self._schema_with_search_rejected: set[str] = set()

    def complete(self, request: CompletionRequest) -> CompletionResult:
        """Upload attachments, call ``generate_content``, collect citations, delete uploads."""
        on = self._today()
        _price(request.model, on)
        client = self._get_client()
        uploaded: list[Any] = []
        try:
            files = [self._upload(client, attachment, uploaded) for attachment in request.attachments]
            response = self._generate(client, request, files)
        finally:
            for file in uploaded:
                with contextlib.suppress(Exception):  # best effort: uploads expire after 48h anyway
                    client.files.delete(name=get_field(file, "name"))
        return self.to_result(response, request, on=on)

    def worst_case_cost(self, request: CompletionRequest, on: date | None = None) -> float:
        return token_worst_case(
            request.model,
            estimate_request_tokens(request),
            request.max_output_tokens,
            search_queries=request.max_search_queries if request.web_search else 0,
            on=on or self._today(),
        )

    @staticmethod
    def build_config(request: CompletionRequest, *, schema_in_prompt: bool = False) -> dict[str, Any]:
        """Pure mapping to ``GenerateContentConfig`` kwargs (unit-tested without SDK calls).

        ``schema_in_prompt`` is the search fallback: no ``response_json_schema``; the schema goes in
        the system instruction instead and the output is validated locally."""
        config: dict[str, Any] = {
            "max_output_tokens": request.max_output_tokens,
            "automatic_function_calling": {"disable": True},  # exactly one request per call, no AFC warning
        }
        system = request.system
        if request.json_schema is not None:
            if schema_in_prompt:
                system = "\n\n".join(s for s in (system, schema_instruction(request.json_schema)) if s)
            else:
                config["response_mime_type"] = "application/json"
                config["response_json_schema"] = request.json_schema
        if system:
            config["system_instruction"] = system
        if request.web_search:
            config["tools"] = [{"google_search": {}}]
        if request.effort is not None:
            config["thinking_config"] = {"thinking_level": THINKING_LEVELS[request.effort]}
        return config

    @staticmethod
    def build_contents(request: CompletionRequest, files: Sequence[UploadedFile] = ()) -> list[dict[str, Any]]:
        """Messages as ``Content`` dicts (``assistant`` becomes ``model``). Uploaded files are placed
        before the text of the last user turn."""
        contents: list[dict[str, Any]] = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in request.messages
        ]
        if files:
            last_user = max(i for i, c in enumerate(contents) if c["role"] == "user")
            file_parts = [{"file_data": {"file_uri": uri, "mime_type": mime}} for uri, mime in files]
            contents[last_user]["parts"] = file_parts + contents[last_user]["parts"]
        return contents

    @staticmethod
    def to_result(response: Any, request: CompletionRequest, *, on: date | None = None) -> CompletionResult:
        """Pure mapping from a ``GenerateContentResponse`` (or fixture dict) to ``CompletionResult``.

        Blocked prompts and safety finishes raise ``ProviderRefusal``; truncated or invalid JSON raises
        ``StructuredOutputError``. Both carry the billed cost."""
        if isinstance(response, dict):
            from google.genai import types

            response = types.GenerateContentResponse.model_validate(response)
        candidates = get_field(response, "candidates", [])
        candidate = candidates[0] if candidates else None
        grounding = get_field(candidate, "grounding_metadata")
        usage = usage_from_response(get_field(response, "usage_metadata"), grounding)
        cost = _price(request.model, on or date.today()).cost(usage)

        block_reason = _enum_name(get_field(get_field(response, "prompt_feedback"), "block_reason"))
        if block_reason:
            raise ProviderRefusal(f"Gemini blocked the prompt ({block_reason})", provider=PROVIDER, cost_usd=cost)
        if candidate is None:
            raise ProviderError("Gemini returned no candidates", provider=PROVIDER, cost_usd=cost)
        finish = _enum_name(get_field(candidate, "finish_reason"))
        if finish in REFUSAL_FINISH_REASONS:
            raise ProviderRefusal(f"Gemini stopped with finish_reason={finish}", provider=PROVIDER, cost_usd=cost)

        parts = get_field(get_field(candidate, "content"), "parts", [])
        text = "".join(get_field(p, "text", "") for p in parts if not get_field(p, "thought", False))
        parsed = None
        if request.json_schema is not None:
            if finish == "MAX_TOKENS":
                raise StructuredOutputError(
                    "Gemini structured output truncated (finish_reason=MAX_TOKENS)", provider=PROVIDER, cost_usd=cost
                )
            parsed = parse_json_output(text, request.json_schema, provider=PROVIDER, cost_usd=cost)
        return CompletionResult(
            text=text,
            parsed=parsed,
            usage=usage,
            cost_usd=cost,
            model=request.model,
            provider=PROVIDER,
            stop_reason=finish,
            response_id=get_field(response, "response_id", ""),
            citations=citations_from_grounding(grounding),
            raw=_dump(response),
        )

    def _generate(self, client: Any, request: CompletionRequest, files: Sequence[UploadedFile]) -> Any:
        """Call ``generate_content``; if schema plus search is rejected, retry once with the schema in the prompt.
        A 400 is not billed, so the retry does not double-spend."""
        from google.genai import types

        contents = self.build_contents(request, files)
        worst = self.worst_case_cost(request)
        combined = request.web_search and request.json_schema is not None
        with self._lock:
            in_prompt = combined and request.model in self._schema_with_search_rejected
        try:
            return client.models.generate_content(
                model=request.model,
                contents=contents,
                config=types.GenerateContentConfig(**self.build_config(request, schema_in_prompt=in_prompt)),
            )
        except Exception as exc:
            if not (combined and not in_prompt and is_schema_with_search_rejection(exc)):
                raise _wrap_error(exc, worst_case_usd=worst) from exc
        with self._lock:
            self._schema_with_search_rejected.add(request.model)
        try:
            return client.models.generate_content(
                model=request.model,
                contents=contents,
                config=types.GenerateContentConfig(**self.build_config(request, schema_in_prompt=True)),
            )
        except Exception as exc:
            raise _wrap_error(exc, worst_case_usd=worst) from exc

    def _upload(self, client: Any, attachment: Attachment, uploaded: list[Any]) -> UploadedFile:
        mime = attachment_mime_type(attachment)
        if not attachment.path.is_file():
            raise ProviderError(f"attachment not found: {attachment.path}", provider=PROVIDER)
        try:
            file = client.files.upload(file=str(attachment.path), config={"mime_type": mime})
            uploaded.append(file)
            deadline = time.monotonic() + self._timeout_s
            while _enum_name(get_field(file, "state")) == "PROCESSING":
                if time.monotonic() > deadline:
                    raise ProviderError(f"Gemini file processing timed out: {attachment.path.name}", provider=PROVIDER)
                self._sleep(FILE_POLL_INTERVAL_S)
                file = client.files.get(name=get_field(file, "name"))
        except Exception as exc:
            raise _wrap_error(exc) from exc
        state = _enum_name(get_field(file, "state"))
        if state not in ("ACTIVE", "", "STATE_UNSPECIFIED"):
            raise ProviderError(f"Gemini file processing failed ({state}): {attachment.path.name}", provider=PROVIDER)
        return get_field(file, "uri", ""), get_field(file, "mime_type", mime)

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                key = env_api_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
                if key is None:
                    raise ProviderError("GEMINI_API_KEY is not set", provider=PROVIDER)
                from google import genai
                from google.genai import types

                self._client = genai.Client(
                    api_key=key,
                    http_options=types.HttpOptions(
                        timeout=int(self._timeout_s * 1000),
                        retry_options=types.HttpRetryOptions(attempts=self._retry_attempts),
                    ),
                )
            return self._client


def usage_from_response(usage: Any, grounding: Any = None) -> Usage:
    """Input = prompt + tool-use prompt; output = candidates + thoughts; one billable query per search query."""
    thoughts = get_field(usage, "thoughts_token_count", 0)
    return Usage(
        input_tokens=get_field(usage, "prompt_token_count", 0) + get_field(usage, "tool_use_prompt_token_count", 0),
        output_tokens=get_field(usage, "candidates_token_count", 0) + thoughts,
        cached_input_tokens=get_field(usage, "cached_content_token_count", 0),
        reasoning_tokens=thoughts,
        search_queries=len(get_field(grounding, "web_search_queries", [])),
    )


def citations_from_grounding(grounding: Any) -> tuple[Citation, ...]:
    """Web sources from grounding chunks, deduplicated by URI in first-seen order."""
    seen: dict[str, Citation] = {}
    for chunk in get_field(grounding, "grounding_chunks", []):
        web = get_field(chunk, "web")
        uri = get_field(web, "uri", "")
        if uri and uri not in seen:
            seen[uri] = Citation(title=get_field(web, "title", ""), uri=uri)
    return tuple(seen.values())


def schema_instruction(schema: dict[str, Any]) -> str:
    return (
        "Respond with a single JSON object and nothing else (no prose, no code fence). "
        "It must validate against this JSON Schema:\n" + json.dumps(schema, indent=2)
    )


def _enum_name(value: Any) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value))


_SCHEMA_TERMS = ("response_json_schema", "response_schema", "response_mime_type", "mime type", "json", "controlled generation")
_TOOL_TERMS = ("tool", "google_search", "search", "grounding")


def is_schema_with_search_rejection(exc: Exception) -> bool:
    """A 400 whose message names both the search tool and the structured-output config. Other 400s (bad
    schema, bad attachment, oversized request) are real errors and must not trigger the fallback."""
    if _status(exc) != 400:
        return False
    message = (getattr(exc, "message", None) or str(exc)).lower()
    return any(t in message for t in _SCHEMA_TERMS) and any(t in message for t in _TOOL_TERMS)


def _status(exc: Exception) -> int | None:
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def _price(model: str, on: date) -> ModelPrice:
    try:
        return price_for(model, on)
    except UnknownModelPrice as exc:
        raise ProviderError(f"refusing unpriced model: {exc}", provider=PROVIDER) from None


def _dump(response: Any) -> dict[str, Any]:
    dump = getattr(response, "model_dump", None)
    if not callable(dump):
        return {}
    return dump(mode="json", exclude_none=True, exclude={"sdk_http_response", "automatic_function_calling_history"})


def _wrap_error(exc: Exception, *, worst_case_usd: float = 0.0) -> ProviderError:
    """Map any SDK exception to ``ProviderError``. google-genai raises ``errors.APIError`` with ``.code``;
    transport failures surface as httpx timeout/connect errors. A timeout after the request was sent is
    charged ``worst_case_usd`` (see ``transport_failure_cost``)."""
    if isinstance(exc, ProviderError):
        return exc
    status = _status(exc)
    if status is not None:
        message = getattr(exc, "message", None) or str(exc)
        return ProviderError(
            f"Gemini HTTP {status}: {message}", provider=PROVIDER, retryable=is_retryable_status(status)
        )
    transient = any(word in type(exc).__name__ for word in ("Timeout", "Connect", "Network", "Protocol"))
    cost = transport_failure_cost(exc, worst_case_usd) if transient else 0.0
    return ProviderError(
        f"Gemini call failed: {type(exc).__name__}: {exc}", provider=PROVIDER, cost_usd=cost, retryable=transient
    )
