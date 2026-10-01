"""ChatGPT through the OpenAI Responses API (openai>=3.20).

Owner: providers.

SDK facts, verified by introspecting openai 3.20.0:

- ``client.responses.create(model=, instructions=, input=, max_output_tokens=,
  reasoning={"effort": ...}, text={"format": {"type": "json_schema", "name": ...,
  "schema": ..., "strict": True}}, store=False)``.
- ``response.output_text`` is the concatenated text. ``response.status`` and
  ``response.incomplete_details`` report truncation.
- ``response.usage``: ``input_tokens`` (includes cached), ``input_tokens_details.cached_tokens``,
  ``input_tokens_details.cache_write_tokens`` (required in 3.20), ``output_tokens`` (includes
  reasoning), ``output_tokens_details.reasoning_tokens``.
- ``client.responses.parse(text_format=PydanticModel, ...)`` also exists. We use raw JSON Schema
  via ``text.format`` so schemas stay dicts, and parse with ``json.loads``.

Strict JSON Schema requires ``additionalProperties: false`` and every property listed in
``required``. ``maf.handoff`` and stage schemas are written to satisfy that.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import date
from typing import Any

from maf.config import ModelPrice, UnknownModelPrice, price_for
from maf.providers.base import (
    CompletionRequest,
    CompletionResult,
    ProviderError,
    ProviderRefusal,
    StructuredOutputError,
    env_api_key,
    estimate_request_tokens,
    get_field,
    is_retryable_status,
    parse_json_output,
    token_worst_case,
    transport_failure_cost,
)
from maf.types import AgentName, ProviderName, Usage

PROVIDER: ProviderName = "openai"


class OpenAIProvider:
    name: ProviderName = PROVIDER
    agent: AgentName = "chatgpt"

    def __init__(
        self,
        client: Any | None = None,
        *,
        timeout_s: float = 600.0,
        max_retries: int = 0,
        today: Callable[[], date] = date.today,
    ) -> None:
        """``client`` is an ``openai.OpenAI`` (or a test double with ``.responses.create``).
        None means construct one from ``OPENAI_API_KEY`` lazily on first call.

        ``max_retries`` defaults to 0: the SDK would silently re-send timed-out (possibly billed) requests.
        ``StageContext.call`` retries instead, metering every attempt."""
        self._client = client
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._today = today
        self._lock = threading.Lock()

    def complete(self, request: CompletionRequest) -> CompletionResult:
        """One ``responses.create`` call. ``system`` becomes ``instructions``; ``messages`` become
        ``input`` items. Raises ``ProviderError`` if status is ``incomplete`` due to
        ``max_output_tokens`` with structured output requested (truncated JSON is useless)."""
        on = self._today()
        _price(request.model, on)  # refuse unpriced models before spending anything
        worst = self.worst_case_cost(request, on)
        client = self._get_client()
        try:
            response = client.responses.create(**self.build_params(request))
        except Exception as exc:
            raise _wrap_error(exc, worst_case_usd=worst) from exc
        return self.to_result(response, request, on=on)

    def worst_case_cost(self, request: CompletionRequest, on: date | None = None) -> float:
        return token_worst_case(
            request.model, estimate_request_tokens(request), request.max_output_tokens, on=on or self._today()
        )

    @staticmethod
    def build_params(request: CompletionRequest) -> dict[str, Any]:
        """Pure mapping from request to ``responses.create`` kwargs (unit-tested without SDK)."""
        params: dict[str, Any] = {
            "model": request.model,
            "input": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_output_tokens": request.max_output_tokens,
            "store": False,
        }
        if request.system:
            params["instructions"] = request.system
        if request.effort is not None:
            params["reasoning"] = {"effort": request.effort}
        if request.json_schema is not None:
            params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.schema_name,
                    "schema": request.json_schema,
                    "strict": True,
                }
            }
        return params

    @staticmethod
    def to_result(response: Any, request: CompletionRequest, *, on: date | None = None) -> CompletionResult:
        """Pure mapping from a ``Response`` (or recorded fixture dict) to ``CompletionResult``.

        Priced at ``request.model`` (the response may name a dated snapshot). Raises
        ``ProviderRefusal`` on a refusal or content filter, ``StructuredOutputError`` on truncated or
        invalid JSON, and ``ProviderError`` on a failed response; each carries the billed cost.
        """
        if isinstance(response, dict):
            from openai.types.responses import Response

            response = Response.model_validate(response)
        usage = usage_from_response(get_field(response, "usage"))
        cost = _price(request.model, on or date.today()).cost(usage)
        status = get_field(response, "status", "completed")
        reason = get_field(get_field(response, "incomplete_details"), "reason", "")
        text = get_field(response, "output_text", "")
        refusal = _refusal_text(response)

        if status == "failed":
            error = get_field(response, "error")
            detail = f"{get_field(error, 'code', '')}: {get_field(error, 'message', '')}" if error else "no detail"
            raise ProviderError(f"OpenAI response failed ({detail})", provider=PROVIDER, cost_usd=cost)
        if refusal and not text:
            raise ProviderRefusal(f"OpenAI model refused: {refusal}", provider=PROVIDER, cost_usd=cost)
        if status == "incomplete" and reason == "content_filter":
            raise ProviderRefusal("OpenAI response stopped by the content filter", provider=PROVIDER, cost_usd=cost)
        if status == "incomplete" and request.json_schema is not None:
            raise StructuredOutputError(
                f"OpenAI structured output truncated (incomplete: {reason or 'unknown'})",
                provider=PROVIDER,
                cost_usd=cost,
            )
        if status not in ("completed", "incomplete"):
            raise ProviderError(f"OpenAI response ended with status {status!r}", provider=PROVIDER, cost_usd=cost)

        parsed = (
            parse_json_output(text, request.json_schema, provider=PROVIDER, cost_usd=cost)
            if request.json_schema is not None
            else None
        )
        return CompletionResult(
            text=text,
            parsed=parsed,
            usage=usage,
            cost_usd=cost,
            model=request.model,
            provider=PROVIDER,
            stop_reason=reason or status,
            response_id=get_field(response, "id", ""),
            raw=_dump(response),
        )

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                key = env_api_key("OPENAI_API_KEY")
                if key is None:
                    raise ProviderError("OPENAI_API_KEY is not set", provider=PROVIDER)
                import openai

                self._client = openai.OpenAI(api_key=key, timeout=self._timeout_s, max_retries=self._max_retries)
            return self._client


def usage_from_response(usage: Any) -> Usage:
    """OpenAI already reports totals: input includes cached reads and writes, output includes reasoning."""
    input_details = get_field(usage, "input_tokens_details")
    return Usage(
        input_tokens=get_field(usage, "input_tokens", 0),
        output_tokens=get_field(usage, "output_tokens", 0),
        cached_input_tokens=get_field(input_details, "cached_tokens", 0),
        cache_write_tokens=get_field(input_details, "cache_write_tokens", 0),
        reasoning_tokens=get_field(get_field(usage, "output_tokens_details"), "reasoning_tokens", 0),
    )


def _refusal_text(response: Any) -> str:
    parts: list[str] = []
    for item in get_field(response, "output", []):
        if get_field(item, "type") != "message":
            continue
        for content in get_field(item, "content", []):
            if get_field(content, "type") == "refusal":
                parts.append(get_field(content, "refusal", ""))
    return " ".join(p for p in parts if p)


def _price(model: str, on: date) -> ModelPrice:
    try:
        return price_for(model, on)
    except UnknownModelPrice as exc:
        raise ProviderError(f"refusing unpriced model: {exc}", provider=PROVIDER) from None


def _dump(response: Any) -> dict[str, Any]:
    dump = getattr(response, "model_dump", None)
    return dump(mode="json", exclude_none=True) if callable(dump) else {}


def _wrap_error(exc: Exception, *, worst_case_usd: float = 0.0) -> ProviderError:
    """Map any SDK exception to ``ProviderError``. Rejected requests are never billed; a timeout or dropped
    connection after the request was sent is charged ``worst_case_usd`` (see ``transport_failure_cost``)."""
    import openai

    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, openai.APIStatusError):
        return ProviderError(
            f"OpenAI HTTP {exc.status_code}: {exc.message}",
            provider=PROVIDER,
            retryable=is_retryable_status(exc.status_code),
        )
    if isinstance(exc, openai.APIConnectionError):  # includes APITimeoutError
        return ProviderError(
            f"OpenAI connection error: {exc}",
            provider=PROVIDER,
            cost_usd=transport_failure_cost(exc, worst_case_usd),
            retryable=True,
        )
    return ProviderError(f"OpenAI call failed: {type(exc).__name__}: {exc}", provider=PROVIDER)
