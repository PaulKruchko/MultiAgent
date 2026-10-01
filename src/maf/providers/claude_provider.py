"""Claude through the Anthropic Messages API (anthropic>=1.9), for writing/editing/reasoning.

Owner: providers.

SDK and model facts, verified by introspection and the claude-api skill:

- Always stream: ``with client.messages.stream(model=, max_tokens=, system=, messages=,
  thinking={"type": "adaptive"}, output_config={...}) as s: msg = s.get_final_message()``.
  Output can reach 128K, and non-streaming requests that large hit SDK HTTP timeouts.
- Structured output: ``output_config={"format": {"type": "json_schema", "schema": {...}}}``.
  Effort: ``output_config={"effort": "low"|"medium"|"high"|"xhigh"|"max"}``. Opus 5.5 defaults to
  ``medium``, so always set it explicitly (default ``high`` for this framework).
- claude-opus-5-5 and claude-fable-5-1: thinking cannot be disabled (omit ``thinking`` or send adaptive);
  no ``temperature``/``top_p``; no assistant prefill; forced ``tool_choice`` returns 400.
- ``msg.stop_reason`` may be ``"refusal"``. Raise ``ProviderRefusal``. ``"max_tokens"`` with a schema
  raises ``StructuredOutputError``.
- ``msg.usage``: ``input_tokens`` (uncached only), ``cache_read_input_tokens``,
  ``cache_creation_input_tokens``, ``output_tokens``. Normalize input = sum of the three.
- Text = concatenation of ``text`` blocks only (skip ``thinking`` blocks).
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

DEFAULT_EFFORT = "high"
PROVIDER: ProviderName = "anthropic"


class ClaudeProvider:
    name: ProviderName = PROVIDER
    agent: AgentName = "claude"

    def __init__(
        self,
        client: Any | None = None,
        *,
        timeout_s: float = 600.0,
        max_retries: int = 0,
        today: Callable[[], date] = date.today,
    ) -> None:
        """``client`` is an ``anthropic.Anthropic`` or test double. None means build from ``ANTHROPIC_API_KEY`` lazily.
        ``max_retries`` defaults to 0 so every attempt is metered (``StageContext.call`` retries)."""
        self._client = client
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._today = today
        self._lock = threading.Lock()

    def complete(self, request: CompletionRequest) -> CompletionResult:
        """Stream one message and return the final result. Partial spend on a mid-stream failure is
        estimated from the accumulated snapshot (``_partial_cost``) and attached to the ``ProviderError``;
        a timeout before the stream started is charged the worst case."""
        on = self._today()
        price = _price(request.model, on)
        params = self.build_params(request)
        client = self._get_client()
        stream: Any = None
        try:
            with client.messages.stream(**params) as stream:
                message = stream.get_final_message()
        except Exception as exc:
            partial = _partial_cost(stream, price, request.max_output_tokens)
            if partial is None:  # failed before message_start
                partial = transport_failure_cost(exc, self.worst_case_cost(request, on)) if _is_timeout(exc) else 0.0
            raise _wrap_error(exc, cost_usd=partial) from exc
        return self.to_result(message, request, on=on)

    def worst_case_cost(self, request: CompletionRequest, on: date | None = None) -> float:
        return token_worst_case(
            request.model, estimate_request_tokens(request), request.max_output_tokens, on=on or self._today()
        )

    @staticmethod
    def build_params(request: CompletionRequest) -> dict[str, Any]:
        """Pure mapping to ``messages.stream`` kwargs. Puts ``cache_control`` on the system
        prompt so repeated crosscheck/repair calls reuse the cached prefix.

        Raises ``ProviderError`` for a trailing assistant message: these models reject prefill."""
        if not request.messages or request.messages[-1].role != "user":
            raise ProviderError("Claude requests must end with a user message (no prefill)", provider=PROVIDER)
        output_config: dict[str, Any] = {"effort": request.effort or DEFAULT_EFFORT}
        if request.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        params: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_output_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "thinking": {"type": "adaptive"},
            "output_config": output_config,
        }
        if request.system:
            params["system"] = [{"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}]
        return params

    @staticmethod
    def to_result(message: Any, request: CompletionRequest, *, on: date | None = None) -> CompletionResult:
        """Pure mapping from an Anthropic ``Message`` (or fixture dict) to ``CompletionResult``.

        Raises ``ProviderRefusal`` on ``stop_reason == "refusal"`` and ``StructuredOutputError`` when a
        schema was requested but the output was truncated or invalid; both carry the billed cost."""
        if isinstance(message, dict):
            from anthropic.types import Message

            message = Message.model_validate(message)
        usage = usage_from_message(get_field(message, "usage"))
        cost = _price(request.model, on or date.today()).cost(usage)
        stop_reason = get_field(message, "stop_reason", "")
        text = "".join(
            get_field(block, "text", "")
            for block in get_field(message, "content", [])
            if get_field(block, "type") == "text"
        )

        if stop_reason == "refusal":
            raise ProviderRefusal(f"Claude refused{': ' + text if text else ''}", provider=PROVIDER, cost_usd=cost)
        parsed = None
        if request.json_schema is not None:
            if stop_reason in ("max_tokens", "model_context_window_exceeded"):
                raise StructuredOutputError(
                    f"Claude structured output truncated (stop_reason={stop_reason})", provider=PROVIDER, cost_usd=cost
                )
            parsed = parse_json_output(text, request.json_schema, provider=PROVIDER, cost_usd=cost)
        return CompletionResult(
            text=text,
            parsed=parsed,
            usage=usage,
            cost_usd=cost,
            model=request.model,
            provider=PROVIDER,
            stop_reason=stop_reason,
            response_id=get_field(message, "id", ""),
            raw=_dump(message),
        )

    def _get_client(self) -> Any:
        with self._lock:
            if self._client is None:
                key = env_api_key("ANTHROPIC_API_KEY")
                if key is None:
                    raise ProviderError("ANTHROPIC_API_KEY is not set", provider=PROVIDER)
                import anthropic

                self._client = anthropic.Anthropic(api_key=key, timeout=self._timeout_s, max_retries=self._max_retries)
            return self._client


def usage_from_message(usage: Any) -> Usage:
    """Anthropic reports uncached input separately from cache reads and writes; sum them."""
    uncached = get_field(usage, "input_tokens", 0)
    cache_read = get_field(usage, "cache_read_input_tokens", 0)
    cache_write = get_field(usage, "cache_creation_input_tokens", 0)
    return Usage(
        input_tokens=uncached + cache_read + cache_write,
        output_tokens=get_field(usage, "output_tokens", 0),
        cached_input_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def _partial_cost(stream: Any, price: ModelPrice, max_output_tokens: int) -> float | None:
    """Billed estimate for a stream that failed after ``message_start`` (None if it never started).

    The accumulator only learns the real ``output_tokens`` from ``message_delta`` at the end of the stream, and
    summarized thinking hides how many tokens were generated. So until ``stop_reason`` is known, the output is
    charged at ``max_output_tokens`` (input from ``message_start`` is exact)."""
    if stream is None:
        return None
    try:
        snapshot = stream.current_message_snapshot
    except (AssertionError, AttributeError):
        return None
    usage = usage_from_message(get_field(snapshot, "usage"))
    if not get_field(snapshot, "stop_reason"):
        usage = usage.model_copy(update={"output_tokens": max(usage.output_tokens, max_output_tokens)})
    return price.cost(usage)


def _is_timeout(exc: Exception) -> bool:
    import anthropic

    return isinstance(exc, anthropic.APITimeoutError)


def _price(model: str, on: date) -> ModelPrice:
    try:
        return price_for(model, on)
    except UnknownModelPrice as exc:
        raise ProviderError(f"refusing unpriced model: {exc}", provider=PROVIDER) from None


def _dump(message: Any) -> dict[str, Any]:
    dump = getattr(message, "model_dump", None)
    return dump(mode="json", exclude_none=True) if callable(dump) else {}


def _wrap_error(exc: Exception, *, cost_usd: float = 0.0) -> ProviderError:
    """Map any SDK exception to ``ProviderError``."""
    import anthropic

    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, anthropic.APIStatusError):
        # A status below 400 is an SSE ``error`` event mid-stream on a 200 response (overloaded_error, api_error...).
        return ProviderError(
            f"Anthropic HTTP {exc.status_code}: {exc.message}",
            provider=PROVIDER,
            cost_usd=cost_usd,
            retryable=exc.status_code < 400 or is_retryable_status(exc.status_code),
        )
    if isinstance(exc, anthropic.APIConnectionError):  # includes APITimeoutError
        return ProviderError(f"Anthropic connection error: {exc}", provider=PROVIDER, cost_usd=cost_usd, retryable=True)
    retryable = isinstance(exc, getattr(anthropic, "RetryableError", ()))
    return ProviderError(
        f"Anthropic call failed: {type(exc).__name__}: {exc}", provider=PROVIDER, cost_usd=cost_usd, retryable=retryable
    )
