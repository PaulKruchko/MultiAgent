"""Shared test fixtures. Frozen by the architect: owners add fixtures in their own test modules.

No test may make a live paid API call. ``_no_network_keys`` removes provider keys from the
environment, so an accidental real client fails fast instead of spending money.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from maf.config import Settings
from maf.providers import CompletionRequest, CompletionResult, Providers
from maf.providers.base import parse_json_output
from maf.types import AgentName, ProviderName, Usage

FIXTURES = Path(__file__).parent / "fixtures"

Reply = str | dict[str, Any] | Exception | Callable[[CompletionRequest], "str | dict[str, Any]"]
"""A scripted reply: text, a parsed-JSON dict (text is its JSON dump), an exception to raise, or a
callable computing either from the request."""


@dataclass
class FakeProvider:
    """Scripted ``maf.providers.base.Provider``. Replies are consumed in FIFO order; ``default`` is used
    when the queue is empty (None means raise ``AssertionError``, so missing script entries are loud)."""

    name: ProviderName
    agent: AgentName
    replies: deque[Reply] = field(default_factory=deque)
    default: Reply | None = None
    cost_per_call: float = 0.01
    worst_case_usd: float = 0.05
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=1000, output_tokens=500))
    calls: list[CompletionRequest] = field(default_factory=list)

    def script(self, *replies: Reply) -> FakeProvider:
        self.replies.extend(replies)
        return self

    def complete(self, request: CompletionRequest) -> CompletionResult:
        self.calls.append(request)
        if self.replies:
            reply = self.replies.popleft()
        elif self.default is not None:
            reply = self.default
        else:
            raise AssertionError(f"FakeProvider[{self.name}] has no scripted reply for call #{len(self.calls)}")
        if callable(reply) and not isinstance(reply, Exception):
            reply = reply(request)
        if isinstance(reply, Exception):
            raise reply
        text = json.dumps(reply) if isinstance(reply, dict) else reply
        parsed = reply if isinstance(reply, dict) else None
        if request.json_schema is not None:
            # Same contract as the real adapters: invalid structured output raises (with the billed cost).
            parsed = parse_json_output(text, request.json_schema, provider=self.name, cost_usd=self.cost_per_call)
        return CompletionResult(
            text=text,
            parsed=parsed,
            usage=self.usage,
            cost_usd=self.cost_per_call,
            model=request.model,
            provider=self.name,
            stop_reason="end_turn",
        )

    def worst_case_cost(self, request: CompletionRequest, on: date | None = None) -> float:
        if self.name == "claude_code" and request.max_budget_usd is not None:
            return min(self.worst_case_usd, request.max_budget_usd)
        return self.worst_case_usd


@dataclass
class FakeProviders:
    """Holds the four fakes; ``.as_providers()`` gives the real ``Providers`` container."""

    chatgpt: FakeProvider
    gemini: FakeProvider
    claude: FakeProvider
    claude_code: FakeProvider

    def as_providers(self) -> Providers:
        return Providers(chatgpt=self.chatgpt, gemini=self.gemini, claude=self.claude, claude_code=self.claude_code)

    def factory(self) -> Callable[[Settings, Path], Providers]:
        """A ``pipeline.ProvidersFactory`` returning these fakes regardless of workspace."""
        return lambda _settings, _workspace: self.as_providers()

    def all(self) -> Iterable[FakeProvider]:
        return (self.chatgpt, self.gemini, self.claude, self.claude_code)


@pytest.fixture(autouse=True)
def _no_network_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for key in ("MAF_CONFIG", "MAF_VAULT", "MAF_WORKSPACES", "MAF_BUDGET_USD"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointing at a temp vault and workspaces dir."""
    return Settings(
        vault_path=tmp_path / "vault",
        workspaces_path=tmp_path / "workspaces",
        claude_executable=Path("/nonexistent/claude"),
        mcp_inbox=tmp_path / "inbox",
        freertos_path=None,
    )


@pytest.fixture
def fake_providers() -> FakeProviders:
    return FakeProviders(
        chatgpt=FakeProvider(name="openai", agent="chatgpt"),
        gemini=FakeProvider(name="gemini", agent="gemini"),
        claude=FakeProvider(name="anthropic", agent="claude"),
        claude_code=FakeProvider(name="claude_code", agent="claude", cost_per_call=0.50, worst_case_usd=2.0),
    )


@pytest.fixture
def fixed_now() -> datetime:
    return datetime(2026, 9, 28, 12, 0, 0)


def sample_body(kind: str) -> str:
    """A valid agent-written body for a handoff kind (``tests/fixtures/handoffs/<kind>.md``)."""
    return (FIXTURES / "handoffs" / f"{kind}.md").read_text(encoding="utf-8")


@pytest.fixture
def sample_bodies() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in (FIXTURES / "handoffs").glob("*.md")}


def load_provider_fixture(name: str) -> dict[str, Any]:
    """Recorded provider payload ``tests/fixtures/providers/<name>.json`` (owned by the providers owner)."""
    return json.loads((FIXTURES / "providers" / f"{name}.json").read_text(encoding="utf-8"))
