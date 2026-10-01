"""Shared vocabulary used by every module. Frozen by the architect.

Changes here ripple across all owners, so they go through the lead.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AgentName = Literal["chatgpt", "gemini", "claude"]
"""Logical agents. Spend is reported per agent in run.md."""

ProviderName = Literal["openai", "gemini", "anthropic", "claude_code"]
"""Concrete transports. ``claude`` the agent uses both ``anthropic`` and ``claude_code``."""

StageName = Literal["ingestion", "strategy", "execution", "crosscheck", "final"]

STAGE_ORDER: tuple[StageName, ...] = ("ingestion", "strategy", "execution", "crosscheck", "final")

Tier = Literal["default", "max"]

ExecutionMode = Literal["code", "prose", "mixed"]
"""Chosen by ChatGPT triage in ingestion. ``code``/``mixed`` use Claude Code; ``prose`` uses Messages."""

Severity = Literal["critical", "major", "minor"]


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    COMPLETED = "completed"
    COMPLETED_WITH_ISSUES = "completed_with_issues"
    """Final ran, but the result is not verified. Either critical issues stayed unresolved after the cross-check loop
    cap or a loop skipped for budget (``unresolved_critical > 0``), or acceptance criteria are not met
    (``criteria_unmet > 0``), maf's own gates included: acceptance, clean-room reproduction, source audit and
    deliverable lint. Either is enough (``maf.vault.RunIndex.has_issues``)."""
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"

    @property
    def terminal(self) -> bool:
        return self in (
            RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_ISSUES, RunStatus.FAILED, RunStatus.BUDGET_EXCEEDED
        )

    @property
    def finished(self) -> bool:
        """The final stage ran (05-final exists), with or without unresolved critical issues."""
        return self in (RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_ISSUES)


class Usage(BaseModel):
    """Provider-normalized token usage for one call.

    Normalization rules (every provider adapter must follow them):

    - ``input_tokens`` is the TOTAL billed input, including cached reads and cache writes.
      (Anthropic reports these separately; the adapter sums them.)
    - ``cached_input_tokens`` and ``cache_write_tokens`` are subsets of ``input_tokens``.
    - ``output_tokens`` is the TOTAL billed output, including reasoning/thinking tokens.
      (Gemini reports ``thoughts_token_count`` separately; the adapter sums it.)
    - ``reasoning_tokens`` is informational, a subset of ``output_tokens``.
    - ``search_queries`` counts billable grounding/search queries (Gemini).
    """

    model_config = ConfigDict(frozen=True)

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    search_queries: int = Field(default=0, ge=0)
