"""Stage backend protocol, the context passed to each stage, and the shared generate-validate-repair loop.

Owner: stages.

A stage backend is pure with respect to run state. It reads prior notes through ``ctx``,
makes metered calls through ``ctx.call``, and returns a ``StageOutput``. **The pipeline**, not the
stage, writes the returned notes to the vault, updates run.md, and advances the state machine.
Exception: Claude Code writes files into the workspace, and stages may copy
assets/deliverables through ``ctx.vault`` (idempotent on re-run).

Stage code reaches ``maf.handoff``, ``maf.vault`` and ``maf.ledger`` functions through their modules
(``hf.build_handoff``, not a from-import), so there is exactly one binding to each contract function.

Claude Code work sessions (the execution pass and the fix pass) go through ``work_session``: a session the budget
cannot give its minimum is not started, one whose clamped budget runs out ends the run ``budget_exceeded``, and a
timed-out one gets exactly one continuation in the same workspace.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from maf import handoff as hf
from maf import ledger as _ledger
from maf import vault as _vault
from maf.config import ModelRole, Settings, agent_for_role
from maf.handoff import Handoff, HandoffInvalid, HandoffKind, HandoffMeta
from maf.ledger import Ledger
from maf.prompts import render_prompt
from maf.providers import CompletionRequest, CompletionResult, Message, ProviderError, Providers, StructuredOutputError
from maf.providers.claude_code import ClaudeCodeBudgetExhausted, ClaudeCodeTimeout, preflight_budget_usd
from maf.types import StageName
from maf.vault import RunIndex, RunPaths, Vault

log = logging.getLogger(__name__)

PYTHON_AUTHOR = "maf"
"""``from`` of notes assembled by Python rather than written by a model."""

NO_MODEL = "none"
"""``model`` of Python-assembled notes that made no call."""

MAX_TRANSIENT_RETRIES = 2
"""Extra attempts for ``ProviderError(retryable=True)`` on token-priced providers (never Claude Code)."""

RETRY_BACKOFF_S: tuple[float, ...] = (5.0, 20.0)

DEFAULT_EFFORT: dict[ModelRole, str | None] = {
    "chatgpt": "high",
    "gemini": None,
    "claude": "high",
    "claude_code": "high",
}
"""Claude models must get an explicit effort (Opus 5.5 would otherwise default to ``medium``)."""

WORKSPACE_META_DIR = ".maf"
"""Workspace subfolder holding prompt copies (traceability); never an artifact."""

CONTINUATION_SUFFIX = "continuation"
"""Ledger ``purpose`` suffix of the session that continues a timed-out work session (``execution-continuation``)."""

CONTINUED_KEY = "maf_continuation"
"""``CompletionResult.raw`` key ``work_session`` sets on a continuation's result: what the timed-out session was
charged and why it stopped (``continuation_note``)."""

CONTINUATION_PREAMBLE = """# Continuation: the previous session was cut off

The previous Claude Code session on this task was stopped because it ran out of time ({reason}), before it gave its \
final answer. Everything it wrote is still in this workspace. Continue that work; do not start over.

1. First inspect the workspace (the files, logs and results the previous session left) to see what is done and \
what is not.
2. Finish only the incomplete parts. Do not restart, redo or rewrite work that is complete and correct.
3. Run the reproduction command and the tests, and check their results, before you answer.
4. Keep every command under 10 minutes: split long simulations and test batteries into shorter runs, and reuse the \
valid results already on disk.
5. Then answer exactly as the original task below asks.

The original task follows, unchanged.

---

"""
"""Opens the prompt of the one continuation session after a timeout (``continuation_prompt``)."""

_sleep: Callable[[float], None] = time.sleep


@dataclass
class StageContext:
    """Everything a stage may use. Built fresh by the pipeline for each stage attempt."""

    index: RunIndex
    """Snapshot of run.md at stage start (``mode``, ``round``, ``brief``, ``input_files``...). Read-only by convention."""
    settings: Settings
    vault: Vault
    paths: RunPaths
    ledger: Ledger
    providers: Providers
    stage: StageName
    now: datetime
    review_note: str | None = None
    """User direction from ``maf resume --note`` after the review gate (strategy/execution consume it)."""

    @property
    def run_id(self) -> str:
        return self.index.run_id

    @property
    def round(self) -> int:
        return self.index.round

    def model(self, role: ModelRole) -> str:
        """Model ID for ``role`` in this stage (tier pin or per-stage override)."""
        return self.settings.model_for(role, self.stage)

    def call(self, role: ModelRole, request: CompletionRequest, *, purpose: str = "") -> CompletionResult:
        """Metered call: ``maf.ledger.metered_call(self.ledger, providers.for_role(role), request, stage=self.stage, purpose=...)``.

        Transient failures (``ProviderError.retryable``) of token-priced providers are retried up to
        ``MAX_TRANSIENT_RETRIES`` times with backoff; each attempt is metered separately. Claude Code is
        never retried here, because a failed run may already have changed the workspace.
        """
        provider = self.providers.for_role(role)
        attempts = 1 if role == "claude_code" else 1 + MAX_TRANSIENT_RETRIES
        for attempt in range(attempts):
            try:
                return _ledger.metered_call(self.ledger, provider, request, stage=self.stage, purpose=purpose)
            except ProviderError as exc:
                if not exc.retryable or attempt == attempts - 1:
                    raise
                delay = RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)]
                log.warning("%s %s call failed transiently (%s); retrying in %.0fs", self.stage, role, exc, delay)
                _sleep(delay)
        raise AssertionError("unreachable")

    def read(self, name: str) -> Handoff:
        """Read a prior note of this run by name (e.g. ``"01-ingestion"``)."""
        return self.vault.read_handoff(self.run_id, name)

    def prior_notes(self) -> dict[str, Handoff]:
        """All notes listed in ``index.handoffs``, keyed by name, for the consumption rules in ARCHITECTURE.md."""
        return {name: self.read(name) for name in dict.fromkeys(self.index.handoffs)}

    def latest_note_name(self, kind: HandoffKind) -> str:
        """Name of the newest round of ``kind`` recorded in ``index.handoffs`` (current round if none is recorded)."""
        recorded = set(self.index.handoffs)
        for rnd in range(self.round, 0, -1):
            name = _vault.note_name(kind, rnd)
            if name in recorded:
                return name
        return _vault.note_name(kind, self.round)


@dataclass(frozen=True)
class NoteOut:
    """One note to persist: ``name`` from ``maf.vault.note_name``."""

    name: str
    handoff: Handoff


@dataclass
class StageOutput:
    """What a stage returns. ``notes[-1]`` is the stage's primary handoff (e.g. ``03-execution``)."""

    notes: list[NoteOut]
    index_updates: dict[str, Any] = field(default_factory=dict)
    """Allowed keys (``maf.pipeline.ALLOWED_INDEX_UPDATES``): ``mode`` (ingestion), ``unresolved_critical``
    (crosscheck), ``criteria_unmet``, ``unmet_criteria``, ``exported_at``, ``export_note`` (final). Others are rejected by
    the pipeline."""
    loop_back: bool = False
    """Crosscheck only: True means go back to execution (the pipeline enforces ``max_crosscheck_loops``)."""
    deliverables: list[Path] = field(default_factory=list)
    """Workspace paths copied to ``deliverables/`` (already copied by the stage; listed for the index)."""


@runtime_checkable
class StageBackend(Protocol):
    """Pluggable backend for one stage. Must be idempotent: re-running after a crash overwrites its notes."""

    name: StageName

    def run_stage(self, ctx: StageContext) -> StageOutput:
        """Execute the stage. May raise ``BudgetExceeded``, ``HandoffInvalid`` (after the repair attempt),
        or ``ProviderError``; the pipeline maps each to a run status."""
        ...


@dataclass(frozen=True)
class Generated:
    """A validated handoff plus every provider result that produced it (first call, then the repair if any)."""

    handoff: Handoff
    results: tuple[CompletionResult, ...]

    @property
    def last(self) -> CompletionResult:
        return self.results[-1]


Check = Callable[[Handoff], list[str]]
"""Extra stage-specific validation run after ``build_handoff``; returned errors trigger the repair."""


def generate(
    ctx: StageContext,
    role: ModelRole,
    kind: HandoffKind,
    *,
    system: str,
    prompt: str,
    to: str,
    inputs: list[str],
    max_output_tokens: int | None = None,
    purpose: str = "",
    check: Check | None = None,
    session: bool = False,
    reserve_usd: float = 0.0,
    **request_kw: Any,
) -> Generated:
    """``generate_handoff`` that also returns the provider results (for citations, costs, raw payloads).

    ``session=True`` (Claude Code only) makes the first call a ``work_session`` (minimum budget, budget exhaustion
    under a clamp, one continuation after a timeout, ``reserve_usd`` held back for later stages as far as the session's
    minimum allows) and the repair one without a minimum, reserve or continuation."""
    spec = hf.format_spec(kind)
    if spec not in prompt:
        prompt = f"{prompt.rstrip()}\n\n{spec}\n"
    request_kw.setdefault("effort", DEFAULT_EFFORT[role])
    if role == "claude_code":
        request_kw.setdefault("max_budget_usd", ctx.settings.output_limits.claude_code_budget_usd)
    model = ctx.model(role)
    limit = max_output_tokens or default_output_tokens(ctx.settings, role)
    label = purpose or kind.value

    if session and role != "claude_code":
        raise ValueError(f"only Claude Code calls can be work sessions, not {role}")
    request = CompletionRequest.simple(model, prompt, system=system, max_output_tokens=limit, **request_kw)
    if session:
        first = work_session(ctx, request, purpose=label, reserve_usd=reserve_usd, soft_reserve=True)
    else:
        first = ctx.call(role, request, purpose=label)
    results: list[CompletionResult] = [first]
    try:
        return Generated(_build_checked(ctx, role, kind, first, results, to, inputs, check), tuple(results))
    except HandoffInvalid as exc:
        log.warning("%s handoff from %s invalid, repairing: %s", kind, role, exc)
        errors = exc.errors

    # The repair only fixes the format: web search, fetched pages and attachments were used already, so drop them.
    repair_kw = {
        k: v for k, v in request_kw.items() if k not in ("web_search", "url_context", "attachments", "max_search_queries")
    }
    repair_request = CompletionRequest.simple(
        model, hf.repair_prompt(kind, first.text, errors), system=system, max_output_tokens=limit, **repair_kw
    )
    if session:
        repair = work_session(ctx, repair_request, purpose="repair", minimum_usd=0.0, continue_on_timeout=False)
    else:
        repair = ctx.call(role, repair_request, purpose="repair")
    results.append(repair)
    return Generated(_build_checked(ctx, role, kind, repair, results, to, inputs, check), tuple(results))


def generate_handoff(
    ctx: StageContext,
    role: ModelRole,
    kind: HandoffKind,
    *,
    system: str,
    prompt: str,
    to: str,
    inputs: list[str],
    max_output_tokens: int | None = None,
    purpose: str = "",
    **request_kw: Any,
) -> Handoff:
    """The strict handoff loop shared by all stages.

    1. Call ``role`` with ``prompt`` plus ``maf.handoff.format_spec(kind)``.
    2. ``maf.handoff.build_handoff(result.text, meta)``; meta is filled by Python
       (``from`` = agent of ``role``, ``model``, ``cost_usd`` = the sum of all calls here, ``round`` = ``ctx.index.round``).
    3. On ``HandoffInvalid``: exactly **one** repair call with ``maf.handoff.repair_prompt``
       (same role and model, ``purpose="repair"``). If that also fails, re-raise ``HandoffInvalid``,
       which stops the run.

    For ``role == "claude_code"`` the repair call is a Claude Code call too, in the same workspace.
    A ``check=`` keyword adds stage-specific validation whose errors also go through the repair.
    """
    return generate(
        ctx,
        role,
        kind,
        system=system,
        prompt=prompt,
        to=to,
        inputs=inputs,
        max_output_tokens=max_output_tokens,
        purpose=purpose,
        **request_kw,
    ).handoff


def _build_checked(
    ctx: StageContext,
    role: ModelRole,
    kind: HandoffKind,
    result: CompletionResult,
    results: list[CompletionResult],
    to: str,
    inputs: list[str],
    check: Check | None,
) -> Handoff:
    meta = make_meta(
        ctx,
        kind,
        from_=agent_for_role(role),
        to=to,
        inputs=inputs,
        model=result.model or ctx.model(role),
        cost_usd=sum(r.cost_usd for r in results),
    )
    handoff = hf.build_handoff(result.text, meta)
    if check is not None:
        errors = check(handoff)
        if errors:
            raise HandoffInvalid(kind, errors)
    return handoff


def work_session(
    ctx: StageContext,
    request: CompletionRequest,
    *,
    purpose: str,
    minimum_usd: float | None = None,
    continue_on_timeout: bool = True,
    reserve_usd: float = 0.0,
    soft_reserve: bool = False,
) -> CompletionResult:
    """A Claude Code work session (execution, fix pass): ``ctx.call("claude_code", ...)`` plus four rules.

    - Minimum budget: when the ``--max-budget-usd`` ``metered_call`` would pass (``maf.ledger.planned_budget``: the
      session's budget clamped to what the run has left minus one turn of headroom) is below the session's floor
      (``session_floor``: ``minimum_usd``, default ``Settings.claude_code_min_session_usd`` but at most
      ``claude_code_min_session_share`` of the run's cap; never more than the session's own budget),
      ``SessionBudgetTooSmall`` is raised before anything is spawned. Its hint (``needed_cap_usd``) covers the
      preflight a resumed process pays first (``preflight_reserve``) and grows with the share of the cap.
    - Reserve: ``reserve_usd`` is held back for the stages after this one, so the session's ``--max-budget-usd``
      leaves it unspent. When the session could then not get its floor it is refused (``SessionBudgetTooSmall`` with
      ``reserve_usd``; the cross-check's fix pass skips on it), unless ``soft_reserve``: then the session gets its
      floor out of the reserve (an execution pass, without whose note nothing later can run).
    - Budget exhaustion: a session that stops at its cap (``ClaudeCodeBudgetExhausted``) after that clamp or the reserve
      cut its budget raises ``SessionBudgetExhausted``; both are ``BudgetExceeded``, so the run ends
      ``budget_exceeded`` and ``maf resume --budget`` continues it. A session that used up its own, uncut budget still
      fails the run.
    - Timeout: a ``ClaudeCodeTimeout`` (the ledger charged its worst case) is followed by exactly one continuation
      session in the same workspace (``continuation_prompt``, ledger purpose ``<purpose>-continuation``, the same
      budget rules), unless ``continue_on_timeout`` is False. Its prompt is copied to
      ``.maf/<purpose>-r<round>-continuation.md``. A second timeout raises ``ClaudeCodeTimeout`` ("timed out twice"),
      which fails the run. A continuation refused or out of budget carries the timed-out session's charge in the
      error's ``charged_usd``.

    Returns the last session's result. After a continuation its ``cost_usd`` includes the timed-out session's
    charge, so the note's cost still adds up, and ``raw[CONTINUED_KEY]`` records it (``continuation_note``)."""

    def session(req: CompletionRequest, label: str) -> CompletionResult:
        return _metered_session(ctx, req, label, minimum_usd, reserve_usd, soft_reserve)

    try:
        return session(request, purpose)
    except ClaudeCodeTimeout as exc:
        if not continue_on_timeout:
            raise
        stopped = exc
    log.warning("%s: Claude Code %s session timed out (%s); running one continuation", ctx.stage, purpose, stopped)
    prompt = continuation_prompt(request.messages[-1].content, str(stopped))
    write_workspace_file(ctx, f"{WORKSPACE_META_DIR}/{purpose}-r{ctx.round}-{CONTINUATION_SUFFIX}.md", prompt)
    continued = request.model_copy(update={"messages": (*request.messages[:-1], Message(role="user", content=prompt))})
    charged = stopped.cost_usd
    try:
        result = session(continued, f"{purpose}-{CONTINUATION_SUFFIX}")
    except ClaudeCodeTimeout as exc:
        raise ClaudeCodeTimeout(
            f"the Claude Code {purpose} session and its one continuation both timed out ({stopped}; then {exc}); "
            f"their work stays in the workspace, and maf resume runs {ctx.stage} again",
            provider=exc.provider,
        ) from exc
    except StructuredOutputError as exc:  # billed; the caller may tolerate it, and its note carries both sessions
        raise StructuredOutputError(str(exc), provider=exc.provider, cost_usd=exc.cost_usd + charged) from exc
    except (_ledger.SessionBudgetTooSmall, _ledger.SessionBudgetExhausted) as exc:
        exc.charged_usd += charged  # a caller that carries on (the fix pass) puts both sessions in its note
        raise
    raw = {**result.raw, CONTINUED_KEY: {"stopped": str(stopped), "charged_usd": charged}}
    return result.model_copy(update={"cost_usd": result.cost_usd + charged, "raw": raw})


def ensure_session_budget(ctx: StageContext, purpose: str) -> None:
    """Refuse a work session with the default budget (``OutputLimits.claude_code_budget_usd``) now, before a sandbox
    preflight or anything else of the stage is paid for, when it could not get its floor (``SessionBudgetTooSmall``).
    While the sandbox is not verified yet, what the preflight may spend (``preflight_reserve``) is left out of what is
    available, so a stage that passes here still has the floor after the preflight. ``work_session`` checks again
    right before spawning."""
    budget = ctx.settings.output_limits.claude_code_budget_usd
    probe = CompletionRequest.simple(ctx.model("claude_code"), "", max_output_tokens=1, max_budget_usd=budget)
    verified = getattr(ctx.providers.for_role("claude_code"), "sandbox_verified", False)
    _session_budget(ctx, probe, purpose, None, pay_first_usd=0.0 if verified else preflight_reserve(ctx))


def preflight_reserve(ctx: StageContext) -> float:
    """What the Claude Code sandbox preflight may spend (its ``--max-budget-usd``: ``claude_code_preflight_budget_usd``,
    or ``preflight_budget_usd`` of the model), or 0 when the provider has no ``preflight`` (test fakes). Every process
    that advances a code/mixed run pays one before its first Claude Code call (``ensure_sandbox``)."""
    if getattr(ctx.providers.for_role("claude_code"), "preflight", None) is None:
        return 0.0
    budget = ctx.settings.claude_code_preflight_budget_usd
    return preflight_budget_usd(ctx.model("claude_code")) if budget is None else budget


def session_floor(
    settings: Settings, cap_usd: float, session_usd: float | None, minimum_usd: float | None = None
) -> float:
    """Pure: the smallest ``--max-budget-usd`` a work session is started with. ``minimum_usd`` when given; by default
    ``claude_code_min_session_usd``, but at most ``claude_code_min_session_share`` of the run's cap ``cap_usd``, so a
    small run (the $5 MCP default) still gets the session its budget was meant for. Never more than the session's own
    budget ``session_usd``."""
    fixed, share = _floor_terms(settings, session_usd, minimum_usd)
    return fixed if share is None else min(fixed, share * cap_usd)


def needed_cap(committed_usd: float, fixed_usd: float, share: float | None) -> float:
    """Pure: the smallest run cap ``c`` with ``c - committed_usd >= min(fixed_usd, share * c)`` (just ``fixed_usd`` when
    ``share`` is None): the cap at which a session whose floor is ``session_floor`` gets it, when ``committed_usd``
    (the spend so far, what must be paid first, a reserve and one turn of headroom) comes off the cap first."""
    if share is not None and share < 1.0:
        cap = committed_usd / (1.0 - share)
        if share * cap <= fixed_usd:
            return cap
    return committed_usd + fixed_usd


def _floor_terms(
    settings: Settings, session_usd: float | None, minimum_usd: float | None
) -> tuple[float, float | None]:
    """``(fixed part, share of the cap or None)`` of ``session_floor``."""
    fixed, share = (
        (settings.claude_code_min_session_usd, settings.claude_code_min_session_share)
        if minimum_usd is None
        else (minimum_usd, None)
    )
    return (fixed if session_usd is None else min(fixed, session_usd)), share


def _session_budget(
    ctx: StageContext,
    request: CompletionRequest,
    purpose: str,
    minimum_usd: float | None,
    reserve_usd: float = 0.0,
    soft_reserve: bool = False,
    *,
    pay_first_usd: float = 0.0,
) -> tuple[float, str]:
    """``(budget, what)``: the ``--max-budget-usd`` the session gets now (see ``work_session`` for the reserve) and its
    label, after raising ``SessionBudgetTooSmall`` when that is below its floor. ``pay_first_usd`` is spend due before
    the session (the preflight ``ensure_session_budget`` accounts for)."""
    provider = ctx.providers.for_role("claude_code")
    ledger = ctx.ledger
    floor = session_floor(ctx.settings, ledger.cap_usd, request.max_budget_usd, minimum_usd)
    budget, headroom = _ledger.planned_budget(ledger, provider, request, reserve_usd=pay_first_usd)
    if reserve_usd > 0:
        kept, _ = _ledger.planned_budget(ledger, provider, request, reserve_usd=pay_first_usd + reserve_usd)
        budget = min(budget, max(kept, floor)) if soft_reserve else kept
    what = _ledger.call_label(provider, request, stage=ctx.stage, purpose=purpose)
    if budget < floor - _ledger.BUDGET_EPSILON_USD:
        held = 0.0 if soft_reserve else reserve_usd
        before = preflight_reserve(ctx)  # a resumed process verifies the sandbox again before this session
        fixed, share = _floor_terms(ctx.settings, request.max_budget_usd, minimum_usd)
        raise _ledger.SessionBudgetTooSmall(
            cap_usd=ledger.cap_usd,
            spent_usd=ledger.spent_usd,
            what=what,
            budget_usd=budget,
            minimum_usd=floor,
            headroom_usd=headroom,
            reserve_usd=held,
            before_usd=before,
            needed_cap_usd=needed_cap(ledger.spent_usd + before + held + headroom, fixed, share),
        )
    return budget, what


def _metered_session(
    ctx: StageContext,
    request: CompletionRequest,
    purpose: str,
    minimum_usd: float | None,
    reserve_usd: float = 0.0,
    soft_reserve: bool = False,
) -> CompletionResult:
    """One metered work session under ``work_session``'s budget rules (no continuation)."""
    budget, what = _session_budget(ctx, request, purpose, minimum_usd, reserve_usd, soft_reserve)
    requested = request.max_budget_usd
    cut = requested is None or budget < requested - _ledger.BUDGET_EPSILON_USD
    if reserve_usd > 0 and cut:
        request = request.model_copy(update={"max_budget_usd": budget})  # metered_call alone would spend the reserve
    try:
        return ctx.call("claude_code", request, purpose=purpose)
    except ClaudeCodeBudgetExhausted as exc:
        if not cut:
            raise  # it ran out of its own budget, not the run's: a failure
        raise _ledger.SessionBudgetExhausted(
            cap_usd=ctx.ledger.cap_usd,
            spent_usd=ctx.ledger.spent_usd,
            what=what,
            budget_usd=budget,
            requested_usd=requested if requested is not None else budget,
            charged_usd=exc.cost_usd,
        ) from exc


def continuation_prompt(prompt: str, reason: str) -> str:
    """The continuation session's prompt: ``CONTINUATION_PREAMBLE`` (inspect the workspace, finish what is incomplete,
    rerun the reproduction and tests, commands under 10 minutes), then the original ``prompt`` unchanged."""
    return CONTINUATION_PREAMBLE.format(reason=one_line(reason, 200)) + prompt


def continuation_note(result: CompletionResult) -> str:
    """One line for a note whose Claude Code pass needed a continuation (``work_session``); empty otherwise."""
    info = result.raw.get(CONTINUED_KEY)
    if not isinstance(info, dict):
        return ""
    return (
        f"maf: the first Claude Code session of this pass was stopped ({one_line(str(info.get('stopped', '')), 200)}) "
        f"and charged its worst case (${float(info.get('charged_usd', 0.0)):.2f}); one continuation session in the "
        "same workspace finished the pass."
    )


def export_excludes(settings: Settings) -> tuple[str, ...]:
    """Exclude patterns of the deliverable tree, in ``maf.lint.excluded`` order: ``maf.vault.DEFAULT_EXPORT_EXCLUDES``
    and ``Settings.export_exclude``, then ``Settings.export_include`` as ``!`` re-includes, then
    ``maf.vault.PROTECTED_EXPORT_EXCLUDES`` again, so no re-include reaches pipeline state or the user's inputs. It is
    what a code/mixed export leaves out of ``deliverables/``, and so what the execution and cross-check lint and the
    source audit skip. Lint and audit see exactly what ships: a link into ``inputs/`` or ``FreeRTOS-Kernel/`` does not
    resolve in the export, so it is broken."""
    excludes = dict.fromkeys((*_vault.DEFAULT_EXPORT_EXCLUDES, *settings.export_exclude))
    if not settings.export_include:
        return tuple(excludes)
    return (*excludes, *(f"!{p}" for p in settings.export_include), *_vault.PROTECTED_EXPORT_EXCLUDES)


def messages_worst_case(
    ctx: StageContext, prompt: str, role: ModelRole = "claude", stage: StageName | None = None
) -> float:
    """Worst-case cost of a ``generate_handoff`` call to ``role`` with ``prompt`` (its system prompt, output limit and
    default effort, the model ``stage`` pins, default ``ctx.stage``), priced by the provider without a call."""
    request = CompletionRequest.simple(
        ctx.settings.model_for(role, stage or ctx.stage),
        prompt,
        system=role_system(role, ctx.settings),
        max_output_tokens=default_output_tokens(ctx.settings, role),
        effort=DEFAULT_EFFORT[role],
    )
    return ctx.providers.for_role(role).worst_case_cost(request)


# ---------------------------------------------------------------------- round budget


FINAL_PROMPT_ALLOWANCE_CHARS = 40_000
"""Final report prompt text besides the notes a stand-in prompt carries (instructions, deliverable list, maf's checks,
the format spec and, before the cross-check has written it, its note): added to a stand-in for final's reserve."""


def final_reserve_usd(ctx: StageContext, mode: str | None, prompt: str) -> float:
    """What final needs to run in full: the worst case of the final report's call on ``prompt`` (a stand-in for its
    prompt) plus, for code/mixed runs, the clean room's ``cleanroom_budget_usd`` and its turn headroom."""
    reserve = messages_worst_case(ctx, prompt, "claude", "final")
    if mode != "prose":
        room = CompletionRequest.simple(
            ctx.settings.model_for("claude_code", "final"),
            "",
            max_output_tokens=1,
            max_budget_usd=ctx.settings.cleanroom_budget_usd,
        )
        headroom = _ledger.turn_headroom(ctx.providers.for_role("claude_code"), room)
        reserve += ctx.settings.cleanroom_budget_usd + headroom
    return reserve


def smallest_session_usd(ctx: StageContext, stage: StageName) -> float:
    """The most the smallest work session maf starts in ``stage`` can cost: the floor (``session_floor``) of a session
    with the default budget, plus one turn of headroom."""
    budget = ctx.settings.output_limits.claude_code_budget_usd
    probe = CompletionRequest.simple(
        ctx.settings.model_for("claude_code", stage), "", max_output_tokens=1, max_budget_usd=budget
    )
    floor = session_floor(ctx.settings, ctx.ledger.cap_usd, budget)
    return floor + _ledger.turn_headroom(ctx.providers.for_role("claude_code"), probe)


def round_entries(entries: Sequence[_ledger.LedgerEntry]) -> list[list[_ledger.LedgerEntry]]:
    """Pure: the ledger entries of each execution + cross-check round in ``entries`` (ledger order), oldest first. A
    round starts at the first execution entry after a cross-check one (or at the first execution entry); cross-check
    entries belong to the round they follow; ingestion, strategy and final entries to none. A failed and resumed stage
    stays in its round, and a timed-out session counts at its worst-case charge."""
    rounds: list[list[_ledger.LedgerEntry]] = []
    last: str | None = None
    for entry in entries:
        if entry.stage not in ("execution", "crosscheck"):
            continue
        if not rounds or (entry.stage == "execution" and last != "execution"):
            rounds.append([])
        rounds[-1].append(entry)
        last = entry.stage
    return rounds


def crosscheck_overhead(entries: Sequence[_ledger.LedgerEntry]) -> float:
    """Pure: the worst cases of the latest cross-check's calls other than Claude Code (its source audits, critiques,
    rebuttal, adjudication and their repairs; not the fix pass or a preflight), from the last round in ``entries`` that
    has any: what another cross-check needs available besides its fix pass. A failed and resumed cross-check counts
    both attempts, which errs on the safe side."""
    for entries_of_round in reversed(round_entries(entries)):
        calls = [e for e in entries_of_round if e.stage == "crosscheck" and e.provider != "claude_code"]
        if calls:
            return sum(e.worst_case_usd for e in calls)
    return 0.0


def round_reserve_usd(ctx: StageContext, mode: str | None, final_prompt: str) -> float:
    """What the execution pass of a looped round (round > 1, the extra round included) holds back for the rest of the
    run: the previous cross-check's calls at their worst case (``crosscheck_overhead``), the smallest fix session
    (``smallest_session_usd``) and final's reserve on ``final_prompt`` plus ``FINAL_PROMPT_ALLOWANCE_CHARS``
    (``final_reserve_usd``): what the cross-check's loop estimate counted beyond the execution session."""
    return (
        crosscheck_overhead(ctx.ledger.entries)
        + smallest_session_usd(ctx, "crosscheck")
        + final_reserve_usd(ctx, mode, final_prompt + "x" * FINAL_PROMPT_ALLOWANCE_CHARS)
    )


def default_output_tokens(settings: Settings, role: ModelRole) -> int:
    """``OutputLimits`` for ``role``. Claude Code ignores the value (its bound is ``--max-budget-usd``)."""
    limits = settings.output_limits
    return {"chatgpt": limits.chatgpt, "gemini": limits.gemini}.get(role, limits.claude)


def make_meta(
    ctx: StageContext,
    kind: HandoffKind,
    *,
    from_: str,
    to: str,
    inputs: list[str],
    model: str,
    cost_usd: float = 0.0,
) -> HandoffMeta:
    """Frontmatter for a note of this run, stamped with ``ctx.now`` and ``ctx.index.round``."""
    return HandoffMeta(
        run_id=ctx.run_id,
        stage=kind,
        from_=from_,
        to=to,
        inputs=[as_wikilink(name) for name in inputs],
        created=ctx.now,
        model=model,
        cost_usd=max(0.0, cost_usd),
        round=ctx.round,
        tags=["maf", f"maf/{kind.value}"],
    )


def assemble_handoff(
    ctx: StageContext,
    kind: HandoffKind,
    sections: dict[str, str],
    *,
    to: str,
    inputs: list[str],
    from_: str = PYTHON_AUTHOR,
    model: str = NO_MODEL,
    cost_usd: float = 0.0,
) -> Handoff:
    """Build (and validate) a note whose body Python wrote. Raises ``HandoffInvalid`` on a bug."""
    meta = make_meta(ctx, kind, from_=from_, to=to, inputs=inputs, model=model, cost_usd=cost_usd)
    return hf.build_handoff(hf.render_body(sections), meta)


def with_sections(handoff: Handoff, updates: dict[str, str]) -> Handoff:
    """Copy of ``handoff`` with some section bodies replaced (order kept), re-validated."""
    sections = dict(handoff.sections)
    sections.update({k: v.strip() for k, v in updates.items()})
    updated = handoff.model_copy(update={"sections": sections})
    errors = hf.validate_handoff(updated)
    if errors:
        raise HandoffInvalid(handoff.meta.stage, errors)
    return updated


def as_wikilink(name: str) -> str:
    return name if name.startswith("[[") else f"[[{name}]]"


def render_inputs(notes: dict[str, Handoff], sections: dict[str, tuple[str, ...]] | None = None) -> str:
    """Render prior notes for inclusion in a prompt: each note as ``<note name="...">...</note>`` with
    only the listed sections (all if ``sections`` is None or omits that note). ``<note``/``</note`` inside a
    body is escaped so note content cannot close its block and pose as prompt text. Web-derived content is
    additionally quoted inside the notes (``maf.stages.ingestion.quote_ingestion``)."""
    blocks: list[str] = []
    for name, note in notes.items():
        wanted = (sections or {}).get(name)
        chosen = note.sections if wanted is None else {t: note.sections[t] for t in wanted if t in note.sections}
        body = hf.render_body(chosen).strip() if chosen else "(no sections)"
        blocks.append(f'<note name="{name}">\n{escape_note_tags(body)}\n</note>')
    return "\n\n".join(blocks)


_NOTE_TAG = re.compile(r"<(/?)(note)\b", re.IGNORECASE)


def escape_note_tags(text: str) -> str:
    """Neutralize ``<note ...>`` and ``</note>`` so embedded text cannot break out of a ``render_inputs`` block."""
    return _NOTE_TAG.sub(lambda m: f"&lt;{m.group(1)}{m.group(2)}", text)


def role_system(role: ModelRole, settings: Settings) -> str:
    """System prompt for ``role`` from ``prompts/roles/<role>.md``."""
    if role == "claude_code":
        return render_prompt("roles/claude_code", python=str(settings.python_executable)).strip()
    return render_prompt(f"roles/{role}").strip()


def review_block(note: str | None) -> str:
    """Prompt block for the user's review direction (empty when there is none)."""
    if not note or not note.strip():
        return ""
    return (
        "## Direction from the user (review gate)\n\n"
        "The user reviewed the strategy and asked for the following. It overrides conflicting earlier guidance.\n\n"
        f"{note.strip()}"
    )


_HEADING_LINE = re.compile(r"^(\s{0,3})(#{1,6})(\s)", re.MULTILINE)


def neutralize_headings(text: str) -> str:
    """Escape Markdown headings so model/user text cannot open a new section in a Python-assembled note."""
    return _HEADING_LINE.sub(lambda m: f"{m.group(1)}\\{m.group(2)}{m.group(3)}", text)


def one_line(text: str, limit: int = 500) -> str:
    """Collapse whitespace to a single line and cap its length."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3].rstrip() + "..."


def write_workspace_file(ctx: StageContext, rel: str, text: str) -> Path:
    """Atomically write ``rel`` under the workspace (temp file + ``os.replace``). Returns the path."""
    target = ctx.paths.workspace / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target
