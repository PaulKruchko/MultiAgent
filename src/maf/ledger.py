"""Per-run cost ledger with a pre-call worst-case budget check.

Owner: core (config + ledger).

Semantics:

- One ``Ledger`` per run, persisted as append-only JSON Lines at
  ``runs/<run_id>/ledger.jsonl``. Reloading the file restores ``spent_usd`` exactly (resume).
- **Before** every provider call, ``metered_call`` asks the provider for its worst-case
  cost and calls ``Ledger.check``. If ``spent + worst_case > cap`` it raises
  ``BudgetExceeded`` and the call is never made.
- **After** the call, the actual cost (from ``CompletionResult.cost_usd``) is recorded.
  Actual cost can exceed the estimate only when the estimate was wrong. That overrun is
  recorded, not refused, and the next check sees it.
- Claude Code calls are bounded by ``--max-budget-usd`` plus one turn (the CLI checks its cap between
  turns). ``metered_call`` clamps the flag to ``remaining_usd`` minus that per-turn headroom, and the
  provider's worst case is the flag plus the headroom.
- Thread-safe: crosscheck critiques run concurrently against one ledger. While a metered call
  is in flight its worst case stays reserved, so concurrent checks cannot jointly overshoot the cap.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, get_args

from pydantic import BaseModel, Field, ValidationError

from maf.types import STAGE_ORDER, AgentName, ProviderName, StageName, Usage

if TYPE_CHECKING:
    from maf.providers.base import CompletionRequest, CompletionResult, Provider

CHARS_PER_TOKEN_WORST_CASE = 3.0
"""Conservative chars/token ratio for local input estimates (real ratio is ~4 for English)."""

BUDGET_EPSILON_USD = 1e-9
"""Float tolerance for the cap comparison, so ``spent + (cap - spent)`` never trips it by rounding."""

_AGENTS: tuple[AgentName, ...] = get_args(AgentName)
_PROVIDERS: tuple[ProviderName, ...] = get_args(ProviderName)

log = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """Raised before a call whose worst case would push spend past the cap."""

    def __init__(self, *, cap_usd: float, spent_usd: float, requested_usd: float, what: str) -> None:
        self.cap_usd = cap_usd
        self.spent_usd = spent_usd
        self.requested_usd = requested_usd
        self.what = what
        super().__init__(
            f"budget cap ${cap_usd:.2f} reached: spent ${spent_usd:.4f}, "
            f"next call ({what}) could cost up to ${requested_usd:.4f}"
        )


class LedgerEntry(BaseModel):
    """One recorded provider call. Serialized as one JSON line."""

    ts: datetime
    run_id: str
    stage: StageName
    agent: AgentName
    provider: ProviderName
    model: str
    usage: Usage = Field(default_factory=Usage)
    cost_usd: float = Field(ge=0)
    worst_case_usd: float = Field(ge=0)
    purpose: str = ""
    """Short label, e.g. ``"triage"``, ``"critique"``, ``"repair"``."""
    error: str = ""
    """Non-empty when the call failed but still spent money (partial spend)."""


class Ledger:
    """Budget-aware, append-only spend record for one run."""

    def __init__(self, run_id: str, cap_usd: float, path: Path | None = None) -> None:
        """Create an empty ledger. ``path=None`` keeps it in memory (tests)."""
        if not math.isfinite(cap_usd) or cap_usd < 0:
            raise ValueError(f"cap_usd must be a finite non-negative number, got {cap_usd!r}")
        self.run_id = run_id
        self.cap_usd = cap_usd
        self.path = path
        self._entries: list[LedgerEntry] = []
        self._lock = threading.Lock()
        self._reserved: dict[int, float] = {}
        self._next_reservation = 0

    @classmethod
    def load(cls, run_id: str, cap_usd: float, path: Path) -> Ledger:
        """Rebuild from ``path`` if it exists, else return an empty ledger bound to it.

        A torn final line (a crash mid-append) is truncated away with a warning so later appends
        start on a clean line. Corruption anywhere else raises ``ValueError``.
        """
        ledger = cls(run_id, cap_usd, path)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return ledger
        lines = raw.split(b"\n")
        torn = lines.pop()  # b"" when the file ends with a newline
        for lineno, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                ledger._entries.append(LedgerEntry.model_validate_json(line))
            except ValidationError as exc:
                raise ValueError(f"corrupt ledger {path} at line {lineno}: {exc}") from exc
        if torn.strip():
            try:
                ledger._entries.append(LedgerEntry.model_validate_json(torn))
            except ValidationError:
                log.warning("ledger %s: dropping torn final line (%d bytes)", path, len(torn))
                with path.open("r+b") as fh:
                    fh.truncate(len(raw) - len(torn))
                    fh.flush()
                    os.fsync(fh.fileno())
            else:
                with path.open("ab") as fh:  # complete record, only the newline is missing
                    fh.write(b"\n")
                    fh.flush()
                    os.fsync(fh.fileno())
        for entry in ledger._entries:
            if entry.run_id != run_id:
                raise ValueError(f"ledger {path} holds an entry for run {entry.run_id!r}, expected {run_id!r}")
        return ledger

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent()

    @property
    def remaining_usd(self) -> float:
        """``max(0, cap - spent)``."""
        with self._lock:
            return max(0.0, self.cap_usd - self._spent())

    def by_agent(self) -> dict[AgentName, float]:
        """Spend per agent (all three keys present, 0.0 if unused). Feeds run.md's cost table."""
        totals: dict[AgentName, float] = {agent: 0.0 for agent in _AGENTS}
        for entry in self.entries:
            totals[entry.agent] += entry.cost_usd
        return totals

    def by_provider(self) -> dict[ProviderName, float]:
        """Spend per provider that was called at least once, in ``ProviderName`` order."""
        totals: dict[ProviderName, float] = {}
        entries = self.entries
        for provider in _PROVIDERS:
            costs = [e.cost_usd for e in entries if e.provider == provider]
            if costs:
                totals[provider] = math.fsum(costs)
        return totals

    def by_stage(self) -> dict[StageName, float]:
        """Spend per stage that was called at least once, in pipeline order."""
        totals: dict[StageName, float] = {}
        entries = self.entries
        for stage in STAGE_ORDER:
            costs = [e.cost_usd for e in entries if e.stage == stage]
            if costs:
                totals[stage] = math.fsum(costs)
        return totals

    def check(self, worst_case_usd: float, what: str) -> None:
        """Raise ``BudgetExceeded`` if ``spent + worst_case_usd > cap``.

        Worst cases reserved by in-flight ``metered_call``s count as spent.
        """
        _validate_amount(worst_case_usd, "worst_case_usd")
        with self._lock:
            self._check_locked(worst_case_usd, what)

    def record(self, entry: LedgerEntry) -> None:
        """Append in memory and (if bound to a path) append one line to the JSONL file with fsync."""
        if entry.run_id != self.run_id:
            raise ValueError(f"entry for run {entry.run_id!r} recorded on ledger for {self.run_id!r}")
        line = entry.model_dump_json() + "\n"
        with self._lock:
            self._entries.append(entry)
            if self.path is not None:
                self._append_line(line)

    # -- internals ---------------------------------------------------------------------------

    def _spent(self) -> float:
        return math.fsum(e.cost_usd for e in self._entries)

    def _committed(self) -> float:
        return self._spent() + math.fsum(self._reserved.values())

    def _check_locked(self, worst_case_usd: float, what: str) -> None:
        committed = self._committed()
        if committed + worst_case_usd > self.cap_usd + BUDGET_EPSILON_USD:
            raise BudgetExceeded(cap_usd=self.cap_usd, spent_usd=committed, requested_usd=worst_case_usd, what=what)

    def _available(self) -> float:
        """Budget not yet spent or reserved by in-flight calls."""
        with self._lock:
            return max(0.0, self.cap_usd - self._committed())

    def _reserve(self, worst_case_usd: float, what: str) -> int:
        """Atomically check and reserve ``worst_case_usd`` until ``_release``."""
        _validate_amount(worst_case_usd, "worst_case_usd")
        with self._lock:
            self._check_locked(worst_case_usd, what)
            token = self._next_reservation
            self._next_reservation += 1
            self._reserved[token] = worst_case_usd
            return token

    def _release(self, token: int) -> None:
        with self._lock:
            self._reserved.pop(token, None)

    def _append_line(self, line: str) -> None:
        assert self.path is not None
        created = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        if created:
            _fsync_dir(self.path.parent)


def _validate_amount(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")


def _fsync_dir(directory: Path) -> None:
    """Make a newly created file's directory entry durable (best effort; not all filesystems allow it)."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


MIN_CLAUDE_CODE_BUDGET_USD = 0.0001
"""Smallest ``--max-budget-usd`` Claude Code accepts (``ClaudeCodeProvider.build_argv``); below it the call is refused."""


def estimate_tokens(text: str) -> int:
    """Conservative local token estimate: ``ceil(len(text) / CHARS_PER_TOKEN_WORST_CASE)``."""
    return math.ceil(len(text) / CHARS_PER_TOKEN_WORST_CASE)


def metered_call(
    ledger: Ledger,
    provider: Provider,
    request: CompletionRequest,
    *,
    stage: StageName,
    purpose: str = "",
) -> CompletionResult:
    """The only sanctioned way to call a provider.

    1. Clamp ``request.max_budget_usd`` to ``ledger.remaining_usd`` minus the provider's per-turn headroom
       (it matters only for Claude Code).
    2. ``worst = provider.worst_case_cost(request)``, then ``ledger.check(worst, ...)``.
    3. ``result = provider.complete(request)``.
    4. Record a ``LedgerEntry`` with ``result.cost_usd``, ``result.usage`` and ``worst``.
    5. Return ``result``.

    If ``provider.complete`` raises after partially spending (e.g. Claude Code hit its budget
    and reported ``total_cost_usd``), the provider raises ``ProviderError`` with ``cost_usd``
    set. That cost is recorded here before re-raising. Any other exception, including
    ``KeyboardInterrupt``, is recorded at the reserved worst case, since the spend is unknown.

    "Remaining" excludes worst cases reserved by other in-flight calls. A Claude Code call
    (unset ``max_budget_usd`` defaults to all remaining budget) with nothing left to spend
    raises ``BudgetExceeded`` instead of launching a session
    with less than ``MIN_CLAUDE_CODE_BUDGET_USD``.
    """
    from maf.providers.base import ProviderError  # runtime import keeps ledger below providers

    what = f"{stage}/{purpose or 'call'} via {provider.name}:{request.model}"
    available = ledger._available()
    if provider.name == "claude_code" or request.max_budget_usd is not None:
        headroom = turn_headroom(provider, request)
        spendable = max(0.0, available - headroom)
        requested = request.max_budget_usd
        clamped = spendable if requested is None else min(requested, spendable)
        if provider.name == "claude_code" and clamped < MIN_CLAUDE_CODE_BUDGET_USD:
            raise BudgetExceeded(
                cap_usd=ledger.cap_usd,
                spent_usd=ledger.cap_usd - available,
                requested_usd=(requested or 0.0) + headroom,
                what=what,
            )
        if clamped != requested:
            request = request.model_copy(update={"max_budget_usd": clamped})

    worst = provider.worst_case_cost(request)
    token = ledger._reserve(worst, what)

    def record_failure(cost_usd: float, exc: BaseException) -> None:
        ledger.record(
            LedgerEntry(
                ts=datetime.now(UTC),
                run_id=ledger.run_id,
                stage=stage,
                agent=provider.agent,
                provider=provider.name,
                model=request.model,
                cost_usd=cost_usd,
                worst_case_usd=worst,
                purpose=purpose,
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
        )

    try:
        try:
            result = provider.complete(request)
        except ProviderError as exc:
            if exc.cost_usd > 0:
                record_failure(exc.cost_usd, exc)
            raise
        except BaseException as exc:
            # Interrupt (Ctrl+C) or an adapter bug mid-call: the request may already be billed, and Claude Code
            # may have spent up to its budget. Charge the reserved worst case so a resume cannot overspend.
            record_failure(worst, exc)
            raise
        if result.cost_usd > worst + BUDGET_EPSILON_USD:
            log.warning("%s cost $%.4f, above its worst-case estimate $%.4f", what, result.cost_usd, worst)
        ledger.record(
            LedgerEntry(
                ts=datetime.now(UTC),
                run_id=ledger.run_id,
                stage=stage,
                agent=provider.agent,
                provider=result.provider,
                model=result.model or request.model,
                usage=result.usage,
                cost_usd=result.cost_usd,
                worst_case_usd=worst,
                purpose=purpose,
            )
        )
        return result
    finally:
        ledger._release(token)


def turn_headroom(provider: Provider, request: CompletionRequest) -> float:
    """Budget held back from Claude Code's ``--max-budget-usd``. The CLI checks its cap between turns, so one
    turn can overshoot it; providers expose the bound as ``turn_headroom_usd(request)`` (0 when absent)."""
    headroom = getattr(provider, "turn_headroom_usd", None)
    if not callable(headroom):
        return 0.0
    value = float(headroom(request))
    _validate_amount(value, "turn_headroom_usd")
    return value
