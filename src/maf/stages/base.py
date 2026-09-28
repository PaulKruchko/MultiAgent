"""Stage backend protocol, the context passed to each stage, and the shared generate-validate-repair loop.

Owner: stages.

A stage backend is pure with respect to run state. It reads prior notes through ``ctx``,
makes metered calls through ``ctx.call``, and returns a ``StageOutput``. **The pipeline**, not the
stage, writes the returned notes to the vault, updates run.md, and advances the state machine.
Exception: Claude Code writes files into the workspace, and stages may copy
assets/deliverables through ``ctx.vault`` (idempotent on re-run).

Stage code reaches ``maf.handoff``, ``maf.vault`` and ``maf.ledger`` functions through their modules
(``hf.build_handoff``, not a from-import), so there is exactly one binding to each contract function.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from collections.abc import Callable
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
from maf.providers import CompletionRequest, CompletionResult, ProviderError, Providers
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
    """Allowed keys: ``mode`` (ingestion), ``unresolved_critical`` (crosscheck). Others are rejected by the pipeline."""
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
    **request_kw: Any,
) -> Generated:
    """``generate_handoff`` that also returns the provider results (for citations, costs, raw payloads)."""
    spec = hf.format_spec(kind)
    if spec not in prompt:
        prompt = f"{prompt.rstrip()}\n\n{spec}\n"
    request_kw.setdefault("effort", DEFAULT_EFFORT[role])
    if role == "claude_code":
        request_kw.setdefault("max_budget_usd", ctx.settings.output_limits.claude_code_budget_usd)
    model = ctx.model(role)
    limit = max_output_tokens or default_output_tokens(ctx.settings, role)
    label = purpose or kind.value

    first = ctx.call(
        role,
        CompletionRequest.simple(model, prompt, system=system, max_output_tokens=limit, **request_kw),
        purpose=label,
    )
    results: list[CompletionResult] = [first]
    try:
        return Generated(_build_checked(ctx, role, kind, first, results, to, inputs, check), tuple(results))
    except HandoffInvalid as exc:
        log.warning("%s handoff from %s invalid, repairing: %s", kind, role, exc)
        errors = exc.errors

    # The repair only fixes the format: web search and attachments were used already, so drop them.
    repair_kw = {k: v for k, v in request_kw.items() if k not in ("web_search", "attachments", "max_search_queries")}
    repair = ctx.call(
        role,
        CompletionRequest.simple(
            model,
            hf.repair_prompt(kind, first.text, errors),
            system=system,
            max_output_tokens=limit,
            **repair_kw,
        ),
        purpose="repair",
    )
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
