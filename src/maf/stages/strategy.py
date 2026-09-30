"""Stage 02, Strategy: ChatGPT consumes the ingestion report, weighs options, and writes the strategy and execution brief.

Owner: stages.

Consumes ``01a-routing`` (all sections) and ``01-ingestion`` (all sections), plus ``ctx.review_note``
when re-run after a review. Produces ``02-strategy`` (kind STRATEGY, ``from: chatgpt``,
``to: claude``). The review gate itself belongs to the pipeline: with ``--review`` the pipeline pauses
after this stage, and the user may hand-edit ``02-strategy.md`` in Obsidian before ``maf resume``.

``## Acceptance Criteria`` has a line grammar that Python reads, part of the handoff spec (``maf.handoff``:
``ACCEPTANCE_CRITERIA_GRAMMAR`` in ``format_spec(STRATEGY)``, ``parse_acceptance_criteria``, re-exported here): one item
per criterion, ``- AC-<n> [hard|soft]: text`` (``CRITERION_RE``), with details on continuation lines indented by two
spaces. The prompt adds mode-specific required criteria (``criteria_guidance``) and calibrates the rest: at most
``MAX_HARD_CRITERIA`` (6) hard criteria, each one the brief directly requires, demonstrable within the run's budget
(the prompt states it: ``budget_note``, with what one Claude Code work session may spend) and naming how it is
checked; everything else soft; no absolute provenance or coverage demands ("every", "all") the
brief does not make. More hard criteria fail the stage's check (``hard_criteria_errors``), which goes through the one
repair. A section that does not follow the grammar is rewritten into it (``normalize_acceptance_criteria``) rather
than sent back for repair: unlabeled criteria become ``hard``, the conservative reading, and unnumbered ones get the
next free id. The parser is equally lenient, so a hand-edited note still yields its criteria (and a hand-edited note
may keep more hard criteria: the review gate re-validates only the handoff grammar).
"""

from __future__ import annotations

import logging

from maf import handoff as hf
from maf import vault as _vault
from maf.handoff import (
    CRITERION_RE,
    MAX_HARD_CRITERIA,
    Criterion,
    Handoff,
    HandoffKind,
    acceptance_criteria_errors,
    hard_criteria_errors,
    normalize_acceptance_criteria,
    parse_acceptance_criteria,
)
from maf.config import Settings
from maf.prompts import render_prompt
from maf.stages.base import (
    NoteOut,
    StageContext,
    StageOutput,
    generate_handoff,
    render_inputs,
    review_block,
    role_system,
    with_sections,
)
from maf.types import ExecutionMode, StageName

log = logging.getLogger(__name__)

CRITERIA_SECTION = hf.ACCEPTANCE_CRITERIA


class StrategyBackend:
    name: StageName = "strategy"

    def run_stage(self, ctx: StageContext) -> StageOutput:
        consumed = [_vault.note_name(HandoffKind.ROUTING), _vault.note_name(HandoffKind.INGESTION)]
        notes = {name: ctx.read(name) for name in consumed}
        prompt = render_prompt(
            "strategy",
            brief=ctx.index.brief,
            inputs=render_inputs(notes),
            max_hard=str(MAX_HARD_CRITERIA),
            budget=budget_note(ctx.index.mode, ctx.index.budget_usd, ctx.settings),
            criteria_guidance=criteria_guidance(ctx.index.mode, source_audit=ctx.settings.source_audit),
            review_note=review_block(ctx.review_note),
            format_spec=hf.format_spec(HandoffKind.STRATEGY),
        )
        strategy = generate_handoff(
            ctx,
            "chatgpt",
            HandoffKind.STRATEGY,
            system=role_system("chatgpt", ctx.settings),
            prompt=prompt,
            to="claude",
            inputs=consumed,
            purpose="strategy",
            check=check_strategy,
        )
        section = strategy.section(CRITERIA_SECTION)
        normalized = normalize_acceptance_criteria(section)
        if normalized != section:
            log.info("strategy: rewrote ## %s into the AC-<n> [hard|soft] grammar", CRITERIA_SECTION)
            strategy = with_sections(strategy, {CRITERIA_SECTION: normalized})
        return StageOutput(notes=[NoteOut(_vault.note_name(HandoffKind.STRATEGY), strategy)])


def check_strategy(handoff: Handoff) -> list[str]:
    """``generate_handoff`` check: at most ``MAX_HARD_CRITERIA`` hard acceptance criteria (``hard_criteria_errors``)."""
    return hard_criteria_errors(handoff.section(CRITERIA_SECTION))


def budget_note(mode: ExecutionMode | None, budget_usd: float, settings: Settings) -> str:
    """The strategy prompt's budget paragraph (``{{budget}}``), so the model can tell which criteria this run can
    demonstrate: the run's whole budget, the cross-check loop cap and, for code/mixed runs (or a mode not known yet),
    what one Claude Code work session may spend (``OutputLimits.claude_code_budget_usd``)."""
    loops = settings.max_crosscheck_loops
    text = (
        f"This run's whole budget is ${budget_usd:.2f}: it pays for every model call of every stage, including each "
        f"cross-check loop (at most {loops})."
    )
    if mode == "prose":
        return (
            f"{text} Every unmet hard criterion can cost another round of writing and review, so make hard only what "
            "this budget can demonstrate."
        )
    session = settings.output_limits.claude_code_budget_usd
    return (
        f"{text} Each Claude Code work session (an execution pass, or a cross-check's fix pass) may spend up to "
        f"${session:.2f} of it, and every unmet hard criterion can cost another execution and cross-check round, so "
        "make hard only what this budget can demonstrate."
    )


def criteria_guidance(mode: ExecutionMode | None, *, source_audit: bool) -> str:
    """The criteria the strategy must include for this run, as a prompt block. ``mode`` None (unknown) includes the
    code criteria conditionally."""
    code = mode in ("code", "mixed")
    when_code = "" if code else "If the deliverables include code: "
    lines = []
    if code or mode is None:
        lines += [
            f"- {when_code}a hard clean-room criterion: a fresh copy of the exported deliverables (plus the provisioned "
            "FreeRTOS kernel and the user's own input files, which the user has; no pipeline notes, no other files) "
            "rebuilds and reproduces every reported result with the single reproduction command documented in the "
            "README.",
            f"- {when_code}a hard repeatability criterion: every test suite and negative control gives the same "
            "result on at least 3 consecutive runs.",
            "- If the work involves modelling, simulation or control design: a soft criterion (hard only if the "
            "request asks for a sensitivity or robustness analysis) that the main conclusions are re-checked under "
            "parameter sensitivity and model mismatch (for example 5-10 % error in the plant model), stating which "
            "survive.",
        ]
    if source_audit:
        lines.append(
            "- If the deliverables cite sources: a hard criterion that every reference is verified by the source "
            "audit, meaning the cited work exists and supports the specific claim, value or formula attributed to it."
        )
    else:
        lines.append(
            "- If the deliverables cite sources: a hard criterion that every reference is listed in the ingestion "
            "report's `## Sources` and supports the specific claim, value or formula attributed to it."
        )
    return (
        "Include these criteria, among your own. The hard ones count toward the "
        f"{MAX_HARD_CRITERIA} hard criteria. maf requires them for every run of this kind, so their \"every\" is not "
        "the kind of absolute demand to avoid in your own criteria:\n\n" + "\n".join(lines)
    )


__all__ = [
    "CRITERIA_SECTION",
    "CRITERION_RE",
    "MAX_HARD_CRITERIA",
    "Criterion",
    "StrategyBackend",
    "acceptance_criteria_errors",
    "budget_note",
    "check_strategy",
    "criteria_guidance",
    "hard_criteria_errors",
    "normalize_acceptance_criteria",
    "parse_acceptance_criteria",
]
