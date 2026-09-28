"""Stage 02, Strategy: ChatGPT consumes the ingestion report, weighs options, and writes the strategy and execution brief.

Owner: stages.

Consumes ``01a-routing`` (all sections) and ``01-ingestion`` (all sections), plus ``ctx.review_note``
when re-run after a review. Produces ``02-strategy`` (kind STRATEGY, ``from: chatgpt``,
``to: claude``). The review gate itself belongs to the pipeline: with ``--review`` the pipeline pauses
after this stage, and the user may hand-edit ``02-strategy.md`` in Obsidian before ``maf resume``.
"""

from __future__ import annotations

from maf import handoff as hf
from maf import vault as _vault
from maf.handoff import HandoffKind
from maf.prompts import render_prompt
from maf.stages.base import (
    NoteOut,
    StageContext,
    StageOutput,
    generate_handoff,
    render_inputs,
    review_block,
    role_system,
)
from maf.types import StageName


class StrategyBackend:
    name: StageName = "strategy"

    def run_stage(self, ctx: StageContext) -> StageOutput:
        consumed = [_vault.note_name(HandoffKind.ROUTING), _vault.note_name(HandoffKind.INGESTION)]
        notes = {name: ctx.read(name) for name in consumed}
        prompt = render_prompt(
            "strategy",
            brief=ctx.index.brief,
            inputs=render_inputs(notes),
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
        )
        return StageOutput(notes=[NoteOut(_vault.note_name(HandoffKind.STRATEGY), strategy)])
