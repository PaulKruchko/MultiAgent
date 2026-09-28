"""Stage 05, Final: Claude assembles the final deliverable, and the run index is updated.

Owner: stages.

Consumes the latest ``03-execution[-rN]`` and ``04-crosscheck[-rN]`` (plus ``02-strategy`` ``## Acceptance
Criteria``). Claude (Messages) writes ``05-final`` (kind FINAL, ``to: user``). ``## Deliverables`` links
each item copied into ``deliverables/`` (wikilinks for .md, embeds for images, inline code paths
for trees). ``## Provenance`` has one bullet per consumed note, as a wikilink. If
``ctx.index.unresolved_critical > 0`` (loops exhausted), the run ends ``completed_with_issues``: the prompt says
so, and ``## Limitations`` must list those issues verbatim. The stage copies the final artifacts from the
workspace into ``deliverables/``.

Python guarantees those rules after generation: missing deliverable links, provenance links and
unresolved issues are appended rather than failing the run, and a ``completed_with_issues`` note gets a
warning callout opening ``## Summary`` (linking the last cross-check) plus the ``maf/completed-with-issues``
tag. Artifact trees too large for a vault (more than ``MAX_DELIVERABLE_FILES`` files or
``MAX_DELIVERABLE_BYTES``) stay in the workspace and are listed by path instead. Deliverable links use
vault-relative paths (``runs/<run_id>/deliverables/...``) because every run has a ``deliverables/`` folder,
and bare image embeds inside copied Markdown documents are rewritten the same way.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from maf import handoff as hf
from maf import vault as _vault
from maf.handoff import Handoff, HandoffKind
from maf.prompts import render_prompt
from maf.stages.base import (
    WORKSPACE_META_DIR,
    NoteOut,
    StageContext,
    StageOutput,
    generate_handoff,
    render_inputs,
    role_system,
    with_sections,
)
from maf.stages.execution import IMAGE_SUFFIXES, parse_artifact_paths, resolve_in_workspace
from maf.types import RunStatus, StageName

log = logging.getLogger(__name__)

MAX_DELIVERABLE_FILES = 2_000
MAX_DELIVERABLE_BYTES = 50_000_000

ISSUES_TAG = "maf/completed-with-issues"
"""Extra tag on a 05-final written for a ``completed_with_issues`` run (searchable in Obsidian)."""


@dataclass(frozen=True)
class Deliverable:
    """One artifact as seen from the final note."""

    rel: str
    """Workspace-relative path (also the path under ``deliverables/`` when copied)."""
    source: Path
    copied: bool
    is_dir: bool
    vault_path: str = ""
    """Vault-relative path of the copy (``runs/<run_id>/deliverables/<rel>``); empty when not copied.
    Links use it because every run has a ``deliverables/`` folder, so short links would be ambiguous."""

    @property
    def link(self) -> str:
        """How ``## Deliverables`` must reference it."""
        if not self.copied:
            return f"`{self.source}`"
        suffix = PurePosixPath(self.rel).suffix.lower()
        if self.is_dir:
            return f"`{self.vault_path}/`"
        if suffix == ".md":
            return f"[[{self.vault_path.removesuffix('.md')}|{PurePosixPath(self.rel).stem}]]"
        if suffix in IMAGE_SUFFIXES:
            return f"![[{self.vault_path}]]"
        return f"`{self.vault_path}`"

    @property
    def bullet(self) -> str:
        note = "" if self.copied else " (left in the workspace: too large to copy into the vault)"
        return f"- {self.link}{note}"


class FinalBackend:
    name: StageName = "final"

    def run_stage(self, ctx: StageContext) -> StageOutput:
        strategy_name = _vault.note_name(HandoffKind.STRATEGY)
        exec_name = ctx.latest_note_name(HandoffKind.EXECUTION)
        cross_name = ctx.latest_note_name(HandoffKind.CROSSCHECK)
        consumed = [strategy_name, exec_name, cross_name]
        notes = {name: ctx.read(name) for name in consumed}

        deliverables = copy_deliverables(ctx, notes[exec_name])
        open_critical = ctx.index.unresolved_critical
        unresolved = unresolved_lines(notes[cross_name]) if open_critical > 0 else []
        prompt = render_prompt(
            "final",
            brief=ctx.index.brief,
            deliverables="\n".join(d.bullet for d in deliverables) or "None.",
            unresolved=_unresolved_block(open_critical, cross_name, unresolved),
            inputs=render_inputs(notes, {strategy_name: ("Acceptance Criteria",)}),
            format_spec=hf.format_spec(HandoffKind.FINAL),
        )
        final = generate_handoff(
            ctx,
            "claude",
            HandoffKind.FINAL,
            system=role_system("claude", ctx.settings),
            prompt=prompt,
            to="user",
            inputs=consumed,
            purpose="final",
        )
        final = enforce_final_rules(final, deliverables, consumed, unresolved)
        if open_critical > 0:
            final = mark_completed_with_issues(final, open_critical, cross_name)
        return StageOutput(
            notes=[NoteOut(_vault.note_name(HandoffKind.FINAL), final)],
            deliverables=[d.source for d in deliverables if d.copied],
        )


def copy_deliverables(ctx: StageContext, execution: Handoff) -> list[Deliverable]:
    """Copy every existing artifact of the latest execution into ``deliverables/`` under its workspace-relative
    path (re-running overwrites the same targets). Missing artifacts are skipped; oversized trees are not copied."""
    out: list[Deliverable] = []
    for rel in parse_artifact_paths(execution.section("Artifacts")):
        try:
            source = resolve_in_workspace(ctx.paths.workspace, rel)
        except ValueError as exc:
            log.warning("skipping deliverable %r: %s", rel, exc)
            continue
        if not source.exists():
            log.warning("skipping deliverable %r: not found in the workspace", rel)
            continue
        clean = source.relative_to(ctx.paths.workspace.resolve()).as_posix()
        if clean == "." or clean.split("/")[0] == WORKSPACE_META_DIR:
            continue
        if source.is_dir() and _too_large(source):
            out.append(Deliverable(clean, source, copied=False, is_dir=True))
            continue
        dst = ctx.vault.copy_deliverable(ctx.run_id, source, clean)
        vault_path = dst.resolve().relative_to(ctx.vault.root.resolve()).as_posix()
        out.append(Deliverable(clean, source, copied=True, is_dir=source.is_dir(), vault_path=vault_path))
    relink_image_embeds(ctx, [d for d in out if d.copied and not d.is_dir and d.rel.lower().endswith(".md")])
    return out


_EMBED = re.compile(r"!\[\[(?P<target>[^\]|/\n]+?)(?P<alias>\|[^\]\n]*)?\]\]")


def relink_image_embeds(ctx: StageContext, documents: list[Deliverable]) -> None:
    """Rewrite bare image embeds (``![[plot.png]]``) in copied Markdown deliverables to the vault-relative
    path of that image under this run's ``deliverables/``, so they cannot resolve to another run's file.
    Names that match several copied images, or none, are left alone."""
    if not documents:
        return
    root = ctx.vault.root.resolve()
    by_name: dict[str, list[str]] = {}
    for image in sorted(ctx.paths.deliverables.rglob("*")):
        if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
            by_name.setdefault(image.name, []).append(image.resolve().relative_to(root).as_posix())

    def replace(match: re.Match[str]) -> str:
        found = by_name.get(match.group("target").strip(), [])
        return f"![[{found[0]}{match.group('alias') or ''}]]" if len(found) == 1 else match.group(0)

    for doc in documents:
        path = root / doc.vault_path
        text = path.read_text(encoding="utf-8")
        updated = _EMBED.sub(replace, text)
        if updated != text:
            _vault.atomic_write_text(path, updated)


def _too_large(directory: Path) -> bool:
    count = size = 0
    for path in directory.rglob("*"):
        if path.is_file():
            count += 1
            size += path.stat().st_size
            if count > MAX_DELIVERABLE_FILES or size > MAX_DELIVERABLE_BYTES:
                return True
    return False


def unresolved_lines(crosscheck: Handoff) -> list[str]:
    """The ``## Unresolved Critical`` bullet lines of a cross-check note (empty for ``None.``)."""
    text = crosscheck.sections.get("Unresolved Critical", "").strip()
    if not text or text == hf.NONE_MARKER:
        return []
    return [line.strip() for line in text.splitlines() if line.strip().startswith("- ")]


def _unresolved_block(count: int, crosscheck: str, lines: list[str]) -> str:
    if count <= 0:
        return ""
    status = RunStatus.COMPLETED_WITH_ISSUES.value
    text = (
        f"## Unresolved critical issues (run status: {status})\n\n"
        f"The cross-check loop cap was reached with {count} critical issue(s) still unresolved (see "
        f"[[{crosscheck}]]), so this run ends with status `{status}`, not `completed`. Say so plainly in "
        "`## Summary`, and do not present work these issues affect as verified."
    )
    if lines:
        text += (
            " `## Limitations` must list every one of them verbatim, exactly as written here:\n\n" + "\n".join(lines)
        )
    return text


def status_callout(count: int, crosscheck: str) -> str:
    """The warning that opens ``## Summary`` of a ``completed_with_issues`` final note."""
    return (
        f"> [!warning] Run status: {RunStatus.COMPLETED_WITH_ISSUES.value}\n"
        f"> {count} critical issue(s) remain unresolved after the cross-check loop cap (see [[{crosscheck}]]); "
        "they are listed under Limitations."
    )


def mark_completed_with_issues(final: Handoff, count: int, crosscheck: str) -> Handoff:
    """Open ``## Summary`` with ``status_callout`` (unless present) and add ``ISSUES_TAG`` to the frontmatter."""
    callout = status_callout(count, crosscheck)
    summary = final.section("Summary").strip()
    if callout.split("\n", 1)[0] not in summary:
        final = with_sections(final, {"Summary": f"{callout}\n\n{summary}"})
    if ISSUES_TAG not in final.meta.tags:
        final = final.model_copy(update={"meta": final.meta.model_copy(update={"tags": [*final.meta.tags, ISSUES_TAG]})})
    return final


def enforce_final_rules(
    final: Handoff, deliverables: list[Deliverable], consumed: list[str], unresolved: list[str]
) -> Handoff:
    """Append whatever the model left out: deliverable links, provenance wikilinks, unresolved issues."""
    updates: dict[str, str] = {}

    def extend(section: str, lines: list[str], lead: str) -> None:
        current = final.section(section).strip()
        if not lines:
            return
        block = f"{lead}\n\n" + "\n".join(lines) if lead else "\n".join(lines)
        updates[section] = block if current in ("", hf.NONE_MARKER) else f"{current}\n\n{block}"

    text = final.section("Deliverables")
    extend("Deliverables", [d.bullet for d in deliverables if d.link not in text], "")
    provenance = final.section("Provenance")
    extend("Provenance", [f"- [[{name}]]" for name in consumed if f"[[{name}]]" not in provenance and f"[[{name}|" not in provenance], "")
    limitations = final.section("Limitations")
    extend(
        "Limitations",
        [line for line in unresolved if line not in limitations],
        "Critical issues left unresolved by the cross-check (added by maf):",
    )
    return with_sections(final, updates) if updates else final
