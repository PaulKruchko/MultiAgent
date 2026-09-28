"""Stage 03, Execution: Claude produces the definitive artifacts.

Owner: stages.

Mode (``ctx.index.mode``):
- ``code`` / ``mixed``: Claude Code (``claude_code`` role) in the workspace. The prompt is written to
  ``workspace/.maf/execution-r<round>.md`` for traceability and sent on stdin. The final message must be
  the EXECUTION handoff body; ``## Artifacts`` lists workspace-relative paths, one per bullet, as
  ``- `path` - description``. PNGs under the workspace referenced there are copied to ``assets/`` and
  embedded; the stage rewrites the Artifacts bullets to add the embeds (``Vault.copy_asset`` reuses
  identical files, so re-runs are idempotent). Listed paths that do not exist are flagged in
  ``## Known Limitations`` rather than rejected.
- ``prose``: Claude Messages (``claude`` role) returns the handoff body with the full document under
  an H3 in ``## Artifacts`` (see ``prompts/execution_prose.md``). The stage moves that document to
  ``workspace/document.md`` and replaces it in ``## Artifacts`` with the bullet ``- `document.md` - <title>``,
  so the note stays small.

Consumes ``01-ingestion``, ``02-strategy`` (possibly user-edited), and on round > 1 the previous
``04-crosscheck[-rN]`` ``## Unresolved Critical`` + ``## Rulings``. Produces ``03-execution[-rN]``.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

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
    review_block,
    role_system,
    with_sections,
    write_workspace_file,
)
from maf.types import ExecutionMode, StageName

FREERTOS_DIR = "FreeRTOS-Kernel"
"""Workspace-relative location of the provisioned FreeRTOS kernel (``Settings.freertos_path``)."""

PROSE_DOCUMENT = "document.md"
"""Workspace-relative path of the prose deliverable (and of the mixed-mode document by convention)."""

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"})

MODE_GUIDANCE: dict[ExecutionMode, str] = {
    "code": (
        "The deliverable is software. Keep sources, build files, tests and test logs in the workspace, "
        "and save the output of every test run you rely on to a log file you list in `## Artifacts`."
    ),
    "mixed": (
        "The deliverable is a document backed by code, simulations or plots that you actually run. "
        f"Write the document as `{PROSE_DOCUMENT}` in the workspace root (Obsidian Markdown, math as "
        "`$...$`/`$$...$$`). Save every plot as a PNG with a unique, descriptive file name under `plots/` and "
        "embed it in the document as `![[file-name.png]]`. List the document, the simulation code and every "
        "plot in `## Artifacts`."
    ),
}

_ARTIFACT_BULLET = re.compile(r"^\s*[-*+]\s+`(?P<path>[^`\n]+)`")
_H3 = re.compile(r"^###\s+(?P<title>\S.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


class ExecutionBackend:
    name: StageName = "execution"

    def run_stage(self, ctx: StageContext) -> StageOutput:
        mode = require_mode(ctx)
        consumed = [_vault.note_name(HandoffKind.INGESTION), _vault.note_name(HandoffKind.STRATEGY)]
        notes = {name: ctx.read(name) for name in consumed}
        fix_context = ""
        if ctx.round > 1:
            previous = _vault.note_name(HandoffKind.CROSSCHECK, ctx.round - 1)
            consumed.append(previous)
            fix_context = render_fix_context(previous, ctx.read(previous))
        inputs = render_inputs(notes)
        if mode == "prose":
            handoff = self._run_prose(ctx, consumed, inputs, fix_context)
        else:
            handoff = self._run_code(ctx, mode, consumed, inputs, fix_context)
        return StageOutput(notes=[NoteOut(_vault.note_name(HandoffKind.EXECUTION, ctx.round), handoff)])

    def _run_code(
        self, ctx: StageContext, mode: ExecutionMode, consumed: list[str], inputs: str, fix_context: str
    ) -> Handoff:
        kernel = provision_freertos(ctx.settings.freertos_path, ctx.paths.workspace)
        prompt = render_prompt(
            "execution_code",
            round=str(ctx.round),
            mode_guidance=MODE_GUIDANCE[mode],
            toolchain=freertos_note(kernel),
            inputs=inputs,
            fix_context=fix_context,
            review_note=review_block(ctx.review_note),
            format_spec=hf.format_spec(HandoffKind.EXECUTION),
        )
        write_workspace_file(ctx, f"{WORKSPACE_META_DIR}/execution-r{ctx.round}.md", prompt)
        workspace = ctx.paths.workspace
        handoff = generate_handoff(
            ctx,
            "claude_code",
            HandoffKind.EXECUTION,
            system=role_system("claude_code", ctx.settings),
            prompt=prompt,
            to="crosscheck",
            inputs=consumed,
            purpose="execution",
            check=lambda h: check_artifacts(h, workspace),
        )
        return finish_code_artifacts(ctx, handoff)

    def _run_prose(self, ctx: StageContext, consumed: list[str], inputs: str, fix_context: str) -> Handoff:
        current = ctx.paths.workspace / PROSE_DOCUMENT
        if ctx.round > 1 and current.is_file():
            fix_context += (
                "\n\n## Current document\n\nRevise this version; keep what the cross-check did not fault.\n\n"
                f"<document path=\"{PROSE_DOCUMENT}\">\n{demote_headings(current.read_text(encoding='utf-8'))}\n</document>"
            )
        prompt = render_prompt(
            "execution_prose",
            round=str(ctx.round),
            inputs=inputs,
            fix_context=fix_context,
            review_note=review_block(ctx.review_note),
            format_spec=hf.format_spec(HandoffKind.EXECUTION),
        )
        handoff = generate_handoff(
            ctx,
            "claude",
            HandoffKind.EXECUTION,
            system=role_system("claude", ctx.settings),
            prompt=prompt,
            to="crosscheck",
            inputs=consumed,
            purpose="execution",
            check=check_prose_document,
        )
        title, document = extract_document(handoff.section("Artifacts"))
        write_workspace_file(ctx, PROSE_DOCUMENT, promote_headings(document))
        return with_sections(handoff, {"Artifacts": f"- `{PROSE_DOCUMENT}` - {title}"})


def provision_freertos(source: Path | None, workspace: Path) -> str | None:
    """Copy the local FreeRTOS-Kernel clone to ``workspace/FREERTOS_DIR`` (Claude Code has no network).

    Returns the workspace-relative directory, or None when no kernel is configured or the source is missing.
    An existing copy is kept, so re-runs and later rounds see the same (possibly patched) tree."""
    target = workspace / FREERTOS_DIR
    if target.is_dir():
        return FREERTOS_DIR
    if source is None or not source.expanduser().is_dir():
        return None
    tmp = workspace / f".{FREERTOS_DIR}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(source.expanduser(), tmp, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    os.replace(tmp, target)
    return FREERTOS_DIR


def freertos_note(kernel: str | None) -> str:
    """Prompt paragraph telling Claude Code where the FreeRTOS kernel is (or that none is available)."""
    if kernel is None:
        return "No FreeRTOS kernel is available in this workspace; say so if the task needs one."
    return (
        f"A FreeRTOS kernel clone is at `{kernel}/` (the POSIX simulator port is in "
        f"`{kernel}/portable/ThirdParty/GCC/Posix/`). Build against it in place; do not list it in `## Artifacts`."
    )


def require_mode(ctx: StageContext) -> ExecutionMode:
    if ctx.index.mode is None:
        raise ValueError("run has no execution mode; ingestion must run first")
    return ctx.index.mode


def render_fix_context(name: str, crosscheck: Handoff) -> str:
    """Prompt block carrying the previous cross-check's unresolved critical issues and rulings."""
    return (
        f"## Fix context from the previous cross-check ([[{name}]])\n\n"
        "The previous pass left the critical issues below unresolved. Resolving them is the first priority "
        "of this pass; the rulings show what the adjudicator required.\n\n"
        + render_inputs({name: crosscheck}, {name: ("Unresolved Critical", "Rulings")})
    )


def parse_artifact_paths(artifacts_section: str) -> list[str]:
    """Extract workspace-relative paths from ``- `path` - description`` bullets."""
    paths: list[str] = []
    for line in artifacts_section.splitlines():
        match = _ARTIFACT_BULLET.match(line)
        if match:
            path = match.group("path").strip()
            if path and path not in paths:
                paths.append(path)
    return paths


def resolve_in_workspace(workspace: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``workspace``; ``ValueError`` if it escapes (``..``, absolute, symlink out)."""
    rel = rel.strip()
    if not rel:
        raise ValueError("empty artifact path")
    if Path(rel).is_absolute() or rel.startswith("~"):
        raise ValueError(f"artifact path must be workspace-relative: {rel!r}")
    root = workspace.resolve()
    resolved = (root / rel).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"artifact path escapes the workspace: {rel!r}")
    return resolved


def check_artifacts(handoff: Handoff, workspace: Path) -> list[str]:
    """Code/mixed grammar for ``## Artifacts``: at least one ``- `path` - description`` bullet, and no
    path outside the workspace. Missing files are reported in the note, not treated as invalid."""
    paths = parse_artifact_paths(handoff.section("Artifacts"))
    if not paths:
        return ["## Artifacts must list workspace-relative paths as bullets: - `relative/path` - description"]
    errors = []
    for rel in paths:
        try:
            resolve_in_workspace(workspace, rel)
        except ValueError as exc:
            errors.append(f"## Artifacts: {exc}")
    return errors


def finish_code_artifacts(ctx: StageContext, handoff: Handoff) -> Handoff:
    """Embed listed images (copied to ``assets/``) and flag listed paths that do not exist."""
    workspace = ctx.paths.workspace
    artifacts = handoff.section("Artifacts")
    embeds: dict[str, str] = {}
    missing: list[str] = []
    for rel in parse_artifact_paths(artifacts):
        path = resolve_in_workspace(workspace, rel)
        if not path.exists():
            missing.append(rel)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            embeds[rel] = ctx.vault.copy_asset(ctx.run_id, path)

    updates: dict[str, str] = {}
    if embeds:
        lines = []
        for line in artifacts.splitlines():
            match = _ARTIFACT_BULLET.match(line)
            embed = embeds.get(match.group("path").strip()) if match else None
            lines.append(f"{line.rstrip()} {embed}" if embed and embed not in line else line)
        updates["Artifacts"] = "\n".join(lines)
    if missing:
        limitations = handoff.section("Known Limitations").strip()
        flagged = "\n".join(f"- maf: listed artifact `{rel}` was not found in the workspace." for rel in missing)
        updates["Known Limitations"] = flagged if limitations == hf.NONE_MARKER else f"{limitations}\n\n{flagged}"
    return with_sections(handoff, updates) if updates else handoff


def check_prose_document(handoff: Handoff) -> list[str]:
    if _find_h3(handoff.section("Artifacts").splitlines()) is None:
        return [
            "## Artifacts must contain the full document under an H3 title line (### Title); "
            "use H4 and deeper for its internal headings, never H1 or H2"
        ]
    return []


def extract_document(artifacts_section: str) -> tuple[str, str]:
    """``(title, document)`` where the document starts at the first H3 outside code fences."""
    lines = artifacts_section.splitlines()
    start = _find_h3(lines)
    if start is None:
        raise ValueError("no H3 document title in ## Artifacts")
    match = _H3.match(lines[start])
    assert match is not None
    return match.group("title").strip(), "\n".join(lines[start:]).strip() + "\n"


def _find_h3(lines: list[str]) -> int | None:
    fenced = False
    for i, line in enumerate(lines):
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced and _H3.match(line):
            return i
    return None


def _shift_headings(text: str, delta: int) -> str:
    out: list[str] = []
    fenced = False
    for line in text.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
        elif not fenced:
            match = re.match(r"^(#{1,6})(\s)", line)
            if match:
                level = min(6, max(1, len(match.group(1)) + delta))
                line = "#" * level + line[len(match.group(1)) :]
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def promote_headings(document: str) -> str:
    """Standalone form of an embedded document: H3 becomes H1, H4 becomes H2, and so on."""
    return _shift_headings(document, -2)


def demote_headings(document: str) -> str:
    """Inverse of ``promote_headings`` for putting a standalone document back into a prompt/section."""
    return _shift_headings(document, +2)
