"""Stage 03, Execution: Claude produces the definitive artifacts.

Owner: stages.

Mode (``ctx.index.mode``):
- ``code`` / ``mixed``: Claude Code (``claude_code`` role) in the workspace. Before the first Claude Code call
  of the run (in this process), ``ensure_sandbox`` runs the provider's sandbox preflight as a metered call
  (``purpose="preflight"``); ``SandboxUnavailable`` propagates and stops the run. The prompt is written to
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

Both modes get ``DELIVERABLE_RULES`` in the prompt (cite only the ingestion's verified sources, never pipeline notes,
no meta-commentary, numbers from the data, clean Markdown); code/mixed guidance adds ``EXPORT_RULES`` (what the export
ships and how the clean room rebuilds it). After the call, ``maf.lint.lint_deliverables`` runs over the deliverable
tree (the workspace minus ``export_excludes(settings)``, as the export and the cross-check see it). Critical and major
findings are listed in an extra ``## Lint`` section (``LINT_SECTION``) appended to the note, and all findings are
logged; a linter failure is logged and noted there in one line (``safe_lint``), never raised. The stage only reports them: the cross-check raises its own ``LINT`` issues from a fresh lint
and leaves this section out of what critics and final read (``without_lint``), so each finding is counted once.

Consumes ``01-ingestion``, ``02-strategy`` (possibly user-edited), and on round > 1 the previous
``04-crosscheck[-rN]`` ``## Unresolved Critical`` + ``## Rulings``. On an extra round after final
(``RunIndex.unmet_criteria`` is set) it also gets the unmet criteria and 05-final's ``## Acceptance`` and
``## Clean-room Reproduction``. Produces ``03-execution[-rN]``.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

from maf import handoff as hf
from maf import lint as _lint
from maf import vault as _vault
from maf.handoff import Handoff, HandoffKind
from maf.lint import LintIssue
from maf.prompts import render_prompt
from maf.stages.base import (
    WORKSPACE_META_DIR,
    NoteOut,
    StageContext,
    StageOutput,
    escape_note_tags,
    export_excludes,
    generate_handoff,
    render_inputs,
    review_block,
    role_system,
    with_sections,
    write_workspace_file,
)
from maf.types import ExecutionMode, StageName

log = logging.getLogger(__name__)

PREFLIGHT_PURPOSE = "preflight"
"""Ledger ``purpose`` of the Claude Code sandbox preflight."""

FREERTOS_DIR = "FreeRTOS-Kernel"
"""Workspace-relative location of the provisioned FreeRTOS kernel (``Settings.freertos_path``)."""

PROSE_DOCUMENT = "document.md"
"""Workspace-relative path of the prose deliverable (and of the mixed-mode document by convention)."""

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"})

EXPORT_RULES = (
    "The whole workspace tree is exported as the deliverables, except `.maf/`, `.claude/`, `inputs/`, "
    "`FreeRTOS-Kernel/`, version control, caches and build output (every `build/` directory, object files and "
    "libraries such as `*.o`, `*.a`, `*.so`, `*.elf`, `a.out`, CMake state), and a fresh copy of it is rebuilt with the "
    "README's reproduction command, with every generated file (results, logs, figures, binaries) treated as out of "
    "date. So every file the build, the tests or the documents need must be in the tree outside those paths, "
    "including files you add while fixing, and no command or document may use this workspace's absolute path."
)
"""How the code/mixed export and the clean room treat the workspace (``maf.vault.DEFAULT_EXPORT_EXCLUDES``,
``maf.stages.final.prepare_cleanroom``): in the execution guidance and the cross-check's fix instructions."""

_REPRODUCIBLE = (
    "Write a `README.md` that documents one reproduction command (such as `make all`) which rebuilds everything and "
    "reruns every test and result from a clean copy of the deliverables. " + EXPORT_RULES + " Run every test suite "
    "and negative control at least 3 times, save each run's output to a log you list, and make negative controls "
    "deterministic (fixed seeds, forced triggers), so that no run passes by luck."
)

MODE_GUIDANCE: dict[ExecutionMode, str] = {
    "code": (
        "The deliverable is software. Keep sources, build files, tests and test logs in the workspace, "
        "and save the output of every test run you rely on to a log file you list in `## Artifacts`. "
        + _REPRODUCIBLE
    ),
    "mixed": (
        "The deliverable is a document backed by code, simulations or plots that you actually run. "
        f"Write the document as `{PROSE_DOCUMENT}` in the workspace root (Obsidian Markdown, math as "
        "`$...$`/`$$...$$`). Save every plot as a PNG with a unique, descriptive file name under `plots/` and "
        "embed it in the document as `![[file-name.png]]`. List the document, the simulation code and every "
        "plot in `## Artifacts`. " + _REPRODUCIBLE + " For modelling or controller work, add sensitivity and "
        "model-mismatch studies (for example 5-10 % error in the plant model, and the modelling choices a result "
        "depends on), and state which conclusions survive them."
    ),
}

DELIVERABLE_RULES = """## Rules for the deliverables

The deliverables are read on their own, outside this pipeline. After you finish, Python lints every Markdown file in
the workspace for the problems below, and the cross-check audits every reference on the web.

- Sources: the ingestion report's `## Sources` lists the verified sources as `[S<n>]` entries with bibliographic
  data and `Excerpt` lines. Cite only those works, by their bibliographic record (authors, title, venue, year, DOI or
  URL) in the deliverable's own reference list, never by `[S<n>]` or a wikilink. Attribute a fact, value or formula
  to a work only if its `Excerpt` lines show the work contains it; present anything else as your own derivation or
  an assumption, and name the claims without a verified source in your report (the execution note's
  `## Known Limitations`, or a fix report's `summary`).
- Never cite, link or name the pipeline's notes in a deliverable (`[[01-ingestion]]`, `[[02-strategy]]`,
  `[[03-execution]]`, `[[04-crosscheck]]`, "the ingestion report", source ids such as `[S1]`, any `runs/...` path),
  and never give an AI model as a source: they are not sources and are not shipped. The source audit rates a cited
  internal note or a work it cannot find as a critical issue, and an attribution the work does not support as a
  major one.
- Write every deliverable as a finished work, with no remarks about revisions, reviews, critiques, cross-checks, fix
  rounds or acceptance criteria ("the previous revision", "as proposed in review"). Put what changed in your report
  instead (`## Implementation Notes`, or a fix report's `summary`).
- Every number in prose and tables must come from the data: generate interpretive text from the results, or assert
  each quoted number against them in a test, and make sure the prose agrees with what the data shows. Never
  hardcode a result in a template, and make a renderer fail on a missing or null value instead of printing `n/a`,
  `None` or `nan`.
- Markdown: no unfilled placeholders (double-brace fields, TODO, TBD, XXX); balanced `$` and `$$`; in a table row
  write `\\lvert x \\rvert` or `\\mid` instead of `|` inside math, because every `|` splits the cell; every embed
  and relative link resolves inside the deliverables."""
"""Prompt block of rules for everything Claude ships: both execution prompts and the cross-check's fix pass
(``{{deliverable_rules}}``)."""

LINT_SECTION = "Lint"
"""Extra H2 appended to the execution note when the deliverables have critical or major lint findings."""

MAX_LINT_ITEMS = 40
"""Findings listed in ``## Lint``; the rest are counted."""

FINAL_VERDICT_SECTIONS: tuple[str, ...] = ("Acceptance", "Clean-room Reproduction")
"""The 05-final sections an extra round's execution reads (``maf.stages.final.ACCEPTANCE_SECTION`` and
``CLEANROOM_SECTION``; final imports this module, so the titles are repeated here)."""

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
            if ctx.index.unmet_criteria:  # an extra round after final: its verdicts are still in the index
                final_name = _vault.note_name(HandoffKind.FINAL)
                final = ctx.read(final_name) if ctx.vault.has_note(ctx.run_id, final_name) else None
                if final is not None:
                    consumed.append(final_name)
                fix_context += "\n\n" + render_unmet_criteria(ctx.index.unmet_criteria, final_name, final)
        inputs = render_inputs(notes)
        if mode == "prose":
            handoff = self._run_prose(ctx, consumed, inputs, fix_context)
        else:
            handoff = self._run_code(ctx, mode, consumed, inputs, fix_context)
        handoff = with_lint(handoff, *safe_lint(ctx))
        return StageOutput(notes=[NoteOut(_vault.note_name(HandoffKind.EXECUTION, ctx.round), handoff)])

    def _run_code(
        self, ctx: StageContext, mode: ExecutionMode, consumed: list[str], inputs: str, fix_context: str
    ) -> Handoff:
        ensure_sandbox(ctx)
        kernel = provision_freertos(ctx.settings.freertos_path, ctx.paths.workspace)
        prompt = render_prompt(
            "execution_code",
            round=str(ctx.round),
            mode_guidance=MODE_GUIDANCE[mode],
            deliverable_rules=DELIVERABLE_RULES,
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
            deliverable_rules=DELIVERABLE_RULES,
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


def ensure_sandbox(ctx: StageContext) -> None:
    """Run the Claude Code provider's sandbox preflight through ``ctx.call`` (metered: ``ctx.stage``, purpose
    ``preflight``, never retried) unless that provider already passed it. Providers are built once per run in a
    process, so this is once per run; a resumed process checks again. Crosscheck calls it too, which matters only
    for a process resumed straight into crosscheck.

    ``SandboxUnavailable`` (and any other preflight failure) propagates, so the run stops before paying for a
    session whose Bash calls would all fail. The budget is ``claude_code_preflight_budget_usd``; None lets the
    provider scale it with the model. Providers without a ``preflight`` method (test fakes) are skipped."""
    provider = ctx.providers.for_role("claude_code")
    preflight = getattr(provider, "preflight", None)
    if preflight is None or getattr(provider, "sandbox_verified", False):
        return
    preflight(
        ctx.model("claude_code"),
        ctx.settings.claude_code_preflight_budget_usd,
        call=lambda request: ctx.call("claude_code", request, purpose=PREFLIGHT_PURPOSE),
    )


def lint_workspace(ctx: StageContext) -> list[LintIssue]:
    """``maf.lint.lint_deliverables`` over the deliverable tree: the workspace minus ``export_excludes(settings)``, the
    same tree the export ships and the cross-check lints."""
    return _lint.lint_deliverables(ctx.paths.workspace, export_excludes(ctx.settings))


def safe_lint(ctx: StageContext) -> tuple[list[LintIssue], str]:
    """``(lint_workspace(ctx), "")``, or ``([], "<error>")`` when the linter itself fails: logged, never raised, so a
    linter bug cannot discard the paid execution pass (and make every resume pay for it again)."""
    try:
        return lint_workspace(ctx), ""
    except Exception as exc:  # noqa: BLE001 - any linter bug
        log.exception("execution: maf lint failed on %s", ctx.paths.workspace)
        return [], f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}"


def without_lint(note: Handoff) -> tuple[str, ...]:
    """Section titles of an execution note except ``## Lint``, for ``render_inputs``: later stages see the cross-check's
    fresh lint instead of this pass's (possibly stale) findings, so each finding reaches them once."""
    return tuple(title for title in note.sections if title != LINT_SECTION)


def with_lint(handoff: Handoff, issues: list[LintIssue], error: str = "") -> Handoff:
    """Log every finding and list the critical and major ones in a ``## Lint`` section (at most ``MAX_LINT_ITEMS``).
    Without such findings the note is returned unchanged. ``error`` (the linter failed, ``safe_lint``) becomes a
    one-line ``## Lint`` instead."""
    if error:
        text = f"maf lint failed after this pass ({error}); the cross-check lints again."
        return with_sections(handoff, {LINT_SECTION: text})
    found = _lint.blocking(issues)
    if issues:
        log.warning(
            "execution: lint found %d critical/major and %d minor issue(s) in the workspace deliverables",
            len(found), len(issues) - len(found),
        )
        for issue in issues:
            log.info("lint: %s", issue)
    if not found:
        return handoff
    lines = [f"- {issue}" for issue in found[:MAX_LINT_ITEMS]]
    if len(found) > MAX_LINT_ITEMS:
        lines.append(f"- ... and {len(found) - MAX_LINT_ITEMS} more")
    intro = (
        "maf lint found these critical and major problems in the workspace's Markdown after this pass. The "
        "cross-check lints again and raises what remains as `LINT` issues (critics are not shown this list)."
    )
    return with_sections(handoff, {LINT_SECTION: intro + "\n\n" + "\n".join(lines)})


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
    """Prompt block carrying the previous cross-check's unresolved critical issues and rulings, and its source audit
    (the auditor's evidence and corrections for SRC issues, quoted) when there is one."""
    return (
        f"## Fix context from the previous cross-check ([[{name}]])\n\n"
        "Resolving the critical issues the previous pass left unresolved (below; `None.` means there are none) is "
        "the first priority of this pass; the rulings show what the adjudicator required, and the source audit (when "
        "shown) gives the evidence behind each SRC issue.\n\n"
        + render_inputs({name: crosscheck}, {name: ("Unresolved Critical", "Rulings", "Source Audit")})
    )


def render_unmet_criteria(unmet: list[str], name: str, final: Handoff | None) -> str:
    """Prompt block for an extra round (``maf resume --extra-round``): the acceptance criteria the last final report
    found not met (``RunIndex.unmet_criteria``), with its verdicts and clean-room record when the note is on disk."""
    text = (
        f"## Acceptance criteria not met at the last final report ([[{name}]])\n\n"
        "The run ended `completed_with_issues` because these criteria were not met. Meeting them is the other first "
        "priority of this pass: the final report judges every criterion again afterwards, and `clean-room` means a "
        "fresh copy of the exported deliverables did not reproduce with the README's command.\n\n"
        + "\n".join(f"- {escape_note_tags(line)}" for line in unmet)
    )
    if final is not None:
        text += "\n\n" + render_inputs({name: final}, {name: FINAL_VERDICT_SECTIONS})
    return text


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
