"""Stage 05, Final: export the deliverables, check them, and have Claude assemble the final report.

Owner: stages.

Consumes the latest ``03-execution[-rN]`` (without its ``## Lint`` section: the cross-check linted afresh) and
``04-crosscheck[-rN]`` (plus ``02-strategy`` ``## Acceptance Criteria``). In order:

1. **Export** (``export_deliverables``). Code/mixed runs: ``deliverables/`` becomes a copy of the whole workspace tree
   (``Vault.export_workspace``), minus ``export_excludes(settings)``, empty sandbox placeholders and unsafe entries,
   swapped in as a whole, so files created in cross-check fix rounds ship too. The execution note's ``## Artifacts``
   is informational: it only orders the ``## Deliverables`` links (listed artifacts first, then every top-level
   entry). A tree over ``Settings.export_max_bytes`` stops the stage with ``ExportTooLarge``. Prose runs copy the
   listed artifacts (``copy_deliverables``); artifact trees too large for a vault (more than
   ``MAX_DELIVERABLE_FILES`` files or ``MAX_DELIVERABLE_BYTES``) stay in the workspace and are listed by path.
   Bare image embeds in exported Markdown are rewritten to vault-relative paths (``relink_image_embeds``).
2. **Checks maf owns**, each recorded as a hard acceptance criterion of its own (``MAF_GATES``), whatever the model
   writes:
   - ``lint`` (every run): ``maf.lint`` over the exported tree; any critical finding (a pipeline link, an internal
     source id) leaves it unmet, since the last fix pass may have introduced it.
   - ``source-audit`` (``settings.source_audit``, when a Markdown deliverable cites works): met only if the latest
     source audit (``maf.stages.crosscheck.AUDIT_RECORD``) covered every such document in its shipped form (same
     digest), finished, and left no reference unverified, unaudited or unaccounted for.
   - ``clean-room`` (code/mixed, ``run_cleanroom``). Python copies ``deliverables/`` into a room *outside* the
     workspace (``cleanroom_dir``: ``<workspaces>/.maf-cleanroom/<run_id>/``) with the provisioned FreeRTOS kernel
     and the user's ``inputs/`` copied in (not deliverables, but the user has them), removes stale ``REPRO_*`` files
     and backdates every file so that generated outputs are older than the sources and build descriptions they come
     from (``make`` rebuilds them). Exported source and Markdown files that name the workspace's absolute path fail
     the gate. One sandboxed Claude Code session (``ctx.call``, purpose ``cleanroom``), bound to the room with the
     workspace unreadable (``ClaudeCodeProvider.bound_to``) and Bash commands allowed ``Settings.bash_timeout_s``,
     runs the reproduction command the README (or the acceptance criteria) documents there as
     ``cd <room> && ( CMD ) > REPRO_LOG 2>&1; echo $? > REPRO_EXIT`` and reports ``CLEANROOM_SCHEMA``. Its budget is
     ``Settings.cleanroom_budget_usd``, capped so the final report's worst case stays affordable
     (``cleanroom_budget``); with less than ``CLEANROOM_MIN_BUDGET_USD`` left the session is not run. Python reads
     ``REPRO_EXIT`` and ``REPRO_LOG`` itself (regular files only, written during the session, the exit code after the
     log) and cross-checks them against the report. Anything but exit 0 in both, with no missing paths, leaves it
     unmet. A judged result is kept in the run folder (``CLEANROOM_RECORD``) under a digest of the room and prompt,
     so a resumed final with identical deliverables does not pay for the session again.
3. **Final note**. Claude (Messages) writes ``05-final`` (kind FINAL, ``to: user``) with an extra ``## Acceptance``
   section: one verdict line ``- AC-<n> [met|partial|unmet]: evidence`` per criterion of 02-strategy (read with
   ``maf.handoff.parse_acceptance_criteria``). The ``acceptance_errors`` check sends a missing or malformed
   verdict through the one repair. ``## Deliverables`` links each deliverable (wikilinks for .md, embeds for images,
   inline code paths otherwise), ``## Provenance`` has one wikilink bullet per consumed note.

Python then guarantees the rules: missing deliverable links, provenance links and unresolved issues are appended,
``## Acceptance`` is rewritten canonically (with maf's gate verdicts), ``## Clean-room Reproduction`` records the
clean room, unmet criteria are listed under ``## Limitations``, and the note's ``cost_usd`` includes the clean-room
session. If ``ctx.index.unresolved_critical > 0`` (loops exhausted) or any hard criterion (maf's gates included) is
not ``met``, the run ends ``completed_with_issues`` (soft criteria are reported, not counted): ``## Summary`` opens
with a warning callout and the note gets the ``maf/completed-with-issues`` tag. ``index_updates`` carry
``criteria_unmet``, ``unmet_criteria``, ``exported_at`` and ``export_note``. ``export_run`` repeats step 1 for
``maf export`` (no model calls).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import stat
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from maf import handoff as hf
from maf import ledger as _ledger
from maf import lint as _lint
from maf import vault as _vault
from maf.handoff import Handoff, HandoffKind
from maf.prompts import render_prompt
from maf.providers import CompletionRequest, StructuredOutputError
from maf.providers.claude_code import ClaudeCodeBudgetExhausted
from maf.stages.base import (
    DEFAULT_EFFORT,
    WORKSPACE_META_DIR,
    NoteOut,
    StageContext,
    StageOutput,
    default_output_tokens,
    escape_note_tags,
    export_excludes,
    generate_handoff,
    one_line,
    render_inputs,
    role_system,
    with_sections,
    write_workspace_file,
)
from maf.handoff import Criterion, parse_acceptance_criteria
from maf.stages.crosscheck import (
    AUDIT_RECORD,
    markdown_documents,
    read_audit_record,
    reference_estimate,
    text_digest,
)
from maf.stages.execution import (
    FREERTOS_DIR,
    IMAGE_SUFFIXES,
    ensure_sandbox,
    parse_artifact_paths,
    resolve_in_workspace,
    without_lint,
)
from maf.types import ExecutionMode, RunStatus, StageName
from maf.vault import ExportError, ExportResult, RunIndex, Vault

log = logging.getLogger(__name__)

MAX_DELIVERABLE_FILES = 2_000
MAX_DELIVERABLE_BYTES = 50_000_000
"""Prose runs: a listed artifact directory above either cap stays in the workspace and is listed by path."""

ISSUES_TAG = "maf/completed-with-issues"
"""Extra tag on a 05-final written for a ``completed_with_issues`` run (searchable in Obsidian)."""

CODE_MODES: frozenset[ExecutionMode] = frozenset({"code", "mixed"})
"""Modes whose deliverables are the workspace tree and get the clean-room gate."""

ACCEPTANCE_SECTION = "Acceptance"
"""Extra 05-final section: one verdict per acceptance criterion (``- AC-<n> [met|partial|unmet]: evidence``)."""

CLEANROOM_SECTION = "Clean-room Reproduction"
"""Extra 05-final section (code/mixed) that Python writes from ``CleanroomResult.details``."""

Verdict = Literal["met", "partial", "unmet"]
VERDICTS: tuple[Verdict, ...] = ("met", "partial", "unmet")

CLEANROOM_ROOT = ".maf-cleanroom"
"""Folder of the workspaces root that holds the clean rooms (``cleanroom_dir``): beside the workspaces, never inside
one, so no relative or absolute path from the room leads back into the workspace it came from."""
REPRO_EXIT = "REPRO_EXIT"
"""Clean-room file the reproduction command's exit code goes to (``; echo $? > REPRO_EXIT``); Python reads it."""
REPRO_LOG = "REPRO_LOG"
"""Clean-room file holding the reproduction command's stdout and stderr."""
CLEANROOM_PROVIDED: tuple[str, ...] = (FREERTOS_DIR, "inputs")
"""Workspace directories copied into the clean room when the export lacks them: provisioned or user-supplied, so a
user reproducing the deliverables has them too."""
CLEANROOM_PURPOSE = "cleanroom"
"""Ledger ``purpose`` of the clean-room session (stage ``final``)."""
CLEANROOM_EFFORT = "medium"
CLEANROOM_ID = "clean-room"
CLEANROOM_CRITERION = (
    "Clean-room reproduction: the reproduction command the deliverables document succeeds in a fresh copy of the "
    "exported deliverables."
)
CLEANROOM_MIN_BUDGET_USD = 0.25
"""Below this, after holding back the final report's worst case, the clean-room session is not run (unmet gate)."""
CLEANROOM_RECORD = "cleanroom.json"
"""Run-folder file (JSON, not a note) keeping the last judged clean-room result and the digest it belongs to."""
STALE_AGE_S = 86_400
"""Every file copied into the clean room is dated this long ago; sources and build descriptions (``is_source``)
an hour later, so every generated file looks older than its inputs and is rebuilt."""
SOURCE_SUFFIXES = frozenset({
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".s", ".asm", ".ld", ".lds", ".inc",
    ".py", ".pyx", ".pxd", ".sh", ".bash", ".zsh", ".mk", ".mak", ".cmake", ".in", ".ac", ".am", ".m4",
    ".rs", ".go", ".java", ".kt", ".scala", ".js", ".mjs", ".ts", ".jl", ".r", ".m", ".f", ".f90", ".lua", ".pl",
    ".tex", ".bib", ".sty", ".cls", ".j2", ".jinja", ".jinja2", ".tmpl", ".dot", ".gp", ".plt",
})
"""Suffixes (lower case) of files a build reads rather than writes: see ``is_source``. Data formats (``.csv``,
``.json``, ``.yaml``, ``.md``, images, logs) are left out: they are as often generated as read, and a generated file
must look stale; a rule usually has the script that reads them as a prerequisite too."""
SOURCE_NAMES = frozenset({
    "Makefile", "makefile", "GNUmakefile", "CMakeLists.txt", "Kconfig", "configure", "meson.build", "SConstruct",
    "Dockerfile", "requirements.txt", "setup.py", "pyproject.toml", "Cargo.toml", "go.mod", "package.json",
})
"""File names of build descriptions without a telling suffix."""
LOG_TAIL_CHARS = 3_000
EXIT_FILE_MAX_BYTES = 64
WORKSPACE_REF_MAX_BYTES = 2_000_000
"""Larger exported files are not scanned for the workspace's absolute path."""
MTIME_SLACK_S = 1.0
"""File times use a coarse clock that can trail ``time.time()``: a result file counts as written during the session
when its time is at most this much before the session started."""

LINT_ID = "lint"
LINT_CRITERION = (
    "Deliverable lint: the exported Markdown has no critical lint finding (no link to a pipeline note, no internal "
    "source id)."
)
SOURCE_AUDIT_ID = "source-audit"
SOURCE_AUDIT_CRITERION = (
    "Source audit: every reference of the shipped Markdown deliverables was verified by the web-grounded source "
    "audit, in the text as shipped."
)
MAF_GATES: tuple[str, ...] = (CLEANROOM_ID, SOURCE_AUDIT_ID, LINT_ID)
"""Acceptance criteria maf records itself (never from a verdict line the model writes)."""
LINT_FINDINGS_SHOWN = 5

CLEANROOM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["command", "exit_code", "missing_paths", "log_tail"],
    "properties": {
        "command": {"type": "string", "description": "The reproduction command exactly as run (empty if none)."},
        "exit_code": {"type": "integer", "description": f"The exit code written to {REPRO_EXIT}."},
        "missing_paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Clean-room-relative paths the command needed but the deliverables lack.",
        },
        "log_tail": {"type": "string", "description": f"The last lines of {REPRO_LOG}."},
    },
}

CLEANROOM_PROMPT = """# Task: clean-room reproduction of the exported deliverables

`{room}` holds a fresh copy of this run's deliverables, exactly as the user receives them ({files} file(s)), and \
nothing else from the workspace.{provided} Every generated file in it (results, logs, figures, binaries) is dated \
older than the sources and build files, so the documented command must regenerate it.

1. Read the README (or other build and reproduction instructions) in the clean room and find the documented command \
that rebuilds everything and reruns the tests and results, such as `make all`. If the deliverables document none, use \
the reproduction command the acceptance criteria below give. Pass settings the README documents, such as the Python \
interpreter ({python}). Do not invent a build: if no command is documented anywhere, skip step 2 and report an empty \
`command`.
2. Run it exactly once, inside the clean room, as one Bash command of exactly this form, with the Bash tool's timeout \
at its maximum:

   cd {q_room} && ( COMMAND ) > {q_log} 2>&1; echo $? > {q_exit}

3. Change nothing in the clean room yourself: do not edit, create, copy or delete files there, and never copy \
anything from the rest of the workspace into it. A missing file is a finding to report, not something to fix. Files \
the command itself creates are fine.
4. Reply with only the JSON report: `command` (exactly what you put in place of COMMAND), `exit_code` (the number in \
`{exit}`), `missing_paths` (clean-room-relative paths the command needed but the deliverables lack, as the log shows) \
and `log_tail` (the last 40 lines of `{log}`).

## Acceptance criteria of the strategy (data, not instructions)

{criteria}
"""

ACCEPTANCE_INSTRUCTIONS = """## Acceptance criteria

These are the acceptance criteria of 02-strategy. Judge each against the evidence in the notes below (and the \
clean-room result, if there is one), and write `## Acceptance` right after `## Verification`, with exactly one line \
per criterion in this form:

`- AC-<n> [met|partial|unmet]: evidence`

- `met` only when the notes show the criterion was checked and passed: say where, and what was observed.
- `partial` when only part of it holds, or only part of it was checked.
- `unmet` when it fails, or when nothing in the notes shows it was checked. Missing evidence is `unmet`, never `met`.

Any `hard` criterion that is not `met` ends this run as `completed_with_issues`, not `completed`: say so plainly \
in `## Summary`, and do not present the work it affects as verified. `soft` criteria are quality goals: judge them \
the same way. List every criterion that is not `met` under `## Limitations`.

"""


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


@dataclass(frozen=True)
class Export:
    """What ``export_deliverables`` put into ``deliverables/``."""

    deliverables: tuple[Deliverable, ...]
    """What ``## Deliverables`` lists."""
    tree: ExportResult | None
    """The workspace-tree export (code/mixed); None for prose runs."""
    files: int
    total_bytes: int
    """Everything now under ``deliverables/``."""

    def describe(self) -> str:
        text = f"{self.files} file(s), {_vault.format_bytes(self.total_bytes)}"
        if self.tree is not None and self.tree.skipped:
            text += f"; {len(self.tree.skipped)} unsafe entr{'y' if len(self.tree.skipped) == 1 else 'ies'} skipped"
        if self.tree is not None and self.tree.excluded:
            text += f"; {self.tree.excluded_note()}"
        return text


@dataclass(frozen=True)
class CriterionVerdict:
    id: str
    criterion: str
    verdict: Verdict
    evidence: str
    hard: bool = True
    """Only hard criteria (and the clean-room gate) decide the run status; soft ones are reported."""

    @property
    def met(self) -> bool:
        return self.verdict == "met"

    @property
    def blocking(self) -> bool:
        """Counts toward ``criteria_unmet``: a hard criterion that is not ``met``."""
        return self.hard and not self.met

    @property
    def line(self) -> str:
        """The canonical ``## Acceptance`` item."""
        soft = "" if self.hard else " (soft criterion)"
        return f"- {self.id} [{self.verdict}]: {self.criterion}{soft}\n  - Evidence: {self.evidence}"

    @property
    def summary(self) -> str:
        """One line for run.md (``RunIndex.unmet_criteria``) and ``## Limitations``."""
        return f"{self.id} [{self.verdict}]: {one_line(self.criterion, 200)}"


@dataclass(frozen=True)
class CleanroomResult:
    """The clean-room gate's outcome, as Python verified it."""

    command: str = ""
    exit_code: int | None = None
    """What ``REPRO_EXIT`` holds (None when it is missing or not an integer)."""
    reported_exit_code: int | None = None
    missing_paths: tuple[str, ...] = ()
    log_tail: str = ""
    log_reported: bool = False
    """``log_tail`` is the session's own report (``REPRO_LOG`` was not written), not the file."""
    problem: str = ""
    """Why the reproduction does not count; empty when it passed."""
    cost_usd: float = 0.0
    workspace_refs: tuple[str, ...] = ()
    """Exported files naming the workspace's absolute path (``workspace_references``)."""
    reused: bool = False
    """Taken from ``CLEANROOM_RECORD`` (identical room and prompt): no session was run this time."""

    @property
    def passed(self) -> bool:
        return not self.problem

    @property
    def verdict(self) -> CriterionVerdict:
        if self.passed:
            evidence = (
                f"{_inline_code(self.command)} exited 0 in a fresh copy of the exported deliverables "
                f"({REPRO_EXIT} agrees); see `## {CLEANROOM_SECTION}`."
            )
        else:
            evidence = f"{self.problem}; see `## {CLEANROOM_SECTION}`."
        return CriterionVerdict(CLEANROOM_ID, CLEANROOM_CRITERION, "met" if self.passed else "unmet", evidence)

    def details(self) -> str:
        """Markdown for ``## Clean-room Reproduction`` (and the final prompt)."""
        recorded = "not recorded" if self.exit_code is None else str(self.exit_code)
        reported = "" if self.reported_exit_code is None else f"; the session reported {self.reported_exit_code}"
        lines = [
            f"- Result: {'passed' if self.passed else 'failed: ' + self.problem}",
            f"- Command: {_inline_code(self.command) if self.command else '(none)'}, run in a fresh copy of "
            "`deliverables/` outside the workspace, with generated files dated older than their sources",
            f"- Exit code: {recorded} (`{REPRO_EXIT}`){reported}",
            f"- Missing paths: {', '.join(_inline_code(p) for p in self.missing_paths) or 'none reported'}",
        ]
        if self.workspace_refs:
            refs = ", ".join(_inline_code(p) for p in self.workspace_refs)
            lines.append(f"- Workspace paths in the deliverables: {refs}")
        if self.reused:
            lines.append("- Reused: the result of an earlier attempt of this stage on identical deliverables")
        if self.log_tail.strip():
            fence = _fence_for(self.log_tail)
            source = f"`{REPRO_LOG}`"
            if self.log_reported:
                source = f"reported by the session; `{REPRO_LOG}` was not written"
            lines += ["", f"Log tail ({source}):", "", f"{fence}text", self.log_tail.strip("\n"), fence]
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["missing_paths"], data["workspace_refs"] = list(self.missing_paths), list(self.workspace_refs)
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CleanroomResult:
        fields = {f.name for f in dataclasses.fields(cls)}
        values = {k: v for k, v in data.items() if k in fields}
        values["missing_paths"] = tuple(values.get("missing_paths", ()))
        values["workspace_refs"] = tuple(values.get("workspace_refs", ()))
        return cls(**values)


class FinalBackend:
    name: StageName = "final"

    def __init__(
        self, *, export_excludes: Sequence[str] | None = None, cleanroom_budget_usd: float | None = None
    ) -> None:
        """``export_excludes`` replaces the exclude patterns of a code/mixed export (default
        ``export_excludes(settings)``); ``cleanroom_budget_usd`` is the clean-room session's cap (default
        ``Settings.cleanroom_budget_usd``)."""
        self.export_excludes = None if export_excludes is None else tuple(export_excludes)
        self.cleanroom_budget_usd = cleanroom_budget_usd

    def run_stage(self, ctx: StageContext) -> StageOutput:
        strategy_name = _vault.note_name(HandoffKind.STRATEGY)
        exec_name = ctx.latest_note_name(HandoffKind.EXECUTION)
        cross_name = ctx.latest_note_name(HandoffKind.CROSSCHECK)
        consumed = [strategy_name, exec_name, cross_name]
        notes = {name: ctx.read(name) for name in consumed}
        mode = ctx.index.mode
        criteria = parse_acceptance_criteria(notes[strategy_name])

        excludes = self.export_excludes if self.export_excludes is not None else export_excludes(ctx.settings)
        export = export_deliverables(
            ctx.vault, ctx.run_id, mode, notes[exec_name], excludes=excludes, max_bytes=ctx.settings.export_max_bytes
        )
        gates = [lint_gate(ctx.paths.deliverables)]
        if ctx.settings.source_audit and (audit := source_audit_gate(ctx, excludes)) is not None:
            gates.insert(0, audit)
        open_critical = ctx.index.unresolved_critical
        unresolved = unresolved_lines(notes[cross_name]) if open_critical > 0 else []

        def final_prompt(cleanroom: CleanroomResult | None) -> str:
            return render_prompt(
                "final",
                brief=ctx.index.brief,
                deliverables="\n".join(d.bullet for d in export.deliverables) or "None.",
                criteria=criteria_block(criteria),
                checks=checks_block(cleanroom, gates),
                unresolved=_unresolved_block(open_critical, cross_name, unresolved),
                inputs=render_inputs(
                    notes, {strategy_name: (hf.ACCEPTANCE_CRITERIA,), exec_name: without_lint(notes[exec_name])}
                ),
                format_spec=hf.format_spec(HandoffKind.FINAL),
            )

        cleanroom = None
        if mode in CODE_MODES:
            budget = self.cleanroom_budget_usd or ctx.settings.cleanroom_budget_usd
            reserve = final_worst_case(ctx, final_prompt(WORST_CASE_CLEANROOM))
            criteria_text = "\n".join(c.line for c in criteria)
            cleanroom = run_cleanroom(ctx, export, criteria_text, budget_usd=budget, reserve_usd=reserve)
        final = generate_handoff(
            ctx,
            "claude",
            HandoffKind.FINAL,
            system=role_system("claude", ctx.settings),
            prompt=final_prompt(cleanroom),
            to="user",
            inputs=consumed,
            purpose="final",
            check=lambda handoff: acceptance_errors(handoff, criteria),
        )
        verdicts = acceptance_verdicts(final, criteria) + ([cleanroom.verdict] if cleanroom else []) + gates
        unmet = [v for v in verdicts if v.blocking]
        final = enforce_final_rules(
            final, list(export.deliverables), consumed, unresolved, verdicts=verdicts, cleanroom=cleanroom
        )
        if cleanroom is not None and cleanroom.cost_usd > 0:
            cost = final.meta.cost_usd + cleanroom.cost_usd
            final = final.model_copy(update={"meta": final.meta.model_copy(update={"cost_usd": cost})})
        if open_critical > 0 or unmet:
            final = mark_completed_with_issues(final, open_critical, cross_name, unmet)
        return StageOutput(
            notes=[NoteOut(_vault.note_name(HandoffKind.FINAL), final)],
            index_updates={
                "criteria_unmet": len(unmet),
                "unmet_criteria": [v.summary for v in unmet],
                "exported_at": ctx.now,
                "export_note": f"final: {export.describe()}",
            },
            deliverables=[d.source for d in export.deliverables if d.copied],
        )


# ---------------------------------------------------------------------------------------------- export


def export_deliverables(
    vault: Vault,
    run_id: str,
    mode: ExecutionMode | None,
    execution: Handoff | None,
    *,
    excludes: Sequence[str] | None = None,
    max_bytes: int | None = None,
) -> Export:
    """Write ``deliverables/``: the workspace tree for code/mixed runs (``Vault.export_workspace`` with ``excludes``
    and ``max_bytes``; ``ExportTooLarge`` above the cap), otherwise the artifacts listed in ``execution``. Then
    rewrite bare image embeds in the exported Markdown files. No model calls."""
    paths = vault.paths(run_id)
    tree: ExportResult | None = None
    if mode in CODE_MODES:
        tree = vault.export_workspace(run_id, excludes=excludes, max_bytes=max_bytes)
        deliverables = tree_deliverables(vault, run_id, tree, execution)
        documents = [rel for rel in tree.files if rel.lower().endswith(".md")]
    else:
        if execution is None:
            raise ExportError(f"{run_id} has no execution note, so there is no prose deliverable to export")
        deliverables = copy_deliverables(vault, run_id, execution)
        documents = [d.rel for d in deliverables if d.copied and not d.is_dir and d.rel.lower().endswith(".md")]
    relink_image_embeds(vault, run_id, documents)
    files = [p for p in paths.deliverables.rglob("*") if p.is_file()] if paths.deliverables.is_dir() else []
    return Export(tuple(deliverables), tree, len(files), sum(p.stat().st_size for p in files))


def export_run(
    vault: Vault, index: RunIndex, *, excludes: Sequence[str] | None = None, max_bytes: int | None = None
) -> Export:
    """``maf export``: repeat the final stage's export for an existing run from its workspace (no model calls, no
    clean-room gate). ``ExportError`` when the run has no mode yet, or a prose run has no execution note."""
    if index.mode is None:
        raise ExportError(
            f"{index.run_id} has no execution mode yet (ingestion has not run), so there is nothing to export"
        )
    name = latest_recorded(index, HandoffKind.EXECUTION)
    execution = vault.read_handoff(index.run_id, name) if name else None
    return export_deliverables(vault, index.run_id, index.mode, execution, excludes=excludes, max_bytes=max_bytes)


def latest_recorded(index: RunIndex, kind: HandoffKind) -> str | None:
    """Name of the newest round of ``kind`` in ``index.handoffs`` (None when there is none)."""
    recorded = set(index.handoffs)
    return next((name for rnd in range(index.round, 0, -1) if (name := _vault.note_name(kind, rnd)) in recorded), None)


def tree_deliverables(vault: Vault, run_id: str, tree: ExportResult, execution: Handoff | None) -> list[Deliverable]:
    """``## Deliverables`` entries of a workspace-tree export: the exported artifacts the execution note lists (in its
    order), then every other top-level entry of the tree."""
    paths = vault.paths(run_id)
    workspace = paths.workspace.resolve()
    files = set(tree.files)
    dirs = {str(parent) for rel in tree.files for parent in PurePosixPath(rel).parents if str(parent) != "."}

    def entry(rel: str) -> Deliverable:
        vault_path = _vault_path(vault, paths.deliverables / rel)
        return Deliverable(rel, workspace / rel, copied=True, is_dir=rel in dirs, vault_path=vault_path)

    out: list[Deliverable] = []
    listed = parse_artifact_paths(execution.section("Artifacts")) if execution is not None else []
    for rel in listed:
        try:
            clean = resolve_in_workspace(workspace, rel).relative_to(workspace).as_posix()
        except ValueError:
            continue
        if (clean in files or clean in dirs) and all(d.rel != clean for d in out):
            out.append(entry(clean))
    for top in sorted({rel.split("/", 1)[0] for rel in tree.files}):
        if all(d.rel != top for d in out):
            out.append(entry(top))
    return out


def copy_deliverables(vault: Vault, run_id: str, execution: Handoff) -> list[Deliverable]:
    """Prose runs: copy every existing artifact of the latest execution into ``deliverables/`` under its
    workspace-relative path (re-running overwrites the same targets). Missing artifacts are skipped; oversized trees
    are not copied."""
    workspace = vault.paths(run_id).workspace
    out: list[Deliverable] = []
    for rel in parse_artifact_paths(execution.section("Artifacts")):
        try:
            source = resolve_in_workspace(workspace, rel)
        except ValueError as exc:
            log.warning("skipping deliverable %r: %s", rel, exc)
            continue
        if not source.exists():
            log.warning("skipping deliverable %r: not found in the workspace", rel)
            continue
        clean = source.relative_to(workspace.resolve()).as_posix()
        if clean == "." or clean.split("/")[0] == WORKSPACE_META_DIR:
            continue
        if source.is_dir() and _too_large(source):
            out.append(Deliverable(clean, source, copied=False, is_dir=True))
            continue
        dst = vault.copy_deliverable(run_id, source, clean)
        out.append(Deliverable(clean, source, copied=True, is_dir=source.is_dir(), vault_path=_vault_path(vault, dst)))
    return out


def _vault_path(vault: Vault, path: Path) -> str:
    return path.resolve().relative_to(vault.root.resolve()).as_posix()


_EMBED = re.compile(r"!\[\[(?P<target>[^\]|/\n]+?)(?P<alias>\|[^\]\n]*)?\]\]")


def relink_image_embeds(vault: Vault, run_id: str, documents: Sequence[str]) -> None:
    """Rewrite bare image embeds (``![[plot.png]]``) in the exported Markdown ``documents`` (paths under
    ``deliverables/``) to the vault-relative path of that image under this run's ``deliverables/``, so they cannot
    resolve to another run's file. Names that match several exported images, or none, are left alone."""
    if not documents:
        return
    root = vault.root.resolve()
    deliverables = vault.paths(run_id).deliverables
    by_name: dict[str, list[str]] = {}
    for image in sorted(deliverables.rglob("*")):
        if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
            by_name.setdefault(image.name, []).append(image.resolve().relative_to(root).as_posix())

    def replace(match: re.Match[str]) -> str:
        found = by_name.get(match.group("target").strip(), [])
        return f"![[{found[0]}{match.group('alias') or ''}]]" if len(found) == 1 else match.group(0)

    for rel in documents:
        path = deliverables / rel
        if path.is_symlink() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # never rewrite a file this lossily
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


# ---------------------------------------------------------------------------------------------- acceptance


_LEVEL_TAG = r"(?:\[(?:hard|soft)\]\s*)?"
_VERDICT_ITEM = re.compile(
    rf"^[-*+]\s+\**(?P<id>AC-\d+|{'|'.join(MAF_GATES)})\**\s*{_LEVEL_TAG}\[(?P<verdict>[^\]\n]*)\]\s*{_LEVEL_TAG}"
    r":?\s*(?P<evidence>.*)$",
    re.IGNORECASE,
)
"""A verdict line, ``- AC-2 [unmet]: evidence``; a copied ``[hard]``/``[soft]`` tag before or after the verdict is
tolerated."""
_NESTED_MARKER = re.compile(r"^(?:[-*+]|\d{1,3}[.)])\s+")
_BULLET = re.compile(r"^[-*+]\s")


def _acceptance_items(text: str) -> tuple[list[tuple[str, str, str]], list[str]]:
    """``([(id, verdict, evidence)], errors)`` for an ``## Acceptance`` section. Indented and lazy continuation lines
    fold into the item's evidence; other top-level prose is ignored; a top-level bullet that is not a verdict line is
    an error."""
    items: list[tuple[str, str, list[str]]] = []
    errors: list[str] = []
    current: list[str] | None = None  # evidence lines of the open item
    lazy = False  # no blank line since the open item's last line: an unindented prose line continues it
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            lazy = False
            continue
        bullet = _BULLET.match(stripped) is not None
        if current is not None and (line[:1] in (" ", "\t") or (lazy and not bullet)):
            current.append(_NESTED_MARKER.sub("", stripped))
            lazy = True
            continue
        current, lazy = None, False
        if not bullet:
            continue  # top-level prose around the verdicts
        match = _VERDICT_ITEM.match(stripped)
        if not match:
            errors.append(
                f"## {ACCEPTANCE_SECTION}: not a verdict line (expected `- AC-<n> [met|partial|unmet]: evidence`): "
                f"{one_line(stripped, 80)!r}"
            )
            continue
        ident = match.group("id")
        ident = ident.lower() if ident.lower() in MAF_GATES else ident.upper()
        current, lazy = [match.group("evidence").strip()], True
        items.append((ident, match.group("verdict").strip().lower(), current))
    return [(i, v, " ".join(" ".join(e).split())) for i, v, e in items], errors


def acceptance_errors(final: Handoff, criteria: Sequence[Criterion]) -> list[str]:
    """``generate_handoff`` check: ``## Acceptance`` must give exactly one well-formed verdict with evidence for every
    criterion (nothing is required when there are none). A line for one of ``MAF_GATES`` is ignored: maf records
    those."""
    if not criteria:
        return []
    ids = ", ".join(c.id for c in criteria)
    if ACCEPTANCE_SECTION not in final.sections:
        return [
            f"missing section '## {ACCEPTANCE_SECTION}' (after '## Verification'): one line per acceptance criterion "
            f"({ids}), each `- AC-<n> [met|partial|unmet]: evidence`"
        ]
    items, errors = _acceptance_items(final.sections[ACCEPTANCE_SECTION])
    known = {c.id for c in criteria}
    seen: set[str] = set()
    for ident, verdict, evidence in items:
        if ident in MAF_GATES:
            continue
        if ident not in known:
            errors.append(f"## {ACCEPTANCE_SECTION}: {ident} is not one of the criteria ({ids})")
            continue
        if ident in seen:
            errors.append(f"## {ACCEPTANCE_SECTION}: {ident} has more than one verdict")
        seen.add(ident)
        if verdict not in VERDICTS:
            errors.append(f"## {ACCEPTANCE_SECTION}: {ident} [{verdict}] must be [met], [partial] or [unmet]")
        if not evidence:
            errors.append(f"## {ACCEPTANCE_SECTION}: {ident} needs its evidence after the colon")
    for criterion in criteria:
        if criterion.id not in seen:
            errors.append(f"## {ACCEPTANCE_SECTION}: no verdict for {criterion.id} ({one_line(criterion.text, 120)})")
    return errors


def acceptance_verdicts(final: Handoff, criteria: Sequence[Criterion]) -> list[CriterionVerdict]:
    """One verdict per criterion from ``## Acceptance`` (the first line for an id wins). A criterion without a
    well-formed verdict (only possible when the check was bypassed) counts as ``unmet``."""
    items, _ = _acceptance_items(final.sections.get(ACCEPTANCE_SECTION, ""))
    found: dict[str, tuple[str, str]] = {}
    for ident, verdict, evidence in items:
        found.setdefault(ident, (verdict, evidence))
    out: list[CriterionVerdict] = []
    for criterion in criteria:
        verdict, evidence = found.get(criterion.id, ("", ""))
        if verdict in VERDICTS:
            given = cast(Verdict, verdict)
            out.append(CriterionVerdict(criterion.id, criterion.text, given, evidence or "none given", criterion.hard))
        else:
            missing = "the final report gave no valid verdict for it (recorded by maf)"
            out.append(CriterionVerdict(criterion.id, criterion.text, "unmet", missing, criterion.hard))
    return out


def criteria_block(criteria: Sequence[Criterion]) -> str:
    """The final prompt's acceptance instructions and the criteria (``Criterion.line``)."""
    if not criteria:
        return f"## Acceptance criteria\n\n02-strategy lists none, so leave out `## {ACCEPTANCE_SECTION}`."
    return ACCEPTANCE_INSTRUCTIONS + "\n".join(escape_note_tags(c.line) for c in criteria)


# ---------------------------------------------------------------------------------------------- clean room


WORST_CASE_CLEANROOM = CleanroomResult(
    command="x" * 300,
    missing_paths=("x" * 200,) * 10,
    log_tail="x" * LOG_TAIL_CHARS,
    problem="x" * 1_000,
    workspace_refs=("x" * 200,) * 5,
)
"""The largest clean-room report the final prompt can carry: what ``final_worst_case`` prices before the session."""


def final_worst_case(ctx: StageContext, prompt: str) -> float:
    """Worst-case cost of the final report's call with ``prompt`` (``generate_handoff`` sends the same request),
    which the clean-room session must leave affordable."""
    request = CompletionRequest.simple(
        ctx.model("claude"),
        prompt,
        system=role_system("claude", ctx.settings),
        max_output_tokens=default_output_tokens(ctx.settings, "claude"),
        effort=DEFAULT_EFFORT["claude"],
    )
    return ctx.providers.for_role("claude").worst_case_cost(request)


def cleanroom_budget(remaining_usd: float, reserve_usd: float, headroom_usd: float, cap_usd: float) -> float:
    """Pure: the clean-room session's ``--max-budget-usd``: ``cap_usd``, but no more than what the run has left after
    the final report's worst case (``reserve_usd``) and the session's own one-turn overshoot (``headroom_usd``, the
    ledger's ``turn_headroom``). Without this, the session could spend what the final call needs, the run would end
    ``budget_exceeded`` without a report, and a resume would pay for the session again."""
    return max(0.0, min(cap_usd, remaining_usd - reserve_usd - headroom_usd))


def run_cleanroom(
    ctx: StageContext, export: Export, criteria_text: str, *, budget_usd: float, reserve_usd: float = 0.0
) -> CleanroomResult:
    """The clean-room gate (code/mixed only; see the module docstring). ``reserve_usd`` is held back for the final
    report (``final_worst_case``). ``SandboxUnavailable`` and other provider failures propagate; a report that does not
    parse, a session that runs out of its budget, or too little budget to start one is a failed gate."""
    if export.files == 0:
        return CleanroomResult(problem="the export is empty, so there is nothing to reproduce")
    room, provided = prepare_cleanroom(ctx.paths.workspace, ctx.paths.deliverables)
    refs = workspace_references(ctx.paths.deliverables, export.tree.files if export.tree else (), ctx.paths.workspace)
    prompt = cleanroom_prompt(room, provided, criteria_text, export.files, str(ctx.settings.python_executable))
    write_workspace_file(ctx, f"{WORKSPACE_META_DIR}/cleanroom-prompt.md", prompt)
    request = CompletionRequest.simple(
        ctx.model("claude_code"),
        prompt,
        system=role_system("claude_code", ctx.settings),
        max_output_tokens=default_output_tokens(ctx.settings, "claude_code"),
        json_schema=CLEANROOM_SCHEMA,
        schema_name="cleanroom",
        effort=CLEANROOM_EFFORT,
        max_budget_usd=budget_usd,
    )
    record, digest = ctx.paths.root / CLEANROOM_RECORD, cleanroom_digest(room, prompt, request.model)
    if (kept := read_cleanroom_record(record, digest)) is not None:
        log.info("clean room: reusing the result recorded for identical deliverables (%s)", record)
        return dataclasses.replace(kept, reused=True)
    ensure_sandbox(ctx)  # before the budget: a resumed process pays for its preflight first
    provider = ctx.providers.for_role("claude_code")
    bind = getattr(provider, "bound_to", None)
    if callable(bind):  # test fakes have no such method: they run as they are
        provider = bind(room, deny_read=workspace_paths(ctx.paths.workspace), bash_default_is_max=True)
    remaining = ctx.ledger.remaining_usd
    budget = cleanroom_budget(remaining, reserve_usd, _ledger.turn_headroom(provider, request), budget_usd)
    if budget < CLEANROOM_MIN_BUDGET_USD:
        problem = (
            f"not run: budget (the run has ${remaining:.2f} left, and ${reserve_usd:.2f} of it is held back for the "
            f"final report, leaving less than the ${CLEANROOM_MIN_BUDGET_USD:.2f} a session needs)"
        )
        return CleanroomResult(problem=_with_refs(problem, refs), workspace_refs=refs)
    request = request.model_copy(update={"max_budget_usd": budget})
    room_ctx = dataclasses.replace(ctx, providers=dataclasses.replace(ctx.providers, claude_code=provider))
    started = time.time() - MTIME_SLACK_S
    try:
        result = room_ctx.call("claude_code", request, purpose=CLEANROOM_PURPOSE)
    except StructuredOutputError as exc:
        problem = f"the clean-room session returned no valid report ({one_line(str(exc), 200)})"
        return _unreported(room, problem, exc.cost_usd, refs)
    except ClaudeCodeBudgetExhausted as exc:
        problem = f"the clean-room session ran out of its ${budget:.2f} budget before reporting"
        return _unreported(room, problem, exc.cost_usd, refs)
    judged = judge_cleanroom(result.parsed or {}, room, cost_usd=result.cost_usd, started=started, workspace_refs=refs)
    if judged.exit_code is not None:  # the command ran to completion: the result belongs to these deliverables
        write_cleanroom_record(record, digest, judged)
    return judged


def cleanroom_dir(workspace: Path) -> Path:
    """The clean room of the run whose workspace is ``workspace``: ``<workspaces>/.maf-cleanroom/<run_id>/``."""
    return workspace.parent / CLEANROOM_ROOT / workspace.name


def workspace_paths(workspace: Path) -> tuple[str, ...]:
    """The workspace's absolute path, as given and resolved: what the clean room may not read or name."""
    return tuple(dict.fromkeys((str(workspace.absolute()), str(workspace.resolve()))))


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def prepare_cleanroom(workspace: Path, deliverables: Path) -> tuple[Path, list[str]]:
    """Recreate ``cleanroom_dir(workspace)`` as a copy of ``deliverables`` (file times not kept; symlinks copied as
    links, never followed), copy in the ``CLEANROOM_PROVIDED`` workspace directories it lacks, delete any
    ``REPRO_EXIT``/``REPRO_LOG`` it came with, and ``backdate`` every file. Returns the room and the directories
    copied in."""
    room = cleanroom_dir(workspace)
    _remove(room)
    room.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(deliverables, room, symlinks=True, copy_function=shutil.copy)
    provided: list[str] = []
    for name in CLEANROOM_PROVIDED:
        source, target = workspace / name, room / name
        if source.is_dir() and not source.is_symlink() and not (target.exists() or target.is_symlink()):
            shutil.copytree(source, target, symlinks=True, copy_function=shutil.copy)
            provided.append(name)
    for name in (REPRO_EXIT, REPRO_LOG):
        _remove(room / name)
    backdate(room, time.time())
    return room, provided


def is_source(name: str) -> bool:
    """Whether a file name looks like something a build reads, never writes: ``SOURCE_NAMES`` or ``SOURCE_SUFFIXES``."""
    return name in SOURCE_NAMES or PurePosixPath(name).suffix.lower() in SOURCE_SUFFIXES


def backdate(root: Path, now: float) -> None:
    """Date every file and directory under ``root`` ``STALE_AGE_S`` before ``now``, and sources and build descriptions
    (``is_source``) an hour later. Exported results, logs, figures and binaries then look older than what they are
    built from, so ``make`` and its kin rebuild them instead of finding them up to date (an export keeps the
    workspace's times, in which outputs are newer than their sources)."""
    stale = now - STALE_AGE_S
    for dirpath, _dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            when = stale + 3_600 if is_source(name) else stale
            os.utime(os.path.join(dirpath, name), (when, when), follow_symlinks=False)
        os.utime(dirpath, (stale, stale))


def workspace_references(deliverables: Path, files: Sequence[str], workspace: Path) -> tuple[str, ...]:
    """Exported sources, build files and Markdown documents (``is_source`` or ``.md``; logs and data are records and
    may quote build paths) that contain the workspace's absolute path. It does not exist in a copy of the
    deliverables, so a command or link that uses it works only in the workspace."""
    needles = [path.encode() for path in workspace_paths(workspace)]
    found: list[str] = []
    for rel in files:
        name = PurePosixPath(rel).name
        if not (is_source(name) or name.lower().endswith(".md")):
            continue
        path = deliverables / rel
        try:
            if path.is_symlink() or path.stat().st_size > WORKSPACE_REF_MAX_BYTES:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        if any(needle in data for needle in needles):
            found.append(rel)
    return tuple(found)


def _with_refs(problem: str, refs: Sequence[str]) -> str:
    if not refs:
        return problem
    shown = ", ".join(_inline_code(r) for r in refs[:5]) + (f" and {len(refs) - 5} more" if len(refs) > 5 else "")
    gap = f"the deliverables name the workspace's absolute path ({shown}), which does not exist in a copy"
    return f"{problem}; {gap}" if problem else gap


def cleanroom_digest(room: Path, prompt: str, model: str) -> str:
    """SHA-256 over the model, the session prompt and every file of the prepared room (paths and contents, link
    targets for symlinks; times left out): equal digests mean the session would get identical inputs."""
    digest = hashlib.sha256(f"{model}\0{prompt}".encode())
    for dirpath, dirnames, filenames in os.walk(room):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            digest.update(b"\0" + path.relative_to(room).as_posix().encode() + b"\0")
            if path.is_symlink():
                digest.update(b"-> " + os.readlink(path).encode())
                continue
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def read_cleanroom_record(path: Path, digest: str) -> CleanroomResult | None:
    """The result ``CLEANROOM_RECORD`` keeps for ``digest``; None when there is none (or it is unreadable)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CleanroomResult.from_json(data["result"]) if data.get("digest") == digest else None
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def write_cleanroom_record(path: Path, digest: str, result: CleanroomResult) -> None:
    _vault.atomic_write_text(path, json.dumps({"digest": digest, "result": result.to_json()}, indent=2) + "\n")


def cleanroom_prompt(room: Path, provided: Sequence[str], criteria_text: str, files: int, python: str) -> str:
    """The clean-room session's prompt (``CLEANROOM_PROMPT``)."""
    note = ""
    if provided:
        copied = " and ".join(f"`{name}/`" for name in provided)
        note = (
            f" {copied} {'is' if len(provided) == 1 else 'are'} copied in from the workspace because the user has "
            "them already (provisioned or their own input); they are not part of the deliverables."
        )
    return CLEANROOM_PROMPT.format(
        room=room,
        files=files,
        provided=note,
        python=python,
        q_room=shlex.quote(str(room)),
        q_log=shlex.quote(str(room / REPRO_LOG)),
        q_exit=shlex.quote(str(room / REPRO_EXIT)),
        exit=REPRO_EXIT,
        log=REPRO_LOG,
        criteria=criteria_text.strip() or "None given.",
    )


def judge_cleanroom(
    report: dict[str, Any],
    room: Path,
    *,
    cost_usd: float = 0.0,
    started: float | None = None,
    workspace_refs: Sequence[str] = (),
) -> CleanroomResult:
    """Cross-check the session's report against the result files: the gate passes only with a reported command, a
    ``REPRO_LOG`` and a ``REPRO_EXIT`` that are regular files written during the session (not before ``started``),
    the exit code written after the log (as the prescribed ``( CMD ) > REPRO_LOG 2>&1; echo $? > REPRO_EXIT`` does),
    an integer there equal to the reported exit code, exit code 0, no missing paths and no ``workspace_refs``. Without
    ``REPRO_LOG`` the session's own log tail is shown, labelled as reported by the session."""
    command = str(report.get("command") or "").strip()
    raw_code = report.get("exit_code")
    reported = raw_code if isinstance(raw_code, int) and not isinstance(raw_code, bool) else None
    missing = tuple(one_line(str(p), 200) for p in report.get("missing_paths") or [] if str(p).strip())
    exit_code, raw = read_repro_exit(room)
    exit_info, log_info = _stat_regular(room / REPRO_EXIT), _stat_regular(room / REPRO_LOG)
    log_tail, log_reported = read_tail(room / REPRO_LOG), False
    if log_info is None:
        log_tail, log_reported = str(report.get("log_tail") or "")[-LOG_TAIL_CHARS:], True
    problem = ""
    if not command:
        problem = "no reproduction command is documented in the deliverables or given in the acceptance criteria"
    elif raw is None or exit_info is None:
        problem = (
            f"{REPRO_EXIT} was not written, so the reproduction command was not run as instructed "
            "(or it was killed before it finished)"
        )
    elif log_info is None:
        problem = (
            f"{REPRO_LOG} was not written, so the command was not run in the prescribed form "
            f"(`( COMMAND ) > {REPRO_LOG} 2>&1; echo $? > {REPRO_EXIT}`)"
        )
    elif started is not None and min(exit_info.st_mtime, log_info.st_mtime) < started:
        problem = f"{REPRO_EXIT} or {REPRO_LOG} predates the clean-room session, so it is not this run's result"
    elif exit_info.st_mtime < log_info.st_mtime - MTIME_SLACK_S:
        problem = f"{REPRO_EXIT} is older than {REPRO_LOG}, so the prescribed command did not write that exit code"
    elif exit_code is None:
        problem = f"{REPRO_EXIT} does not hold an integer exit code ({one_line(raw, 40)!r})"
    elif reported is not None and reported != exit_code:
        problem = f"the session reported exit code {reported}, but {REPRO_EXIT} holds {exit_code}"
    elif exit_code != 0:
        problem = f"{_inline_code(command)} exited with {exit_code} in the clean room"
    if missing:
        gap = "the exported deliverables lack " + ", ".join(_inline_code(p) for p in missing)
        problem = f"{problem}; {gap}" if problem else gap
    refs = tuple(workspace_refs)
    return CleanroomResult(
        command, exit_code, reported, missing, log_tail, log_reported, _with_refs(problem, refs), cost_usd, refs
    )


def _unreported(room: Path, problem: str, cost_usd: float, refs: Sequence[str] = ()) -> CleanroomResult:
    exit_code, _ = read_repro_exit(room)
    log_tail = read_tail(room / REPRO_LOG)
    return CleanroomResult(
        exit_code=exit_code, log_tail=log_tail, problem=_with_refs(problem, refs), cost_usd=cost_usd,
        workspace_refs=tuple(refs),
    )


def _stat_regular(path: Path) -> os.stat_result | None:
    """``lstat`` of ``path`` when it is a regular file (never through a symlink); None otherwise."""
    try:
        info = path.lstat()
    except OSError:
        return None
    return info if stat.S_ISREG(info.st_mode) else None


def _read_regular(path: Path, max_bytes: int, *, tail: bool = False) -> bytes | None:
    """Up to ``max_bytes`` of the regular file ``path`` (its end with ``tail``), never through a symlink; None when it
    is missing or not a regular file."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as fh:
        info = os.fstat(fh.fileno())
        if not stat.S_ISREG(info.st_mode):
            return None
        if tail and info.st_size > max_bytes:
            fh.seek(info.st_size - max_bytes)
        return fh.read(max_bytes)


def read_repro_exit(room: Path) -> tuple[int | None, str | None]:
    """``(exit_code, raw_text)`` of ``room/REPRO_EXIT``: ``(None, None)`` when it is missing, a symlink or not a
    regular file; ``(None, raw)`` when it does not hold exactly one integer."""
    data = _read_regular(room / REPRO_EXIT, EXIT_FILE_MAX_BYTES)
    if data is None:
        return None, None
    raw = data.decode("utf-8", "replace")
    match = re.fullmatch(r"\s*(-?\d{1,5})\s*", raw)
    return (int(match.group(1)) if match else None), raw


def read_tail(path: Path, chars: int = LOG_TAIL_CHARS) -> str:
    """The last ``chars`` characters of a regular file (not followed if it is a symlink); empty when unreadable."""
    data = _read_regular(path, chars * 4, tail=True)
    return "" if data is None else data.decode("utf-8", "replace")[-chars:]


def cleanroom_block(result: CleanroomResult | None) -> str:
    """The final prompt's report of the clean-room gate (empty for prose runs)."""
    if result is None:
        return ""
    return (
        "## Clean-room reproduction (run by maf)\n\n"
        "maf copied the exported deliverables into an empty folder outside the workspace and had Claude Code run their "
        f"documented reproduction command there. maf records the result as acceptance criterion `{CLEANROOM_ID}`: do "
        "not write a verdict line for it, but report it in `## Verification`, and under `## Limitations` if it failed. "
        "Judge a strategy criterion about reproducing the deliverables from a clean copy by this result: never `met` "
        "when it failed.\n\n"
        + result.details()
    )


# ---------------------------------------------------------------------------------------------- maf's other gates


def lint_gate(deliverables: Path) -> CriterionVerdict:
    """The ``lint`` gate: ``maf.lint`` over the exported tree (all of it ships, so nothing is excluded). Unmet with any
    critical finding (the last fix pass may have introduced one), or when the linter itself fails; major findings
    are counted in the evidence."""
    try:
        findings = _lint.lint_deliverables(deliverables, exclude=())
    except Exception as exc:  # noqa: BLE001 - any linter bug: the gate is unmet, the run goes on
        log.exception("final: maf lint failed on %s", deliverables)
        evidence = f"maf lint failed on the export ({type(exc).__name__}: {one_line(str(exc), 200)})"
        return CriterionVerdict(LINT_ID, LINT_CRITERION, "unmet", evidence)
    critical = [f for f in findings if f.severity == "critical"]
    major = sum(1 for f in findings if f.severity == "major")
    remark = f"; {major} major finding(s) remain (broken tables, links or math, placeholders, meta-commentary)"
    if critical:
        shown = "; ".join(f"{_inline_code(f'{f.path}:{f.line}')} {f.rule}" for f in critical[:LINT_FINDINGS_SHOWN])
        more = f" and {len(critical) - LINT_FINDINGS_SHOWN} more" if len(critical) > LINT_FINDINGS_SHOWN else ""
        evidence = f"maf lint found {len(critical)} critical problem(s) in the exported Markdown: {shown}{more}"
        return CriterionVerdict(LINT_ID, LINT_CRITERION, "unmet", evidence + (remark if major else ""))
    evidence = "maf lint found no critical problem in the exported Markdown" + (remark if major else "")
    return CriterionVerdict(LINT_ID, LINT_CRITERION, "met", evidence)


def source_audit_gate(ctx: StageContext, excludes: Sequence[str]) -> CriterionVerdict | None:
    """The ``source-audit`` gate: None when no Markdown deliverable cites works (no ``reference_estimate`` signal and
    no audited references in its current text). Otherwise met only if ``AUDIT_RECORD`` holds, for every such
    document, a finished audit of exactly the text that ships (``text_digest``) that left no reference unverified,
    unaudited or unaccounted for. A document the last fix pass changed after its audit, or one never audited, is
    unmet: a verdict read from the pre-fix audit would be about another text."""
    record = read_audit_record(ctx.paths.root / AUDIT_RECORD)
    applicable = []
    for rel, text in markdown_documents(ctx.paths.workspace, excludes):
        digest, entry = text_digest(text), record.get(rel)
        if reference_estimate(text) or (entry is not None and entry.sha256 == digest and entry.references > 0):
            applicable.append((rel, digest, entry))
    if not applicable:
        return None
    problems: list[str] = []
    references = 0
    for rel, digest, entry in applicable:
        where = _inline_code(rel)
        if entry is None:
            problems.append(f"{where} was never audited")
        elif entry.sha256 != digest:
            problems.append(f"{where} changed after its last audit (round {entry.round})")
        elif entry.status != "done":
            problems.append(f"the audit of {where} failed")
        else:
            references += entry.references
            if entry.not_verified:
                problems.append(f"{where} has {entry.not_verified} reference(s) the audit did not verify")
            if entry.unaudited:
                problems.append(f"{where} has {entry.unaudited} reference(s) left unaudited (over the cap)")
            if entry.incomplete:
                problems.append(f"the audit of {where} is incomplete ({entry.incomplete})")
    if problems:
        evidence = "; ".join(problems) + "; see `## Source Audit` in the latest cross-check"
        return CriterionVerdict(SOURCE_AUDIT_ID, SOURCE_AUDIT_CRITERION, "unmet", evidence)
    documents = ", ".join(_inline_code(rel) for rel, _, _ in applicable)
    evidence = f"the source audit verified all {references} reference(s) of {documents} in the text as shipped"
    return CriterionVerdict(SOURCE_AUDIT_ID, SOURCE_AUDIT_CRITERION, "met", evidence)


def checks_block(cleanroom: CleanroomResult | None, gates: Sequence[CriterionVerdict]) -> str:
    """The final prompt's report of maf's own checks: the clean room (``cleanroom_block``) and the other gates."""
    blocks = [cleanroom_block(cleanroom)] if cleanroom is not None else []
    if gates:
        ids = ", ".join(f"`{g.id}`" for g in gates)
        blocks.append(
            "## Other checks run by maf\n\n"
            f"maf records these as acceptance criteria ({ids}): do not write verdict lines for them, but report them "
            "in `## Verification`, and under `## Limitations` when unmet. Judge a strategy criterion about the same "
            "thing (verified references, clean Markdown) by these results: never `met` when the check is unmet.\n\n"
            + "\n".join(f"- {g.id} [{g.verdict}]: {escape_note_tags(g.evidence)}" for g in gates)
        )
    return "\n\n".join(blocks)


def _inline_code(text: str) -> str:
    text = one_line(text, 300)
    return f"`` {text} ``" if "`" in text else f"`{text}`"


def _fence_for(text: str) -> str:
    longest = max((len(m) for m in re.findall(r"`{3,}", text)), default=0)
    return "`" * max(3, longest + 1)


# ---------------------------------------------------------------------------------------------- the note


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


def status_callout(count: int, crosscheck: str, unmet: Sequence[CriterionVerdict] = ()) -> str:
    """The warning that opens ``## Summary`` of a ``completed_with_issues`` final note; ``unmet`` are the blocking
    criteria."""
    lines = [f"> [!warning] Run status: {RunStatus.COMPLETED_WITH_ISSUES.value}"]
    if count > 0:
        lines.append(
            f"> {count} critical issue(s) remain unresolved after the cross-check loop cap (see [[{crosscheck}]]); "
            "they are listed under Limitations."
        )
    if unmet:
        n = len(unmet)
        ids = ", ".join(v.id for v in unmet)
        lines.append(
            f"> {n} acceptance criteri{'on is' if n == 1 else 'a are'} not met ({ids}); see Acceptance and Limitations."
        )
    return "\n".join(lines)


def mark_completed_with_issues(
    final: Handoff, count: int, crosscheck: str, unmet: Sequence[CriterionVerdict] = ()
) -> Handoff:
    """Open ``## Summary`` with ``status_callout`` (unless present) and add ``ISSUES_TAG`` to the frontmatter."""
    callout = status_callout(count, crosscheck, unmet)
    summary = final.section("Summary").strip()
    if callout.split("\n", 1)[0] not in summary:
        final = with_sections(final, {"Summary": f"{callout}\n\n{summary}"})
    if ISSUES_TAG not in final.meta.tags:
        final = final.model_copy(update={"meta": final.meta.model_copy(update={"tags": [*final.meta.tags, ISSUES_TAG]})})
    return final


def _mentions(text: str, ident: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(ident)}(?![\w-])", text) is not None


def enforce_final_rules(
    final: Handoff,
    deliverables: list[Deliverable],
    consumed: list[str],
    unresolved: list[str],
    *,
    verdicts: Sequence[CriterionVerdict] = (),
    cleanroom: CleanroomResult | None = None,
) -> Handoff:
    """Append whatever the model left out (deliverable links, provenance wikilinks, unresolved issues, unmet
    criteria), write ``## Acceptance`` canonically from ``verdicts`` and ``## Clean-room Reproduction`` from
    ``cleanroom``."""
    updates: dict[str, str] = {}

    def extend(section: str, lines: list[str], lead: str) -> None:
        current = updates.get(section, final.section(section)).strip()
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
    extend(
        "Limitations",
        [f"- {v.summary}" for v in verdicts if v.blocking and not _mentions(limitations, v.id)],
        "Acceptance criteria not met (added by maf):",
    )
    extend(
        "Limitations",
        [f"- {v.summary}" for v in verdicts if not v.hard and not v.met and not _mentions(limitations, v.id)],
        "Soft acceptance criteria not met (added by maf):",
    )
    if verdicts:
        updates[ACCEPTANCE_SECTION] = "\n".join(v.line for v in verdicts)
    if cleanroom is not None:
        updates[CLEANROOM_SECTION] = cleanroom.details()
    return with_sections(final, updates) if updates else final
