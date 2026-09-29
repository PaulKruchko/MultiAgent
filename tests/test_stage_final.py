"""Tests for the final stage: deliverable export, the clean-room gate, acceptance verdicts, the final note, and the
Python-enforced rules."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from maf import handoff as hf
from maf.handoff import HandoffInvalid, HandoffKind
from maf.providers import CompletionRequest, SandboxUnavailable
from maf.providers.claude_code import ClaudeCodeBudgetExhausted
from maf.stages import final as final_mod
from maf.stages.final import (
    CLEANROOM_SECTION,
    REPRO_EXIT,
    REPRO_LOG,
    CleanroomResult,
    FinalBackend,
    acceptance_errors,
    acceptance_verdicts,
    cleanroom_dir,
    export_run,
    judge_cleanroom,
    unresolved_lines,
)
from maf.stages.strategy import Criterion
from maf.vault import ExportError, ExportTooLarge
from conftest import FakeProvider
from test_stages_base import RUN_ID, SandboxedFake, StageEnv, prompt_of, stage_env  # noqa: F401

EXECUTION = """## Summary

Done.

## Artifacts

- `document.md` - the thesis
- `plots/power.png` - fusion power
- `src/` - simulation code
- `gone.txt` - never produced
- `../outside` - escapes

## Implementation Notes

n

## Verification

v

## Known Limitations

None.
"""

ACCEPTANCE = """## Acceptance

- AC-1 [met]: all suites pass (see [[03-execution-r2]])
- AC-2 [met]: text=1804 bytes
"""

FINAL_BODY = """## Summary

The thesis is complete.

## Deliverables

- [[runs/{run}/deliverables/document|document]]

## Verification

All simulations ran.

{acceptance}
## Provenance

- [[02-strategy]]

## Limitations

None.
"""

UNRESOLVED = "- [critical] GEM-2: runaway not suppressed above 500 MW"
CRITERION_2 = "`.text` < 2048 bytes on Cortex-M3 at `-Os`."


def final_body(acceptance: str = ACCEPTANCE) -> str:
    return FINAL_BODY.format(run=RUN_ID, acceptance=acceptance)


@pytest.fixture
def ready(stage_env: StageEnv, sample_bodies: dict[str, str]) -> StageEnv:
    stage_env.put("02-strategy", HandoffKind.STRATEGY, sample_bodies["strategy"], from_="chatgpt")
    stage_env.put("03-execution", HandoffKind.EXECUTION, sample_bodies["execution"])
    stage_env.put("04-crosscheck", HandoffKind.CROSSCHECK, sample_bodies["crosscheck"], from_="maf")
    stage_env.put("03-execution-r2", HandoffKind.EXECUTION, EXECUTION, round=2)
    looped = sample_bodies["crosscheck"].replace("## Unresolved Critical\n\nNone.", f"## Unresolved Critical\n\n{UNRESOLVED}")
    stage_env.put("04-crosscheck-r2", HandoffKind.CROSSCHECK, looped.replace("PASS", "LOOP"), round=2, from_="maf")
    stage_env.workspace_file("document.md", "# Thesis\n\n![[power.png]]\n\n![[power.png|400]]\n\n![[other.png]]\n")
    stage_env.workspace_file("plots/power.png", b"\x89PNG power")
    stage_env.workspace_file("src/sim.py", "print('sim')\n")
    stage_env.workspace_file("src/__pycache__/sim.cpython-312.pyc", b"\x00")
    return stage_env


# ---------------------------------------------------------------------------------------------- prose runs


def test_final_copies_deliverables_and_enforces_links(ready: StageEnv) -> None:
    ready.fakes.claude.script(final_body())
    output = FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose"))

    assert [n.name for n in output.notes] == ["05-final"]
    note = output.notes[0].handoff
    assert (note.meta.from_, note.meta.to, note.meta.stage) == ("claude", "user", HandoffKind.FINAL)
    assert note.meta.inputs == ["[[02-strategy]]", "[[03-execution-r2]]", "[[04-crosscheck-r2]]"]

    ws = ready.paths.workspace.resolve()
    assert output.deliverables == [ws / "document.md", ws / "plots" / "power.png", ws / "src"]
    deliverables = ready.paths.deliverables
    assert (deliverables / "plots" / "power.png").is_file()
    assert (deliverables / "src" / "sim.py").is_file()
    assert not (deliverables / "src" / "__pycache__").exists()
    assert not (deliverables / "gone.txt").exists()

    # Bare embeds in the copied document now point at this run's copy; unknown names are untouched.
    document = (deliverables / "document.md").read_text()
    prefix = f"runs/{RUN_ID}/deliverables"
    assert f"![[{prefix}/plots/power.png]]" in document
    assert f"![[{prefix}/plots/power.png|400]]" in document
    assert "![[other.png]]" in document
    assert (ready.paths.workspace / "document.md").read_text().startswith("# Thesis\n\n![[power.png]]")

    listed = note.section("Deliverables")
    assert listed.count(f"[[{prefix}/document|document]]") == 1
    assert f"- ![[{prefix}/plots/power.png]]" in listed
    assert f"- `{prefix}/src/`" in listed
    provenance = note.section("Provenance")
    assert provenance.count("[[02-strategy]]") == 1
    assert "- [[03-execution-r2]]" in provenance and "- [[04-crosscheck-r2]]" in provenance
    assert note.section("Limitations") == "None."  # nothing unresolved, every criterion met
    assert "[!warning]" not in note.section("Summary")
    assert final_mod.ISSUES_TAG not in note.meta.tags

    # Acceptance: rewritten canonically, with the criterion text; prose runs have no clean-room gate.
    assert note.section("Acceptance") == (
        "- AC-1 [met]: All tests pass on POSIX, FreeRTOS and QEMU.\n  - Evidence: all suites pass (see [[03-execution-r2]])\n"
        f"- AC-2 [met]: {CRITERION_2}\n  - Evidence: text=1804 bytes\n"
        f"- lint [met]: {final_mod.LINT_CRITERION}\n  - Evidence: maf lint found no critical problem in the exported "
        "Markdown; 1 major finding(s) remain (broken tables, links or math, placeholders, meta-commentary)"
    )  # the major one: ![[other.png]] embeds nothing that was exported
    assert CLEANROOM_SECTION not in note.sections
    assert ready.fakes.claude_code.calls == []
    assert output.index_updates.pop("export_note").startswith("final: 3 file(s), ")
    assert output.index_updates == {"criteria_unmet": 0, "unmet_criteria": [], "exported_at": ready.now}
    assert hf.validate_handoff(note) == []

    prompt = prompt_of(ready.fakes.claude.calls[0])
    assert f"- [[{prefix}/document|document]]" in prompt
    assert '<note name="03-execution-r2">' in prompt and '<note name="04-crosscheck-r2">' in prompt
    assert "## Acceptance Criteria" in prompt and "## Risks" not in prompt
    assert "- AC-1 [hard]: All tests pass on POSIX, FreeRTOS and QEMU." in prompt
    assert "`- AC-<n> [met|partial|unmet]: evidence`" in prompt and "Missing evidence is `unmet`" in prompt
    assert "Clean-room" not in prompt
    assert ready.index.brief in prompt
    assert "Unresolved critical issues" not in prompt
    assert prompt.count(hf.format_spec(HandoffKind.FINAL)) == 1


def test_final_does_not_read_the_execution_notes_stale_lint(ready: StageEnv) -> None:
    """``## Lint`` of 03-execution lists findings from before the fix pass; final reads the cross-check instead."""
    lint = "\n## Lint\n\n- [critical] pipeline-wikilink document.md:3: stale finding\n"
    ready.put("03-execution-r2", HandoffKind.EXECUTION, EXECUTION + lint, round=2)
    ready.fakes.claude.script(final_body())
    FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose"))
    prompt = prompt_of(ready.fakes.claude.calls[0])
    assert "stale finding" not in prompt and '<note name="03-execution-r2">' in prompt and "## Artifacts" in prompt


def test_unresolved_critical_issues_reach_limitations(ready: StageEnv) -> None:
    ready.fakes.claude.script(final_body())
    note = FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose", unresolved_critical=1)).notes[0].handoff
    assert UNRESOLVED in prompt_of(ready.fakes.claude.calls[0])
    limitations = note.section("Limitations")
    assert limitations.startswith("Critical issues left unresolved by the cross-check (added by maf):")
    assert limitations.endswith(UNRESOLVED)


def test_completed_with_issues_status_is_stated_in_prompt_and_note(ready: StageEnv) -> None:
    ready.fakes.claude.script(final_body())
    note = FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose", unresolved_critical=1)).notes[0].handoff

    prompt = prompt_of(ready.fakes.claude.calls[0])
    assert "## Unresolved critical issues (run status: completed_with_issues)" in prompt
    assert "ends with status `completed_with_issues`, not `completed`" in prompt and "[[04-crosscheck-r2]]" in prompt
    summary = note.section("Summary")
    assert summary.startswith(
        "> [!warning] Run status: completed_with_issues\n"
        "> 1 critical issue(s) remain unresolved after the cross-check loop cap (see [[04-crosscheck-r2]])"
    )
    assert "acceptance criteri" not in summary
    assert summary.endswith("The thesis is complete.")
    assert note.meta.tags == ["maf", "maf/final", final_mod.ISSUES_TAG]
    assert hf.validate_handoff(note) == []


def test_mark_completed_with_issues_is_idempotent(ready: StageEnv) -> None:
    ready.fakes.claude.script(final_body())
    note = FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose", unresolved_critical=2)).notes[0].handoff
    again = final_mod.mark_completed_with_issues(note, 2, "04-crosscheck-r2")
    assert again.sections == note.sections and again.meta.tags == note.meta.tags


def test_limitations_already_listing_the_issue_are_left_alone(ready: StageEnv) -> None:
    body = final_body().replace("## Limitations\n\nNone.", f"## Limitations\n\n{UNRESOLVED}")
    ready.fakes.claude.script(body)
    note = FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose", unresolved_critical=1)).notes[0].handoff
    assert note.section("Limitations") == UNRESOLVED


def test_rerun_is_idempotent(ready: StageEnv) -> None:
    ready.fakes.claude.script(final_body(), final_body())
    first = FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose")).notes[0].handoff
    second = FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose")).notes[0].handoff
    assert first.sections == second.sections
    assert sorted(p.name for p in ready.paths.deliverables.iterdir()) == ["document.md", "plots", "src"]


def test_oversized_trees_stay_in_the_workspace(ready: StageEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(final_mod, "MAX_DELIVERABLE_FILES", 0)
    ready.fakes.claude.script(final_body())
    output = FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose"))
    assert not (ready.paths.deliverables / "src").exists()
    assert ready.paths.workspace.resolve() / "src" not in output.deliverables
    listed = output.notes[0].handoff.section("Deliverables")
    assert f"`{ready.paths.workspace.resolve() / 'src'}` (left in the workspace" in listed


def test_uses_round_one_notes_when_no_loop(stage_env: StageEnv, sample_bodies: dict[str, str]) -> None:
    stage_env.put("02-strategy", HandoffKind.STRATEGY, sample_bodies["strategy"], from_="chatgpt")
    stage_env.put("03-execution", HandoffKind.EXECUTION, sample_bodies["execution"])
    stage_env.put("04-crosscheck", HandoffKind.CROSSCHECK, sample_bodies["crosscheck"], from_="maf")
    stage_env.fakes.claude.script(f"{sample_bodies['final'].rstrip()}\n\n{ACCEPTANCE}")
    output = FinalBackend().run_stage(stage_env.ctx("final", mode="prose"))
    note = output.notes[0].handoff
    assert note.meta.inputs == ["[[02-strategy]]", "[[03-execution]]", "[[04-crosscheck]]"]
    assert output.deliverables == []  # the sample execution's artifacts do not exist in the workspace
    assert "None." in prompt_of(stage_env.fakes.claude.calls[0])


def test_final_repair_then_failure(ready: StageEnv) -> None:
    ready.fakes.claude.script("nope", "still nope")
    with pytest.raises(HandoffInvalid):
        FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose"))


def test_unresolved_lines(sample_bodies: dict[str, str], stage_env: StageEnv) -> None:
    passed = stage_env.put("04-crosscheck", HandoffKind.CROSSCHECK, sample_bodies["crosscheck"], from_="maf")
    assert unresolved_lines(passed) == []
    body = sample_bodies["crosscheck"].replace("## Unresolved Critical\n\nNone.", f"## Unresolved Critical\n\n{UNRESOLVED}\nprose")
    looped = stage_env.put("04-crosscheck-r2", HandoffKind.CROSSCHECK, body, round=2, from_="maf")
    assert unresolved_lines(looped) == [UNRESOLVED]


# ---------------------------------------------------------------------------------------------- acceptance gate


def test_unmet_hard_criteria_end_completed_with_issues(ready: StageEnv) -> None:
    """The thesis-run gap: the final report says a hard criterion is unmet, so the run must not be ``completed``."""
    acceptance = "## Acceptance\n\n- AC-1 [partial]: FreeRTOS suite not run\n- AC-2 [unmet]: measured 2210 bytes\n"
    ready.fakes.claude.script(final_body(acceptance))
    output = FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose"))
    note = output.notes[0].handoff

    assert output.index_updates["criteria_unmet"] == 2
    assert output.index_updates["unmet_criteria"] == [
        "AC-1 [partial]: All tests pass on POSIX, FreeRTOS and QEMU.",
        f"AC-2 [unmet]: {CRITERION_2}",
    ]
    assert note.section("Summary").startswith(
        "> [!warning] Run status: completed_with_issues\n"
        "> 2 acceptance criteria are not met (AC-1, AC-2); see Acceptance and Limitations.\n\n"
    )
    assert final_mod.ISSUES_TAG in note.meta.tags
    assert note.section("Limitations") == (
        "Acceptance criteria not met (added by maf):\n\n"
        "- AC-1 [partial]: All tests pass on POSIX, FreeRTOS and QEMU.\n"
        f"- AC-2 [unmet]: {CRITERION_2}"
    )
    assert f"- AC-2 [unmet]: {CRITERION_2}\n  - Evidence: measured 2210 bytes" in note.section("Acceptance")
    assert hf.validate_handoff(note) == []


def test_soft_criteria_are_reported_but_not_counted(ready: StageEnv) -> None:
    strategy = ready.written["02-strategy"]
    soft = "- AC-1 [hard]: All tests pass on POSIX, FreeRTOS and QEMU.\n- AC-2 [soft]: README has a diagram."
    ready.put("02-strategy", HandoffKind.STRATEGY, hf.render_body({**strategy.sections, "Acceptance Criteria": soft}))
    acceptance = "## Acceptance\n\n- AC-1 [hard] [met]: passes\n- AC-2 [unmet] [soft]: no diagram\n"
    ready.fakes.claude.script(final_body(acceptance))
    output = FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose"))
    note = output.notes[0].handoff

    assert (output.index_updates["criteria_unmet"], output.index_updates["unmet_criteria"]) == (0, [])
    assert "[!warning]" not in note.section("Summary") and final_mod.ISSUES_TAG not in note.meta.tags
    assert "- AC-2 [unmet]: README has a diagram. (soft criterion)\n  - Evidence: no diagram" in note.section("Acceptance")
    assert note.section("Limitations") == (
        "Soft acceptance criteria not met (added by maf):\n\n- AC-2 [unmet]: README has a diagram."
    )
    assert "- AC-2 [soft]: README has a diagram." in prompt_of(ready.fakes.claude.calls[0])


def test_limitations_that_already_name_an_unmet_criterion_are_not_repeated(ready: StageEnv) -> None:
    acceptance = "## Acceptance\n\n- AC-1 [met]: passes\n- AC-2 [unmet]: 2210 bytes\n"
    body = final_body(acceptance).replace("## Limitations\n\nNone.", "## Limitations\n\n- AC-2 fails: 2210 bytes.")
    ready.fakes.claude.script(body)
    note = FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose")).notes[0].handoff
    assert note.section("Limitations") == "- AC-2 fails: 2210 bytes."


def test_missing_verdicts_get_one_repair(ready: StageEnv) -> None:
    ready.fakes.claude.script(final_body(""), final_body())
    output = FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose"))

    repair = prompt_of(ready.fakes.claude.calls[1])
    assert repair.startswith("Your previous final handoff did not pass validation.")
    assert "missing section '## Acceptance' (after '## Verification')" in repair and "(AC-1, AC-2)" in repair
    assert output.index_updates["criteria_unmet"] == 0
    assert output.notes[0].handoff.meta.cost_usd == pytest.approx(0.02)  # the call plus its repair


def test_verdicts_still_missing_after_the_repair_fail_the_stage(ready: StageEnv) -> None:
    half = "## Acceptance\n\n- AC-1 [met]: passes\n"
    ready.fakes.claude.script(final_body(half), final_body(half))
    with pytest.raises(HandoffInvalid, match="no verdict for AC-2"):
        FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose"))


CRITERIA = [Criterion("AC-1", True, "tests pass"), Criterion("AC-3", False, "has a diagram")]


def _final_with(acceptance: str | None, stage_env: StageEnv) -> Any:
    sections = {"Summary": "s", "Deliverables": "d", "Verification": "v", "Provenance": "- [[02-strategy]]", "Limitations": "None."}
    if acceptance is not None:
        sections["Acceptance"] = acceptance
    return stage_env.put("05-final", HandoffKind.FINAL, hf.render_body(sections))


@pytest.mark.parametrize(
    ("acceptance", "errors"),
    [
        ("- AC-1 [met]: ok\n- AC-3 [unmet]: none", []),
        ("Verdicts below.\n\n- **AC-1** [MET]: ok\n  - more evidence\n- AC-3 [partial]\n  started only\n", []),
        ("- AC-1 [met]: ok\n- AC-3 [soft] [unmet]: none\n- clean-room [met]: written by maf, ignored", []),
        (None, ["missing section '## Acceptance' (after '## Verification'): one line per acceptance criterion "
                "(AC-1, AC-3), each `- AC-<n> [met|partial|unmet]: evidence`"]),
        ("- AC-1 [met]: ok", ["## Acceptance: no verdict for AC-3 (has a diagram)"]),
        ("- AC-1 [met]: ok\n- AC-1 [unmet]: no\n- AC-3 [met]: ok", ["## Acceptance: AC-1 has more than one verdict"]),
        ("- AC-1 [done]: ok\n- AC-3 [met]: ok", ["## Acceptance: AC-1 [done] must be [met], [partial] or [unmet]"]),
        ("- AC-1 [met]:\n- AC-3 [met]: ok", ["## Acceptance: AC-1 needs its evidence after the colon"]),
        ("- AC-1 [met]: ok\n- AC-3 [met]: ok\n- AC-9 [met]: extra", ["## Acceptance: AC-9 is not one of the criteria (AC-1, AC-3)"]),
        ("- AC-1: met, ok\n- AC-3 [met]: ok", [
            "## Acceptance: not a verdict line (expected `- AC-<n> [met|partial|unmet]: evidence`): '- AC-1: met, ok'",
            "## Acceptance: no verdict for AC-1 (tests pass)",
        ]),
    ],
)
def test_acceptance_errors(stage_env: StageEnv, acceptance: str | None, errors: list[str]) -> None:
    assert acceptance_errors(_final_with(acceptance, stage_env), CRITERIA) == errors


def test_acceptance_verdicts_fold_continuations_and_keep_the_first(stage_env: StageEnv) -> None:
    note = _final_with("- AC-1 [partial]: ran POSIX\n  - not QEMU\nlazy line\n\n- AC-1 [met]: dup\nprose after a gap", stage_env)
    verdicts = acceptance_verdicts(note, CRITERIA)
    assert [(v.id, v.verdict, v.evidence, v.hard, v.blocking) for v in verdicts] == [
        ("AC-1", "partial", "ran POSIX not QEMU lazy line", True, True),
        ("AC-3", "unmet", "the final report gave no valid verdict for it (recorded by maf)", False, False),
    ]
    assert acceptance_errors(note, []) == []  # no criteria: nothing is required


# ---------------------------------------------------------------------------------------------- code and mixed runs


CODE_EXECUTION = """## Summary

Allocator done.

## Artifacts

- `src/alloc.c` - allocator
- `logs/unit.log` - unit test output
- `README.md` - build and reproduction

## Implementation Notes

n

## Verification

v

## Known Limitations

None.
"""


@dataclass
class CleanRoom:
    """Scripted clean-room session: records what the clean room held, writes the exit-code and log files."""

    room: Path
    exit_file: str | None = "0\n"
    reported: int = 0
    command: str = "make all"
    missing: list[str] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    requests: list[CompletionRequest] = field(default_factory=list)
    before: Callable[[Path], None] | None = None

    def __call__(self, request: CompletionRequest) -> dict[str, Any]:
        self.requests.append(request)
        self.seen = {p.relative_to(self.room).as_posix() for p in self.room.rglob("*") if p.is_file() or p.is_symlink()}
        if self.before is not None:
            self.before(self.room)
        (self.room / REPRO_LOG).write_text("cc -o build/unit tests/unit.c\nALL TESTS PASSED\n", encoding="utf-8")
        if self.exit_file is not None:
            (self.room / REPRO_EXIT).write_text(self.exit_file, encoding="utf-8")
        return {"command": self.command, "exit_code": self.reported, "missing_paths": self.missing, "log_tail": "tail"}


@pytest.fixture
def code_ready(stage_env: StageEnv, sample_bodies: dict[str, str]) -> StageEnv:
    stage_env.put("02-strategy", HandoffKind.STRATEGY, sample_bodies["strategy"], from_="chatgpt")
    stage_env.put("03-execution", HandoffKind.EXECUTION, CODE_EXECUTION)
    stage_env.put("04-crosscheck", HandoffKind.CROSSCHECK, sample_bodies["crosscheck"], from_="maf")
    files: dict[str, str | bytes] = {
        "README.md": "# Allocator\n\nReproduce: `make all`.\n\n![[frag.png]]\n",
        "Makefile": "all:\n\tcc tests/host_sizes.c\n",
        "src/alloc.c": "int x;\n",
        "tests/host_sizes.c": "/* added in a fix round, never listed */\n",
        "logs/unit.log": "ok\n",
        "plots/frag.png": b"\x89PNG",
        "burnsim/__init__.py": "",  # empty, but not a root placeholder: kept
        ".gitignore": "build/\n",
        # left out:
        ".maf/execution-r1.md": "prompt",
        ".claude/.cc-writes/x": "state",
        "FreeRTOS-Kernel/tasks.c": "kernel",
        "build/alloc.o": b"\x7fELF",
        "src/__pycache__/a.cpython-312.pyc": b"\x00",
        "pkg.egg-info/PKG-INFO": "meta",
        "node_modules/m/index.js": "js",
        ".env": "",
        "package.json": "",
        "bunfig.toml": "",
    }
    for rel, content in files.items():
        stage_env.workspace_file(rel, content)
    return stage_env


EXPORTED = {
    ".gitignore", "Makefile", "README.md", "burnsim/__init__.py", "logs/unit.log", "plots/frag.png", "src/alloc.c",
    "tests/host_sizes.c",
}


def _exported(env: StageEnv) -> set[str]:
    root = env.paths.deliverables
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_code_run_exports_the_workspace_tree_and_passes_the_clean_room(code_ready: StageEnv) -> None:
    room = cleanroom_dir(code_ready.paths.workspace)
    session = CleanRoom(room)
    code_ready.fakes.claude_code.script(session)
    code_ready.fakes.claude.script(final_body())
    output = FinalBackend().run_stage(code_ready.ctx("final", mode="code"))
    note = output.notes[0].handoff

    # (1) Export: the workspace tree minus pipeline state, the kernel, inputs, build output and empty placeholders.
    assert _exported(code_ready) == EXPORTED
    assert not (code_ready.paths.deliverables / "inputs").exists()  # the brief's spec.pdf stays in the workspace
    prefix = f"runs/{RUN_ID}/deliverables"
    assert f"![[{prefix}/plots/frag.png]]" in (code_ready.paths.deliverables / "README.md").read_text()
    listed = note.section("Deliverables").splitlines()[1:]  # after the link the model wrote itself
    assert listed[:4] == [
        "", f"- `{prefix}/src/alloc.c`", f"- `{prefix}/logs/unit.log`", f"- [[{prefix}/README|README]]"
    ]
    assert f"- `{prefix}/tests/`" in listed and f"- `{prefix}/Makefile`" in listed and f"- `{prefix}/.gitignore`" in listed
    ws = code_ready.paths.workspace.resolve()
    assert ws / "tests" in output.deliverables and ws / "README.md" in output.deliverables

    # (2) Clean room: a copy of the export plus the provisioned kernel, outside the workspace, one metered session.
    assert session.seen == EXPORTED | {"FreeRTOS-Kernel/tasks.c", "inputs/spec.pdf"}  # provided, not deliverables
    assert not room.is_relative_to(code_ready.paths.workspace)
    assert room == code_ready.paths.workspace.parent / ".maf-cleanroom" / RUN_ID
    assert not (room / "FreeRTOS-Kernel").is_symlink() and (room / "FreeRTOS-Kernel" / "tasks.c").read_text() == "kernel"
    (request,) = session.requests
    assert (request.schema_name, request.effort, request.json_schema) == ("cleanroom", "medium", final_mod.CLEANROOM_SCHEMA)
    assert request.max_budget_usd == code_ready.settings.cleanroom_budget_usd
    prompt = prompt_of(request)
    assert f"cd {room} && ( COMMAND ) > {room / REPRO_LOG} 2>&1; echo $? > {room / REPRO_EXIT}" in prompt
    assert "`FreeRTOS-Kernel/` and `inputs/` are copied in from the workspace" in prompt and "8 file(s)" in prompt
    assert "dated older than the sources and build files" in prompt
    assert "- AC-1 [hard]: All tests pass on POSIX, FreeRTOS and QEMU." in prompt  # canonical criterion lines
    assert (code_ready.paths.workspace / ".maf" / "cleanroom-prompt.md").read_text() == prompt
    entry = code_ready.ledger.entries[0]
    assert (entry.stage, entry.purpose, entry.provider) == ("final", "cleanroom", "claude_code")

    # (3) The note: clean-room verdict and section, final prompt told about it, cost includes the session.
    assert "- clean-room [met]: Clean-room reproduction" in note.section("Acceptance")
    assert note.section(CLEANROOM_SECTION).startswith(
        "- Result: passed\n- Command: `make all`, run in a fresh copy of `deliverables/` outside the workspace, with "
        "generated files dated older than their sources\n"
        "- Exit code: 0 (`REPRO_EXIT`); the session reported 0\n- Missing paths: none reported\n"
    )
    assert "Log tail (`REPRO_LOG`):" in note.section(CLEANROOM_SECTION)
    assert "ALL TESTS PASSED" in note.section(CLEANROOM_SECTION)
    final_prompt = prompt_of(code_ready.fakes.claude.calls[0])
    assert "## Clean-room reproduction (run by maf)" in final_prompt and "- Result: passed" in final_prompt
    assert note.meta.cost_usd == pytest.approx(0.01 + 0.50)
    assert output.index_updates["criteria_unmet"] == 0
    assert output.index_updates["export_note"].startswith("final: 8 file(s), ")
    assert "[!warning]" not in note.section("Summary")
    assert hf.validate_handoff(note) == []


def test_reexport_replaces_the_previous_tree(code_ready: StageEnv) -> None:
    code_ready.fakes.claude_code.default = CleanRoom(cleanroom_dir(code_ready.paths.workspace))
    code_ready.fakes.claude.script(final_body(), final_body())
    FinalBackend().run_stage(code_ready.ctx("final", mode="mixed"))
    (code_ready.paths.workspace / "logs" / "unit.log").unlink()
    (code_ready.paths.deliverables / "stray.txt").write_text("user file")
    FinalBackend().run_stage(code_ready.ctx("final", mode="mixed"))
    assert _exported(code_ready) == EXPORTED - {"logs/unit.log"}
    assert not [p for p in code_ready.paths.root.iterdir() if p.name.startswith(".deliverables")]


def test_settings_export_exclude_adds_patterns(code_ready: StageEnv) -> None:
    code_ready.settings = code_ready.settings.model_copy(update={"export_exclude": ("*.log",)})
    code_ready.fakes.claude_code.script(CleanRoom(cleanroom_dir(code_ready.paths.workspace)))
    code_ready.fakes.claude.script(final_body())
    FinalBackend().run_stage(code_ready.ctx("final", mode="code"))
    assert _exported(code_ready) == EXPORTED - {"logs/unit.log"}  # and the defaults still apply


def _gate_problem(output: Any) -> str:
    note = output.notes[0].handoff
    assert output.index_updates["criteria_unmet"] == 1
    assert output.index_updates["unmet_criteria"] == [
        "clean-room [unmet]: Clean-room reproduction: the reproduction command the deliverables document succeeds in "
        "a fresh copy of the exported deliverables."
    ]
    assert note.section("Summary").startswith(
        "> [!warning] Run status: completed_with_issues\n"
        "> 1 acceptance criterion is not met (clean-room); see Acceptance and Limitations."
    )
    assert final_mod.ISSUES_TAG in note.meta.tags
    assert "- clean-room [unmet]: Clean-room reproduction" in note.section("Limitations")
    details = note.section(CLEANROOM_SECTION)
    assert details.startswith("- Result: failed: ")
    return details.split("\n", 1)[0].removeprefix("- Result: failed: ")


def _symlinked_exit(room: Path) -> None:
    target = room.parent / "real-exit"
    target.write_text("0\n")
    (room / REPRO_EXIT).symlink_to(target)


@pytest.mark.parametrize(
    ("session", "problem"),
    [
        (dict(exit_file="2\n", reported=2), "`make all` exited with 2 in the clean room"),
        (dict(exit_file=None), "REPRO_EXIT was not written, so the reproduction command was not run as instructed "
                               "(or it was killed before it finished)"),
        (dict(exit_file="1\n", reported=0), "the session reported exit code 0, but REPRO_EXIT holds 1"),
        (dict(exit_file="ok\n"), "REPRO_EXIT does not hold an integer exit code ('ok')"),
        (dict(missing=["tests/host_sizes.c"]), "the exported deliverables lack `tests/host_sizes.c`"),
        (dict(exit_file="2\n", reported=2, missing=["data/x.csv"]),
         "`make all` exited with 2 in the clean room; the exported deliverables lack `data/x.csv`"),
        (dict(command="", exit_file=None, reported=-1),
         "no reproduction command is documented in the deliverables or given in the acceptance criteria"),
        (dict(exit_file=None, before=_symlinked_exit), "REPRO_EXIT was not written, so the reproduction command was "
                                                       "not run as instructed (or it was killed before it finished)"),
    ],
)
def test_failed_clean_room_is_an_unmet_hard_criterion(code_ready: StageEnv, session: dict[str, Any], problem: str) -> None:
    code_ready.fakes.claude_code.script(CleanRoom(cleanroom_dir(code_ready.paths.workspace), **session))
    code_ready.fakes.claude.script(final_body())
    output = FinalBackend().run_stage(code_ready.ctx("final", mode="code"))
    assert _gate_problem(output) == problem
    assert problem in prompt_of(code_ready.fakes.claude.calls[0])


def test_unparsable_report_and_exhausted_budget_fail_the_gate_without_failing_the_run(code_ready: StageEnv) -> None:
    code_ready.fakes.claude_code.script(
        "I ran it and it worked!",
        ClaudeCodeBudgetExhausted("max budget", provider="claude_code", cost_usd=1.5),
    )
    code_ready.fakes.claude.script(final_body(), final_body())
    first = FinalBackend().run_stage(code_ready.ctx("final", mode="code"))
    assert _gate_problem(first).startswith("the clean-room session returned no valid report (output is not valid JSON")
    assert first.notes[0].handoff.meta.cost_usd == pytest.approx(0.01 + 0.50)
    second = FinalBackend(cleanroom_budget_usd=0.75).run_stage(code_ready.ctx("final", mode="code"))
    assert _gate_problem(second) == "the clean-room session ran out of its $0.75 budget before reporting"
    assert second.notes[0].handoff.meta.cost_usd == pytest.approx(0.01 + 1.5)
    assert code_ready.fakes.claude_code.calls[1].max_budget_usd == 0.75
    assert sum(e.cost_usd for e in code_ready.ledger.entries if e.purpose == "cleanroom") == pytest.approx(2.0)


def test_sandbox_failure_in_the_clean_room_propagates(code_ready: StageEnv) -> None:
    code_ready.fakes.claude_code.script(SandboxUnavailable("bwrap: setting up uid map", provider="claude_code"))
    with pytest.raises(SandboxUnavailable):
        FinalBackend().run_stage(code_ready.ctx("final", mode="code"))
    assert code_ready.fakes.claude.calls == []


def test_clean_room_verifies_the_sandbox_first(code_ready: StageEnv) -> None:
    """A process resumed straight into final has not preflighted yet: the clean room checks the sandbox first."""
    code = SandboxedFake(name="claude_code", agent="claude", cost_per_call=0.50, workspace=code_ready.paths.workspace)
    code_ready.fakes.claude_code = code

    def reply(request: CompletionRequest) -> dict[str, Any]:
        if request.schema_name == "preflight":
            return {"digest": code.probe_digest()}
        return CleanRoom(cleanroom_dir(code_ready.paths.workspace))(request)

    code.default = reply
    code_ready.fakes.claude.script(final_body())
    output = FinalBackend().run_stage(code_ready.ctx("final", mode="code"))
    assert [r.schema_name for r in code.calls] == ["preflight", "cleanroom"]
    assert [(e.stage, e.purpose) for e in code_ready.ledger.entries][:2] == [("final", "preflight"), ("final", "cleanroom")]
    assert output.index_updates["criteria_unmet"] == 0
    assert output.notes[0].handoff.meta.cost_usd == pytest.approx(0.01 + 0.50)  # the preflight belongs to no note


def test_empty_export_fails_the_gate_without_a_session(stage_env: StageEnv, sample_bodies: dict[str, str]) -> None:
    stage_env.put("02-strategy", HandoffKind.STRATEGY, sample_bodies["strategy"], from_="chatgpt")
    stage_env.put("03-execution", HandoffKind.EXECUTION, sample_bodies["execution"])
    stage_env.put("04-crosscheck", HandoffKind.CROSSCHECK, sample_bodies["crosscheck"], from_="maf")
    stage_env.workspace_file(".maf/execution-r1.md", "prompt only")
    stage_env.fakes.claude.script(final_body())
    output = FinalBackend().run_stage(stage_env.ctx("final", mode="code"))
    assert _gate_problem(output) == "the export is empty, so there is nothing to reproduce"
    assert stage_env.fakes.claude_code.calls == []


def test_oversized_export_stops_the_stage_before_any_call(code_ready: StageEnv) -> None:
    (code_ready.paths.deliverables / "old.txt").write_text("previous export")
    code_ready.settings = code_ready.settings.model_copy(update={"export_max_mb": 0.00001})
    with pytest.raises(ExportTooLarge, match="over the cap of 10 B"):
        FinalBackend().run_stage(code_ready.ctx("final", mode="code"))
    assert _exported(code_ready) == {"old.txt"}  # left as it was
    assert code_ready.fakes.claude_code.calls == [] and code_ready.fakes.claude.calls == []


def test_judge_cleanroom_requires_the_prescribed_command_form(tmp_path: Path) -> None:
    """``( CMD ) > REPRO_LOG 2>&1; echo $? > REPRO_EXIT`` always writes the log first: a session that only ran
    ``echo 0 > REPRO_EXIT`` (or left an old file behind) does not pass, and its self-reported log is labelled so."""
    report = {"command": "make", "exit_code": 0, "missing_paths": [], "log_tail": "ok"}
    (tmp_path / REPRO_EXIT).write_text("0")
    alone = judge_cleanroom(report, tmp_path, started=0.0)
    assert not alone.passed and alone.problem.startswith("REPRO_LOG was not written, so the command was not run in")
    assert alone.log_reported and "Log tail (reported by the session; `REPRO_LOG` was not written):" in alone.details()

    (tmp_path / REPRO_LOG).write_text("cc ...\nok\n")
    now = __import__("time").time()
    os.utime(tmp_path / REPRO_EXIT, (now - 60, now - 60))  # the exit code predates the log
    assert "REPRO_EXIT is older than REPRO_LOG" in judge_cleanroom(report, tmp_path).problem
    assert "predates the clean-room session" in judge_cleanroom(report, tmp_path, started=now - 30).problem
    (tmp_path / REPRO_EXIT).write_text("0")
    passed = judge_cleanroom(report, tmp_path, started=now - 30)
    assert passed == CleanroomResult("make", 0, 0, (), "cc ...\nok\n", False, "", 0.0)
    assert passed.passed and passed.verdict.met
    fenced = CleanroomResult("make", 0, 0, (), "```\nx\n```").details()
    assert "\n````text\n```\nx\n```\n````" in fenced


# ---------------------------------------------------------------------------------------------- clean-room hardening


def test_the_clean_room_leaves_the_final_reports_worst_case_affordable(code_ready: StageEnv) -> None:
    """With a $2.50 cap, a $2.00 final worst case and a $1.50 clean-room budget, the session used to spend what the
    final call needed: BudgetExceeded, no 05-final, and a resume paid for the session again. Now it gets $0.50."""
    from maf.ledger import Ledger

    code_ready.ledger = Ledger(RUN_ID, 2.50)
    code_ready.fakes.claude.worst_case_usd = 2.00
    code_ready.fakes.claude_code.cost_per_call = 0.50  # the CLI keeps to its --max-budget-usd
    session = CleanRoom(cleanroom_dir(code_ready.paths.workspace))
    code_ready.fakes.claude_code.script(session)
    code_ready.fakes.claude.script(final_body())
    output = FinalBackend().run_stage(code_ready.ctx("final", mode="code"))
    assert session.requests[0].max_budget_usd == pytest.approx(0.50)
    assert output.index_updates["criteria_unmet"] == 0 and output.notes[0].name == "05-final"


def test_too_little_budget_skips_the_session_and_fails_the_gate(code_ready: StageEnv) -> None:
    from maf.ledger import Ledger

    code_ready.ledger = Ledger(RUN_ID, 2.20)
    code_ready.fakes.claude.worst_case_usd = 2.00
    code_ready.fakes.claude.script(final_body())
    output = FinalBackend().run_stage(code_ready.ctx("final", mode="code"))
    assert code_ready.fakes.claude_code.calls == []
    assert _gate_problem(output) == (
        "not run: budget (the run has $2.20 left, and $2.00 of it is held back for the final report, leaving less "
        "than the $0.25 a session needs)"
    )


def test_cleanroom_budget() -> None:
    assert final_mod.cleanroom_budget(25.0, 2.0, 1.2, 1.5) == 1.5
    assert final_mod.cleanroom_budget(4.0, 2.0, 1.2, 1.5) == pytest.approx(0.8)
    assert final_mod.cleanroom_budget(2.0, 2.0, 1.2, 1.5) == 0.0


@pytest.mark.skipif(shutil.which("make") is None or shutil.which("cc") is None, reason="needs make and cc")
def test_generated_files_are_stale_in_the_clean_room(tmp_path: Path) -> None:
    """The export keeps the workspace's file times, where results and binaries are newer than their sources, so
    ``make all`` in the room said "Nothing to be done" and REPRO_EXIT=0 proved nothing. The room backdates them."""
    ws, deliverables = tmp_path / "workspaces" / "r1", tmp_path / "deliverables"
    (ws / "tools").mkdir(parents=True)
    (ws / "results").mkdir()
    (ws / "tools" / "sim.py").write_text("print('1,2')\n")
    (ws / "prog.c").write_text("int main(void){return 0;}\n")
    (ws / "Makefile").write_text(
        "all: results/metrics.csv prog\nresults/metrics.csv: tools/sim.py\n\tpython3 tools/sim.py > $@\n"
        "prog: prog.c\n\tcc -o prog prog.c\n"
    )
    subprocess.run(["make", "all"], cwd=ws, check=True, capture_output=True)
    assert subprocess.run(["make", "-q", "all"], cwd=ws).returncode == 0  # up to date in the workspace
    shutil.copytree(ws, deliverables)  # copy2: the export keeps the times
    (deliverables / REPRO_EXIT).write_text("0\n")  # stale result files never reach the room
    (deliverables / REPRO_LOG).write_text("old\n")

    room, provided = final_mod.prepare_cleanroom(ws, deliverables)
    assert provided == [] and not (room / REPRO_EXIT).exists() and not (room / REPRO_LOG).exists()
    assert subprocess.run(["make", "-q", "all"], cwd=room).returncode == 1  # both targets out of date
    assert (room / "prog.c").stat().st_mtime > (room / "prog").stat().st_mtime
    assert (room / "tools" / "sim.py").stat().st_mtime > (room / "results" / "metrics.csv").stat().st_mtime


class BindableCleanRoom(FakeProvider):
    """A ``claude_code`` fake with ``bound_to``, as the real provider has: records how the clean room binds it."""

    bound: list[tuple[Path, tuple[str, ...], bool]] = []

    def bound_to(self, workspace: Path, *, deny_read: tuple[str, ...] = (), bash_default_is_max: bool = False) -> FakeProvider:
        self.bound.append((workspace, tuple(deny_read), bash_default_is_max))
        return self


def test_the_session_is_bound_to_the_room_and_workspace_paths_fail_the_gate(code_ready: StageEnv) -> None:
    """The room lies beside the workspace, the session may neither read nor write the workspace, and deliverables
    that reach the workspace by its absolute path fail even when the command exits 0 there."""
    ws = code_ready.paths.workspace
    code_ready.workspace_file("Makefile", f"all:\n\tcat {ws.resolve()}/build/table.txt\n")
    provider = BindableCleanRoom(name="claude_code", agent="claude", cost_per_call=0.50, worst_case_usd=2.0)
    provider.bound = []
    code_ready.fakes.claude_code = provider
    room = cleanroom_dir(ws)
    provider.script(CleanRoom(room))
    code_ready.fakes.claude.script(final_body())
    output = FinalBackend().run_stage(code_ready.ctx("final", mode="code"))
    ((bound_room, deny, long_commands),) = provider.bound
    assert bound_room == room and str(ws.resolve()) in deny and long_commands
    assert _gate_problem(output) == (
        "the deliverables name the workspace's absolute path (`Makefile`), which does not exist in a copy"
    )
    assert "- Workspace paths in the deliverables: `Makefile`" in output.notes[0].handoff.section(CLEANROOM_SECTION)


def test_a_resumed_final_reuses_the_clean_room_result_for_identical_deliverables(code_ready: StageEnv) -> None:
    """Final re-runs from the start on resume: the judged result of an identical room and prompt is kept in the run
    folder, so the session is not paid for twice; a changed deliverable runs it again."""
    room = cleanroom_dir(code_ready.paths.workspace)
    code_ready.fakes.claude_code.script(CleanRoom(room))
    code_ready.fakes.claude.script(final_body(), final_body(), final_body())
    first = FinalBackend().run_stage(code_ready.ctx("final", mode="code")).notes[0].handoff
    second = FinalBackend().run_stage(code_ready.ctx("final", mode="code")).notes[0].handoff
    assert len(code_ready.fakes.claude_code.calls) == 1
    assert "- Reused: the result of an earlier attempt of this stage on identical deliverables" in (
        second.section(CLEANROOM_SECTION)
    )
    assert second.meta.cost_usd == first.meta.cost_usd == pytest.approx(0.01 + 0.50)
    assert (code_ready.paths.root / final_mod.CLEANROOM_RECORD).is_file()
    code_ready.workspace_file("src/alloc.c", "int y;\n")
    code_ready.fakes.claude_code.script(CleanRoom(room))
    FinalBackend().run_stage(code_ready.ctx("final", mode="code"))
    assert len(code_ready.fakes.claude_code.calls) == 2


def test_a_pipeline_link_in_the_export_fails_the_lint_gate(code_ready: StageEnv) -> None:
    """The last fix pass can leave a pipeline link that no later lint sees: final lints the exported tree."""
    code_ready.workspace_file("README.md", "# Allocator\n\nReproduce: `make all`. See [[04-crosscheck]].\n")
    code_ready.fakes.claude_code.script(CleanRoom(cleanroom_dir(code_ready.paths.workspace)))
    code_ready.fakes.claude.script(final_body())
    output = FinalBackend().run_stage(code_ready.ctx("final", mode="code"))
    assert output.index_updates["unmet_criteria"] == [f"lint [unmet]: {final_mod.LINT_CRITERION}"]
    acceptance = output.notes[0].handoff.section("Acceptance")
    assert "- lint [unmet]:" in acceptance
    assert "maf lint found 1 critical problem(s) in the exported Markdown: `README.md:3` pipeline-wikilink" in acceptance
    assert "## Other checks run by maf" in prompt_of(code_ready.fakes.claude.calls[0])


CITING = "# Survey\n\nTLSF is O(1) [1].\n\n## References\n\n1. M. Masmano et al., TLSF, ECRTS 2004.\n"


def _audit_record(env: StageEnv, text: str, **entry: object) -> None:
    from maf.stages import crosscheck as cc

    fields = {"sha256": cc.text_digest(text), "status": "done", "references": 1, "not_verified": 0, "round": 1} | entry
    record = {"documents": {"document.md": fields}}
    (env.paths.root / cc.AUDIT_RECORD).write_text(__import__("json").dumps(record))


@pytest.mark.parametrize(
    ("entry", "verdict", "evidence"),
    [
        ({}, "met", "the source audit verified all 1 reference(s) of `document.md` in the text as shipped"),
        (None, "unmet", "`document.md` was never audited"),
        ({"sha256": "0" * 64}, "unmet", "`document.md` changed after its last audit (round 1)"),
        ({"not_verified": 1}, "unmet", "`document.md` has 1 reference(s) the audit did not verify"),
        ({"unaudited": 4}, "unmet", "`document.md` has 4 reference(s) left unaudited (over the cap)"),
        ({"status": "failed"}, "unmet", "the audit of `document.md` failed"),
        ({"incomplete": "1 of 2 audit call(s) failed"}, "unmet", "the audit of `document.md` is incomplete (1 of 2"),
    ],
)
def test_the_source_audit_gate_judges_the_shipped_text(
    ready: StageEnv, entry: dict[str, object] | None, verdict: str, evidence: str
) -> None:
    """The thesis replay: the fixer re-attributed from memory after the only audit, and final's model marked the
    source criterion met from ``SRC-n: fixed``. maf records ``source-audit`` itself from the latest audit record."""
    ready.workspace_file("document.md", CITING)
    if entry is not None:
        _audit_record(ready, CITING, **entry)
    ready.fakes.claude.script(final_body())
    output = FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose"))
    line = next(l for l in output.notes[0].handoff.section("Acceptance").splitlines() if l.startswith("- source-audit"))
    assert line.startswith(f"- source-audit [{verdict}]: {final_mod.SOURCE_AUDIT_CRITERION}")
    assert evidence in output.notes[0].handoff.section("Acceptance")
    assert output.index_updates["criteria_unmet"] == (verdict == "unmet")


def test_the_source_audit_gate_applies_only_to_documents_that_cite(ready: StageEnv) -> None:
    ready.fakes.claude.script(final_body(), final_body())
    note = FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose")).notes[0].handoff
    assert "source-audit" not in note.section("Acceptance")  # the thesis fixture cites nothing
    ready.workspace_file("document.md", CITING)
    ready.settings = ready.settings.model_copy(update={"source_audit": False})
    note = FinalBackend().run_stage(ready.ctx("final", round=2, mode="prose")).notes[0].handoff
    assert "source-audit" not in note.section("Acceptance")  # the audit is off: the strategy's criterion decides


def test_verdict_lines_for_maf_gates_are_ignored(stage_env: StageEnv) -> None:
    note = _final_with("- AC-1 [met]: ok\n- AC-3 [met]: ok\n- lint [met]: x\n- source-audit [met]: y", stage_env)
    assert acceptance_errors(note, CRITERIA) == []


# ---------------------------------------------------------------------------------------------- maf export


def test_export_run_repeats_the_export_without_model_calls(code_ready: StageEnv) -> None:
    index = code_ready.index.model_copy(update={"mode": "code"})
    export = export_run(code_ready.vault, index)
    assert _exported(code_ready) == EXPORTED
    assert export.files == len(EXPORTED) and export.tree is not None
    assert sorted(export.tree.placeholders) == [".env", "bunfig.toml", "package.json"]
    assert export.describe() == (
        f"{len(EXPORTED)} file(s), {export.total_bytes} B; excluded: build/, node_modules/, pkg.egg-info/, src/__pycache__/"
    )  # pipeline state (.maf/, .claude/, the kernel) is not reported
    assert [d.rel for d in export.deliverables][:3] == ["src/alloc.c", "logs/unit.log", "README.md"]
    assert all(not f.calls for f in code_ready.fakes.all())


def test_export_run_for_prose_and_undecided_runs(ready: StageEnv) -> None:
    prose = ready.index.model_copy(update={"mode": "prose", "round": 2})
    export = export_run(ready.vault, prose)
    assert [d.rel for d in export.deliverables] == ["document.md", "plots/power.png", "src"]
    with pytest.raises(ExportError, match="no execution mode yet"):
        export_run(ready.vault, ready.index.model_copy(update={"mode": None}))
    with pytest.raises(ExportError, match="no execution note"):
        export_run(ready.vault, prose.model_copy(update={"handoffs": []}))
