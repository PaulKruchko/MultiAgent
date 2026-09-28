"""Tests for the final stage: deliverable copies, the final note, and the Python-enforced rules."""

from __future__ import annotations

import pytest

from maf import handoff as hf
from maf.handoff import HandoffInvalid, HandoffKind
from maf.stages import final as final_mod
from maf.stages.final import FinalBackend, unresolved_lines
from test_stages_base import RUN_ID, StageEnv, prompt_of, stage_env  # noqa: F401

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

FINAL_BODY = """## Summary

The thesis is complete.

## Deliverables

- [[runs/{run}/deliverables/document|document]]

## Verification

All simulations ran.

## Provenance

- [[02-strategy]]

## Limitations

None.
"""

UNRESOLVED = "- [critical] GEM-2: runaway not suppressed above 500 MW"


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


def test_final_copies_deliverables_and_enforces_links(ready: StageEnv) -> None:
    ready.fakes.claude.script(FINAL_BODY.format(run=RUN_ID))
    output = FinalBackend().run_stage(ready.ctx("final", round=2))

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
    assert note.section("Limitations") == "None."  # unresolved_critical == 0

    prompt = prompt_of(ready.fakes.claude.calls[0])
    assert f"- [[{prefix}/document|document]]" in prompt
    assert '<note name="03-execution-r2">' in prompt and '<note name="04-crosscheck-r2">' in prompt
    assert "## Acceptance Criteria" in prompt and "## Risks" not in prompt
    assert ready.index.brief in prompt
    assert "Unresolved critical issues" not in prompt
    assert prompt.count(hf.format_spec(HandoffKind.FINAL)) == 1


def test_unresolved_critical_issues_reach_limitations(ready: StageEnv) -> None:
    ready.fakes.claude.script(FINAL_BODY.format(run=RUN_ID))
    note = FinalBackend().run_stage(ready.ctx("final", round=2, unresolved_critical=1)).notes[0].handoff
    assert UNRESOLVED in prompt_of(ready.fakes.claude.calls[0])
    limitations = note.section("Limitations")
    assert limitations.startswith("Critical issues left unresolved by the cross-check (added by maf):")
    assert limitations.endswith(UNRESOLVED)


def test_limitations_already_listing_the_issue_are_left_alone(ready: StageEnv) -> None:
    body = FINAL_BODY.format(run=RUN_ID).replace("## Limitations\n\nNone.", f"## Limitations\n\n{UNRESOLVED}")
    ready.fakes.claude.script(body)
    note = FinalBackend().run_stage(ready.ctx("final", round=2, unresolved_critical=1)).notes[0].handoff
    assert note.section("Limitations") == UNRESOLVED


def test_rerun_is_idempotent(ready: StageEnv) -> None:
    ready.fakes.claude.script(FINAL_BODY.format(run=RUN_ID), FINAL_BODY.format(run=RUN_ID))
    first = FinalBackend().run_stage(ready.ctx("final", round=2)).notes[0].handoff
    second = FinalBackend().run_stage(ready.ctx("final", round=2)).notes[0].handoff
    assert first.sections == second.sections
    assert sorted(p.name for p in ready.paths.deliverables.iterdir()) == ["document.md", "plots", "src"]


def test_oversized_trees_stay_in_the_workspace(ready: StageEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(final_mod, "MAX_DELIVERABLE_FILES", 0)
    ready.fakes.claude.script(FINAL_BODY.format(run=RUN_ID))
    output = FinalBackend().run_stage(ready.ctx("final", round=2))
    assert not (ready.paths.deliverables / "src").exists()
    assert ready.paths.workspace.resolve() / "src" not in output.deliverables
    listed = output.notes[0].handoff.section("Deliverables")
    assert f"`{ready.paths.workspace.resolve() / 'src'}` (left in the workspace" in listed


def test_uses_round_one_notes_when_no_loop(stage_env: StageEnv, sample_bodies: dict[str, str]) -> None:
    stage_env.put("02-strategy", HandoffKind.STRATEGY, sample_bodies["strategy"], from_="chatgpt")
    stage_env.put("03-execution", HandoffKind.EXECUTION, sample_bodies["execution"])
    stage_env.put("04-crosscheck", HandoffKind.CROSSCHECK, sample_bodies["crosscheck"], from_="maf")
    stage_env.fakes.claude.script(sample_bodies["final"])
    output = FinalBackend().run_stage(stage_env.ctx("final"))
    note = output.notes[0].handoff
    assert note.meta.inputs == ["[[02-strategy]]", "[[03-execution]]", "[[04-crosscheck]]"]
    assert output.deliverables == []  # the sample execution's artifacts do not exist in the workspace
    assert "None." in prompt_of(stage_env.fakes.claude.calls[0])


def test_final_repair_then_failure(ready: StageEnv) -> None:
    ready.fakes.claude.script("nope", "still nope")
    with pytest.raises(HandoffInvalid):
        FinalBackend().run_stage(ready.ctx("final", round=2))


def test_unresolved_lines(sample_bodies: dict[str, str], stage_env: StageEnv) -> None:
    passed = stage_env.put("04-crosscheck", HandoffKind.CROSSCHECK, sample_bodies["crosscheck"], from_="maf")
    assert unresolved_lines(passed) == []
    body = sample_bodies["crosscheck"].replace("## Unresolved Critical\n\nNone.", f"## Unresolved Critical\n\n{UNRESOLVED}\nprose")
    looped = stage_env.put("04-crosscheck-r2", HandoffKind.CROSSCHECK, body, round=2, from_="maf")
    assert unresolved_lines(looped) == [UNRESOLVED]
