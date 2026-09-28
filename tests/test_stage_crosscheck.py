"""Tests for the cross-check stage: concurrent critiques, rebuttal, adjudication, fixes and the Python verdict."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from maf.handoff import HandoffInvalid, HandoffKind, Issue, Response, Ruling
from maf.providers import CompletionRequest, ProviderError
from maf.stages import crosscheck as cc
from maf.stages.crosscheck import (
    FIX_REPORT_SCHEMA,
    PROSE_FIX_REPORT_SCHEMA,
    CrosscheckBackend,
    collect_artifact_text,
    issues_to_fix,
    parse_fix_report,
    unresolved_critical,
)
from maf.types import Usage
from test_stages_base import StageEnv, prompt_of, stage_env  # noqa: F401


def critique(prefix: str, *issues: str) -> str:
    lines = "\n".join(f"- [{sev}] {prefix}-{n}: {text}" for n, (sev, text) in enumerate(_split(issues), start=1))
    return f"## Summary\n\nReviewed.\n\n## Issues\n\n{lines or 'None.'}\n"


def _split(issues: tuple[str, ...]) -> list[tuple[str, str]]:
    return [tuple(i.split(":", 1)) for i in issues]  # type: ignore[misc]


def rebuttal(*responses: str) -> str:
    return "## Summary\n\nAnswered.\n\n## Responses\n\n" + ("\n".join(responses) or "None.") + "\n"


def adjudication(*rulings: str) -> str:
    return "## Summary\n\nRuled.\n\n## Rulings\n\n" + "\n".join(rulings) + "\n"


@pytest.fixture
def ready(stage_env: StageEnv, sample_bodies: dict[str, str]) -> StageEnv:
    stage_env.put("02-strategy", HandoffKind.STRATEGY, sample_bodies["strategy"], from_="chatgpt")
    stage_env.put("03-execution", HandoffKind.EXECUTION, sample_bodies["execution"])
    stage_env.workspace_file("src/alloc.c", "void *tlsf_malloc(unsigned n);\n")
    stage_env.workspace_file("test/posix_test.log", "42 tests passed\n")
    return stage_env


def script_disputed_round(env: StageEnv, fix_report: dict | str) -> None:
    env.fakes.chatgpt.script(
        critique("GPT", "critical:free() lacks a range check", "minor:README lacks an example"),
        adjudication("- GPT-1 [fix]: cheap and required", "- GEM-1 [wontfix]: out of scope"),
    )
    env.fakes.gemini.script(critique("GEM", "major:no Cortex-M0 support"))
    env.fakes.claude.script(
        critique("CLA"),
        rebuttal(
            "- GPT-1 [reject]: caller responsibility",
            "- GPT-2 [accept]: will add",
            "- GEM-1 [partial]: documented only",
        ),
    )
    env.fakes.claude_code.script(fix_report)


def test_disputed_round_passes_when_fixes_are_confirmed(ready: StageEnv) -> None:
    script_disputed_round(ready, {"fixed": ["GPT-1", "GPT-2", "BOGUS-1"], "not_fixed": [], "summary": "Added checks.\n## x"})
    output = CrosscheckBackend().run_stage(ready.ctx("crosscheck", mode="code"))

    assert [n.name for n in output.notes] == [
        "04a-critique-chatgpt", "04a-critique-gemini", "04a-critique-claude",
        "04b-rebuttal", "04c-adjudication", "04-crosscheck",
    ]
    assert output.loop_back is False
    assert output.index_updates == {"unresolved_critical": 0}

    froms = [(n.handoff.meta.from_, n.handoff.meta.to) for n in output.notes]
    assert froms == [
        ("chatgpt", "claude"), ("gemini", "claude"), ("claude", "claude"),
        ("claude", "chatgpt"), ("chatgpt", "claude"), ("maf", "final"),
    ]
    final = output.notes[-1].handoff
    assert final.section("Verdict") == "PASS"
    assert final.section("Unresolved Critical") == "None."
    assert final.section("Issues").splitlines() == [
        "- [critical] GPT-1: free() lacks a range check",
        "- [minor] GPT-2: README lacks an example",
        "- [major] GEM-1: no Cortex-M0 support",
    ]
    assert final.section("Rulings").splitlines() == ["- GPT-1 [fix]: cheap and required", "- GEM-1 [wontfix]: out of scope"]
    assert final.section("Applied Fixes").splitlines() == ["- GPT-1: fixed", "- GPT-2: fixed"]
    summary = final.section("Summary")
    assert "3 issue(s) raised (1 critical, 1 major, 1 minor), 2 disputed, 2 to fix, 2 confirmed fixed" in summary
    assert "Fixer's summary: Added checks. ## x" in summary  # flattened: cannot open a section
    assert "x" not in final.sections
    assert final.meta.cost_usd == pytest.approx(0.50)  # the fix pass
    assert "[[04c-adjudication]]" in final.meta.inputs

    # The fixer got exactly the issues to fix, with the dispute context, and a JSON schema.
    (fix_request,) = ready.fakes.claude_code.calls
    assert fix_request.json_schema == FIX_REPORT_SCHEMA
    assert fix_request.max_budget_usd == pytest.approx(ready.settings.output_limits.claude_code_budget_usd)
    fix_prompt = prompt_of(fix_request)
    assert "GPT-1" in fix_prompt and "GPT-2" in fix_prompt and "GEM-1" not in fix_prompt
    assert "Adjudicator ruling [fix]: cheap and required" in fix_prompt
    assert "<artifact" not in fix_prompt  # Claude Code reads the workspace itself
    assert (ready.paths.workspace / ".maf" / "fixes-r1.md").read_text() == fix_prompt

    # Adjudication saw only the disputed issues; critics saw the artifacts and the acceptance criteria.
    adjudication_prompt = prompt_of(ready.fakes.chatgpt.calls[1])
    assert "### GPT-1 [critical]" in adjudication_prompt and "### GEM-1 [major]" in adjudication_prompt
    assert "### GPT-2" not in adjudication_prompt
    critique_prompt = prompt_of(ready.fakes.gemini.calls[0])
    assert '<artifact path="src/alloc.c">' in critique_prompt and "42 tests passed" in critique_prompt
    assert "## Acceptance Criteria" in critique_prompt and "## Risks" not in critique_prompt
    assert "`GEM`" in critique_prompt


def test_unfixed_critical_loops_back(ready: StageEnv) -> None:
    script_disputed_round(
        ready, {"fixed": ["GPT-2"], "not_fixed": [{"id": "GPT-1", "reason": "breaks\nthe 2 KB limit"}], "summary": ""}
    )
    output = CrosscheckBackend().run_stage(ready.ctx("crosscheck", mode="code"))
    final = output.notes[-1].handoff
    assert output.loop_back is True
    assert output.index_updates == {"unresolved_critical": 1}
    assert final.section("Verdict") == "LOOP"
    assert final.meta.to == "execution"
    assert final.section("Unresolved Critical") == "- [critical] GPT-1: free() lacks a range check"
    assert "- GPT-1: not fixed - breaks the 2 KB limit" in final.section("Applied Fixes")


def test_no_issues_skips_rebuttal_adjudication_and_fixes(ready: StageEnv) -> None:
    ready.fakes.chatgpt.script(critique("GPT"))
    ready.fakes.gemini.script(critique("GEM"))
    ready.fakes.claude.script(critique("CLA"))
    output = CrosscheckBackend().run_stage(ready.ctx("crosscheck", mode="code"))
    assert [len(f.calls) for f in ready.fakes.all()] == [1, 1, 1, 0]
    rebuttal_note, adjudication_note, final = (n.handoff for n in output.notes[3:])
    assert rebuttal_note.meta.from_ == "maf" and rebuttal_note.section("Responses") == "None."
    assert adjudication_note.meta.from_ == "maf" and adjudication_note.section("Rulings") == "None."
    assert final.section("Verdict") == "PASS"
    assert final.section("Applied Fixes") == "None."
    assert final.meta.model == "none" and final.meta.cost_usd == 0


def test_all_accepted_skips_adjudication_and_unanswered_counts_as_accept(ready: StageEnv) -> None:
    ready.fakes.chatgpt.script(critique("GPT", "critical:bug one", "major:bug two"))
    ready.fakes.gemini.script(critique("GEM"))
    ready.fakes.claude.script(critique("CLA"), rebuttal("- GPT-1 [accept]: yes"))
    ready.fakes.claude_code.script({"fixed": ["GPT-1", "GPT-2"], "not_fixed": [], "summary": "done"})
    output = CrosscheckBackend().run_stage(ready.ctx("crosscheck", mode="code"))
    assert len(ready.fakes.chatgpt.calls) == 1
    assert output.notes[4].handoff.meta.from_ == "maf"
    assert "GPT-2" in prompt_of(ready.fakes.claude_code.calls[0])
    assert output.notes[-1].handoff.section("Verdict") == "PASS"


def test_critique_with_foreign_prefix_is_repaired(ready: StageEnv) -> None:
    ready.fakes.chatgpt.script(critique("CLA", "minor:x"), critique("GPT", "minor:x"))
    ready.fakes.gemini.script(critique("GEM"))
    ready.fakes.claude.script(critique("CLA"), rebuttal("- GPT-1 [accept]: ok"))
    ready.fakes.claude_code.script({"fixed": ["GPT-1"], "not_fixed": [], "summary": ""})
    output = CrosscheckBackend().run_stage(ready.ctx("crosscheck", mode="code"))
    assert "must use your own ID prefix GPT-" in prompt_of(ready.fakes.chatgpt.calls[1])
    assert output.notes[0].handoff.meta.cost_usd == pytest.approx(0.02)


def test_rebuttal_with_unknown_id_is_repaired(ready: StageEnv) -> None:
    ready.fakes.chatgpt.script(critique("GPT", "minor:x"))
    ready.fakes.gemini.script(critique("GEM"))
    ready.fakes.claude.script(
        critique("CLA"),
        rebuttal("- GPT-1 [accept]: ok", "- GEM-9 [reject]: no"),
        rebuttal("- GPT-1 [accept]: ok"),
    )
    ready.fakes.claude_code.script({"fixed": ["GPT-1"], "not_fixed": [], "summary": ""})
    CrosscheckBackend().run_stage(ready.ctx("crosscheck", mode="code"))
    assert "unknown issue ID GEM-9" in prompt_of(ready.fakes.claude.calls[2])


def test_round_two_note_names(ready: StageEnv, sample_bodies: dict[str, str]) -> None:
    ready.put("03-execution-r2", HandoffKind.EXECUTION, sample_bodies["execution"], round=2)
    ready.fakes.chatgpt.script(critique("GPT"))
    ready.fakes.gemini.script(critique("GEM"))
    ready.fakes.claude.script(critique("CLA"))
    output = CrosscheckBackend().run_stage(ready.ctx("crosscheck", mode="code", round=2))
    assert [n.name for n in output.notes] == [
        "04a-critique-chatgpt-r2", "04a-critique-gemini-r2", "04a-critique-claude-r2",
        "04b-rebuttal-r2", "04c-adjudication-r2", "04-crosscheck-r2",
    ]
    assert all(n.handoff.meta.round == 2 for n in output.notes)
    assert "[[03-execution-r2]]" in output.notes[-1].handoff.meta.inputs


def test_prose_fix_rewrites_the_document(ready: StageEnv) -> None:
    ready.workspace_file("document.md", "# Report\n\nOld.\n")
    ready.fakes.chatgpt.script(critique("GPT", "critical:wrong equation"))
    ready.fakes.gemini.script(critique("GEM"))
    ready.fakes.claude.script(
        critique("CLA"),
        rebuttal("- GPT-1 [accept]: yes"),
        {"fixed": ["GPT-1"], "not_fixed": [], "summary": "fixed eq", "document": "# Report\n\nNew.\n"},
    )
    output = CrosscheckBackend().run_stage(ready.ctx("crosscheck", mode="prose"))
    fix_request = ready.fakes.claude.calls[-1]
    assert fix_request.json_schema == PROSE_FIX_REPORT_SCHEMA
    assert fix_request.max_output_tokens == ready.settings.output_limits.claude
    assert ready.fakes.claude_code.calls == []
    assert (ready.paths.workspace / "document.md").read_text() == "# Report\n\nNew.\n"
    assert output.notes[-1].handoff.section("Verdict") == "PASS"


def test_unreadable_fix_report_is_fail_safe(ready: StageEnv) -> None:
    ready.fakes.chatgpt.script(critique("GPT", "critical:bug"))
    ready.fakes.gemini.script(critique("GEM"))
    ready.fakes.claude.script(critique("CLA"), rebuttal("- GPT-1 [accept]: yes"))
    ready.fakes.claude_code.script("I fixed everything, trust me")
    output = CrosscheckBackend().run_stage(ready.ctx("crosscheck", mode="code"))
    assert output.loop_back is True
    final = output.notes[-1].handoff
    assert "could not be parsed" in final.section("Applied Fixes")
    assert "could not be parsed" in final.section("Summary")


def test_critiques_run_concurrently_and_failures_propagate_after_all_finish(ready: StageEnv) -> None:
    barrier = threading.Barrier(3, timeout=5)

    def waiting(body: str):  # type: ignore[no-untyped-def]
        def reply(_request: CompletionRequest) -> str:
            barrier.wait()  # deadlocks (and times out) unless all three critics run at once
            return body

        return reply

    def failing(_request: CompletionRequest) -> str:
        barrier.wait()
        raise ProviderError("gemini down", provider="gemini")

    ready.fakes.chatgpt.script(waiting(critique("GPT")))
    ready.fakes.gemini.script(failing)
    ready.fakes.claude.script(waiting(critique("CLA")))
    with pytest.raises(ProviderError, match="gemini down"):
        CrosscheckBackend().run_stage(ready.ctx("crosscheck", mode="code"))
    assert sorted(e.agent for e in ready.ledger.entries) == ["chatgpt", "claude"]


def test_mode_is_required(ready: StageEnv) -> None:
    with pytest.raises(ValueError):
        CrosscheckBackend().run_stage(ready.ctx("crosscheck"))


# ---------------------------------------------------------------------- pure helpers


def _issue(id: str, severity: str = "critical") -> Issue:
    return Issue(id=id, severity=severity, text=f"text {id}", raised_by="chatgpt")  # type: ignore[arg-type]


def test_issues_to_fix() -> None:
    issues = [_issue("GPT-1"), _issue("GPT-2"), _issue("GPT-3"), _issue("GPT-4"), _issue("GPT-5"), _issue("GPT-6")]
    responses = [
        Response(id="GPT-1", stance="accept", text="a"),
        Response(id="GPT-2", stance="reject", text="r"),
        Response(id="GPT-3", stance="partial", text="p"),
        Response(id="GPT-4", stance="reject", text="r"),
        Response(id="GPT-6", stance="partial", text="p"),
    ]
    rulings = [
        Ruling(id="GPT-2", ruling="wontfix", text="w"),
        Ruling(id="GPT-3", ruling="fix", text="f"),
        Ruling(id="GPT-6", ruling="wontfix", text="w"),
        Ruling(id="GPT-1", ruling="wontfix", text="accepted issues ignore rulings"),
    ]
    # GPT-4: disputed without a ruling -> fix; GPT-5: unanswered -> accept.
    assert [i.id for i in issues_to_fix(issues, responses, rulings)] == ["GPT-1", "GPT-3", "GPT-4", "GPT-5"]
    assert issues_to_fix([], responses, rulings) == []


def test_unresolved_critical() -> None:
    to_fix = [_issue("GPT-1"), _issue("GPT-2"), _issue("GPT-3"), _issue("GEM-1", "major")]
    unresolved = unresolved_critical(to_fix, fixed=["GPT-1", "GPT-3"], not_fixed=["GPT-3", "GEM-1"])
    assert [i.id for i in unresolved] == ["GPT-2", "GPT-3"]
    assert unresolved_critical([], ["x"], []) == []


def test_parse_fix_report_variants() -> None:
    from maf.providers import CompletionResult

    def result(text: str, parsed: dict | None = None) -> CompletionResult:
        return CompletionResult(text=text, parsed=parsed, usage=Usage(), cost_usd=0, model="m", provider="claude_code")

    ok = parse_fix_report(result('{"fixed": ["A-1"], "not_fixed": [], "summary": "s"}'), ["A-1"])
    assert ok.fixed == ["A-1"]
    bad = parse_fix_report(result('{"fixed": "A-1"}'), ["A-1", "B-2"])
    assert bad.fixed == [] and [n.id for n in bad.not_fixed] == ["A-1", "B-2"]
    assert parse_fix_report(result("[1, 2]"), []).fixed == []


def test_collect_artifact_text(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "src" / ".git").mkdir(parents=True)
    (ws / "src" / "a.c").write_text("int a;\n")
    (ws / "src" / "b.h").write_text("``` backticks ````\n")
    (ws / "src" / ".git" / "HEAD").write_text("ref")
    (ws / "src" / "blob.bin").write_bytes(b"\x00\x01")
    (ws / "src" / "latin.txt").write_bytes(b"\xff\xfe bad utf8")
    (ws / "big.log").write_text("x" * (cc.MAX_ARTIFACT_FILE_BYTES + 1))
    text = collect_artifact_text(ws, ["src/", "src/a.c", "big.log", "gone.txt", "../escape"])
    assert text.count('<artifact path="src/a.c">') == 1
    assert '<artifact path="src/b.h">\n`````\n``` backticks ````\n\n`````' in text
    assert ".git" not in text
    assert "`src/blob.bin`: binary file" in text and "`src/latin.txt`: binary file" in text
    assert "`big.log`: not shown (larger than 200 KB)" in text
    assert "`gone.txt`: listed but not found" in text
    assert "`../escape`: not shown" in text


def test_collect_artifact_text_budget(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    for name in ("1.txt", "2.txt", "3.txt"):
        (ws / name).write_text("y" * 60)
    text = collect_artifact_text(ws, ["1.txt", "2.txt", "3.txt"], budget_chars=100)
    assert "y" * 60 in text
    assert "[... truncated: artifact budget exhausted ...]" in text
    assert text.count("y") == 100
    assert "1 more file(s) not shown" in text
    assert collect_artifact_text(ws, []) == ""


def test_critique_invalid_twice_stops_the_stage(ready: StageEnv) -> None:
    ready.fakes.chatgpt.script("junk", "junk")
    ready.fakes.gemini.script(critique("GEM"))
    ready.fakes.claude.script(critique("CLA"))
    with pytest.raises(HandoffInvalid):
        CrosscheckBackend().run_stage(ready.ctx("crosscheck", mode="code"))
