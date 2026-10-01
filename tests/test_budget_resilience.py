"""Budget and timeout resilience, from the thesis rerun of 2026-09-29 (run 2026-09-29-produce-a-master-s-thesis-...).

That run failed twice for want of money or time while its workspace held nearly finished work: execution round 1
timed out at 60 minutes (charged its $10.28 worst case), and execution round 3 got a Claude Code budget clamped to
$1.43, ran out of it and ended FAILED with nothing exported. Ten hard acceptance criteria, several of them absolute
provenance demands, drove two expensive cross-check loops before that. These tests cover the fixes:

- budget exhaustion under a clamp is BUDGET_EXCEEDED, and a session below its minimum budget is never started;
- a timed-out execution or fix session gets one continuation, a second timeout fails the run;
- the strategy may set at most six hard criteria, and the adjudicator may relax an over-specified one;
- a LOOP that the budget cannot pay for goes to final instead;
- a failed or budget-stopped code run still exports its workspace, marked partial.

Unit tests first, then end-to-end runs through the real pipeline with scripted fakes (no network, no subprocess).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import date

import frontmatter
import pytest
from conftest import FakeProvider, FakeProviders
from test_e2e import (
    ACCEPTANCE_MET,
    CLEAN_CRITIQUE,
    EXPECTED_HANDOFFS,
    BRIEF,
    Script,
    _assert_ledger_mirrored,
    _code_kinds,
    _is_repair,
    _kind_of,
    _sandboxed_pipeline,
    _start,
    _total_calls,
)
from test_stages_base import RUN_ID, SandboxedFake, StageEnv, prompt_of, stage_env  # noqa: F401

from maf import cli
from maf import handoff as hf
from maf import ledger as _ledger
from maf.config import DEFAULT_MCP_MAX_BUDGET_USD, Settings
from maf.handoff import HandoffInvalid, HandoffKind, Issue, Relaxation, Ruling
from maf.ledger import BudgetExceeded, Ledger, LedgerEntry, SessionBudgetExhausted, SessionBudgetTooSmall
from maf.providers import CompletionRequest, CompletionResult
from maf.providers.claude_code import ClaudeCodeBudgetExhausted, ClaudeCodeTimeout, preflight_budget_usd
from maf.stages import base
from maf.stages import crosscheck as cc
from maf.stages import final as final_mod
from maf.stages.base import CONTINUATION_PREAMBLE, CONTINUED_KEY, continuation_note, work_session
from maf.stages.crosscheck import CrosscheckBackend, LoopEstimate, round_costs
from maf.stages.execution import ExecutionBackend
from maf.stages.final import FinalBackend
from maf.stages.strategy import StrategyBackend, budget_note, criteria_guidance
from maf.types import RunStatus
from maf.vault import PARTIAL_EXPORT_NOTE, RunIndex, describe_issues, render_run_body

TIMEOUT = "Claude Code timed out after 5400s"


def _timeout(cost: float = 2.0) -> ClaudeCodeTimeout:
    """What ``ClaudeCodeProvider.complete`` raises when the CLI is killed: the worst case is charged."""
    return ClaudeCodeTimeout(TIMEOUT, provider="claude_code", cost_usd=cost)


def _exhausted(cost: float) -> ClaudeCodeBudgetExhausted:
    return ClaudeCodeBudgetExhausted(
        "Claude Code failed (exit 1, subtype=error_max_budget_usd): no detail", provider="claude_code", cost_usd=cost
    )


def _code_request(
    budget: float | None = 8.0, prompt: str = "# Task: produce the definitive artifacts"
) -> CompletionRequest:
    return CompletionRequest.simple("claude-opus-5-5", prompt, max_output_tokens=64_000, max_budget_usd=budget)


@dataclass
class HeadroomFake(FakeProvider):
    """A ``claude_code`` fake with the real provider's budget shape: one turn of ``headroom`` held back from the
    clamp, and a worst case of budget plus that turn."""

    headroom: float = 1.0

    def turn_headroom_usd(self, request: CompletionRequest, on: date | None = None) -> float:
        return self.headroom

    def worst_case_cost(self, request: CompletionRequest, on: date | None = None) -> float:
        assert request.max_budget_usd is not None
        return request.max_budget_usd + self.headroom


# ============================================================================= ledger


def test_clamp_budget_is_what_metered_call_passes() -> None:
    assert _ledger.clamp_budget(10.0, 2.0, 8.0) == 8.0
    assert _ledger.clamp_budget(5.0, 2.0, 8.0) == 3.0
    assert _ledger.clamp_budget(1.0, 2.0, 8.0) == 0.0
    assert _ledger.clamp_budget(5.0, 2.0, None) == 3.0


def test_planned_budget_holds_back_the_turn_headroom_and_in_flight_reservations() -> None:
    ledger, fake = Ledger(RUN_ID, 10.0), HeadroomFake(name="claude_code", agent="claude", headroom=1.5)
    assert _ledger.planned_budget(ledger, fake, _code_request(8.0)) == (8.0, 1.5)
    token = ledger._reserve(4.0, "a concurrent call")
    assert _ledger.planned_budget(ledger, fake, _code_request(8.0)) == (4.5, 1.5)
    ledger._release(token)


def test_session_budget_errors_are_budget_exceeded_and_say_how_much_is_missing() -> None:
    small = SessionBudgetTooSmall(
        cap_usd=38.0, spent_usd=34.33, what="execution/execution via claude_code:m", budget_usd=1.39,
        minimum_usd=3.0, headroom_usd=2.28,
    )
    assert isinstance(small, BudgetExceeded)
    assert small.shortfall_usd == pytest.approx(1.61)
    text = str(small)
    assert "$3.67 is left" in text and "one turn ($2.28)" in text and "would get $1.39" in text
    assert "below the $3.00 a work session needs" in text and "this session was not started" in text
    assert "at least $1.61 more: resume with --budget 39.61 or higher" in text
    assert "nothing was spent" not in text  # the stage may have paid for earlier calls
    exhausted = SessionBudgetExhausted(cap_usd=38.0, spent_usd=35.76, what="w", budget_usd=1.43, requested_usd=8.0)
    assert isinstance(exhausted, BudgetExceeded)
    assert "stopped at its $1.43 cap, which the run's remaining budget had cut from $8.00" in str(exhausted)


def test_timed_out_claude_code_call_is_charged_its_worst_case() -> None:
    """The cap stays hard: the ledger records the timeout's worst-case charge, with the error."""
    ledger = Ledger(RUN_ID, 25.0)
    fake = HeadroomFake(name="claude_code", agent="claude", headroom=2.28)
    fake.script(_timeout(cost=8.0 + 2.28))
    with pytest.raises(ClaudeCodeTimeout):
        _ledger.metered_call(ledger, fake, _code_request(8.0), stage="execution", purpose="execution")
    (entry,) = ledger.entries
    assert entry.cost_usd == entry.worst_case_usd == pytest.approx(10.28)
    assert entry.error == f"ClaudeCodeTimeout: {TIMEOUT}"


# ============================================================================= work sessions


def _full_minimum(ctx: base.StageContext) -> base.StageContext:
    """The minimum at its full $3 whatever the run's cap (no share of a small cap)."""
    ctx.settings = ctx.settings.model_copy(update={"claude_code_min_session_share": 1.0})
    return ctx


def test_work_session_below_its_minimum_is_refused_before_spawning(stage_env: StageEnv) -> None:
    fake = HeadroomFake(name="claude_code", agent="claude", headroom=1.0)
    stage_env.fakes.claude_code = fake
    ctx = _full_minimum(stage_env.ctx("execution", mode="code"))
    ctx.ledger = Ledger(RUN_ID, 3.5)

    with pytest.raises(SessionBudgetTooSmall) as info:
        work_session(ctx, _code_request(8.0), purpose="execution")

    assert fake.calls == [] and ctx.ledger.entries == ()  # nothing spawned, nothing spent
    assert (info.value.budget_usd, info.value.minimum_usd) == (pytest.approx(2.5), 3.0)
    assert "at least $0.50 more: resume with --budget 4.00 or higher" in str(info.value)


def test_execution_refuses_a_too_small_session_before_paying_for_the_preflight(
    stage_env: StageEnv, sample_bodies: dict[str, str]
) -> None:
    """$3.20 left covers the $3 minimum, but not after the preflight's budget: refused up front, and the hint covers
    the preflight a resumed process pays again."""
    stage_env.put("01-ingestion", HandoffKind.INGESTION, sample_bodies["ingestion"], from_="gemini")
    stage_env.put("02-strategy", HandoffKind.STRATEGY, sample_bodies["strategy"], from_="chatgpt")
    code = SandboxedFake(name="claude_code", agent="claude", workspace=stage_env.paths.workspace)
    stage_env.fakes.claude_code = code
    ctx = _full_minimum(stage_env.ctx("execution", mode="code"))
    ctx.ledger = Ledger(RUN_ID, 3.2)
    preflight = preflight_budget_usd(ctx.model("claude_code"))
    assert base.preflight_reserve(ctx) == pytest.approx(preflight) and 0.2 < preflight < 3.2

    with pytest.raises(SessionBudgetTooSmall) as info:
        ExecutionBackend().run_stage(ctx)

    assert code.calls == [] and ctx.ledger.entries == ()  # no preflight, no session, nothing spent
    assert info.value.budget_usd == pytest.approx(3.2 - preflight)
    assert info.value.needed_cap_usd == pytest.approx(3.0 + preflight, abs=0.01)
    assert f"(a resumed stage first pays up to ${preflight:.2f} for the sandbox preflight)" in str(info.value)


def test_work_session_minimum_is_configurable_and_never_above_the_sessions_own_budget(stage_env: StageEnv) -> None:
    ctx = stage_env.ctx("execution", mode="code")
    ctx.ledger = Ledger(RUN_ID, 2.5)
    stage_env.fakes.claude_code.script("ok", "ok")
    assert work_session(ctx, _code_request(2.0), purpose="execution").text == "ok"  # asks for $2, gets $2
    ctx.settings = ctx.settings.model_copy(update={"claude_code_min_session_usd": 1.0})
    assert work_session(ctx, _code_request(8.0), purpose="execution").text == "ok"  # $1.x left meets a $1 minimum


def test_budget_exhaustion_under_a_clamp_is_budget_exceeded(stage_env: StageEnv) -> None:
    ctx = stage_env.ctx("execution", mode="code")
    ctx.ledger = Ledger(RUN_ID, 5.0)  # $8 asked, $5 available: the clamp cuts the session to $5
    stage_env.fakes.claude_code.script(_exhausted(4.9))

    with pytest.raises(SessionBudgetExhausted) as info:
        work_session(ctx, _code_request(8.0), purpose="execution")

    assert isinstance(info.value.__cause__, ClaudeCodeBudgetExhausted)
    assert info.value.budget_usd == pytest.approx(5.0)
    (entry,) = ctx.ledger.entries
    assert entry.cost_usd == pytest.approx(4.9) and entry.error.startswith("ClaudeCodeBudgetExhausted")


def test_budget_exhaustion_of_an_unclamped_session_still_fails(stage_env: StageEnv) -> None:
    ctx = stage_env.ctx("execution", mode="code")
    stage_env.fakes.claude_code.script(_exhausted(8.0))
    with pytest.raises(ClaudeCodeBudgetExhausted):
        work_session(ctx, _code_request(8.0), purpose="execution")


def test_timeout_gets_one_continuation_in_the_same_workspace(stage_env: StageEnv) -> None:
    ctx = stage_env.ctx("execution", round=2, mode="code")
    stage_env.fakes.claude_code.script(_timeout(), "finished")
    original = "# Task: produce the definitive artifacts (round 2)\n\nBuild it."

    result = work_session(ctx, _code_request(8.0, original), purpose="execution")

    assert result.text == "finished"
    assert result.cost_usd == pytest.approx(2.0 + 0.5)  # the timed-out charge is carried into the note's cost
    assert result.raw[CONTINUED_KEY] == {"stopped": TIMEOUT, "charged_usd": 2.0}
    first, second = stage_env.fakes.claude_code.calls
    assert prompt_of(first) == original
    continued = prompt_of(second)
    assert continued.startswith("# Continuation: the previous session was cut off")
    assert continued.endswith(original) and TIMEOUT in continued
    for rule in ("inspect the workspace", "Finish only the incomplete parts", "Do not restart",
                 "Run the reproduction command and the tests", "under 10 minutes"):
        assert rule in continued
    assert second.max_budget_usd == first.max_budget_usd == 8.0
    assert (ctx.paths.workspace / ".maf" / "execution-r2-continuation.md").read_text() == continued
    assert [(e.purpose, e.cost_usd, e.error.split(":")[0]) for e in ctx.ledger.entries] == [
        ("execution", 2.0, "ClaudeCodeTimeout"),
        ("execution-continuation", 0.5, ""),
    ]
    assert continuation_note(result) == (
        f"maf: the first Claude Code session of this pass was stopped ({TIMEOUT}) and charged its worst case "
        "($2.00); one continuation session in the same workspace finished the pass."
    )
    assert continuation_note(result.model_copy(update={"raw": {}})) == ""


def test_a_second_timeout_fails_with_a_clear_message(stage_env: StageEnv) -> None:
    ctx = stage_env.ctx("execution", mode="code")
    stage_env.fakes.claude_code.script(_timeout(), _timeout())
    with pytest.raises(ClaudeCodeTimeout) as info:
        work_session(ctx, _code_request(), purpose="execution")
    assert str(info.value) == (
        f"the Claude Code execution session and its one continuation both timed out ({TIMEOUT}; then {TIMEOUT}); "
        "their work stays in the workspace, and maf resume runs execution again"
    )
    assert info.value.cost_usd == 0.0  # both sessions are already in the ledger
    assert [e.cost_usd for e in ctx.ledger.entries] == [2.0, 2.0]


def test_continuation_respects_the_minimum_budget(stage_env: StageEnv) -> None:
    ctx = _full_minimum(stage_env.ctx("execution", mode="code"))
    ctx.ledger = Ledger(RUN_ID, 4.0)  # the timed-out session is charged $2, leaving less than the $3 minimum
    stage_env.fakes.claude_code.script(_timeout())
    with pytest.raises(SessionBudgetTooSmall):
        work_session(ctx, _code_request(), purpose="execution")
    assert len(stage_env.fakes.claude_code.calls) == 1


def test_without_continuation_a_timeout_propagates(stage_env: StageEnv) -> None:
    ctx = stage_env.ctx("execution", mode="code")
    stage_env.fakes.claude_code.script(_timeout())
    with pytest.raises(ClaudeCodeTimeout, match="timed out after"):
        work_session(ctx, _code_request(), purpose="repair", minimum_usd=0.0, continue_on_timeout=False)


def test_continuation_preamble_has_only_the_reason_field() -> None:
    assert CONTINUATION_PREAMBLE.format(reason="x").count("x") >= 1
    assert base.continuation_prompt("TASK", "why\nnot").endswith("---\n\nTASK")


# ============================================================================= criteria calibration


def _criteria(hard: int, soft: int = 0) -> str:
    lines = [f"- AC-{n} [hard]: check {n} passes (`make check-{n}`)" for n in range(1, hard + 1)]
    lines += [f"- AC-{n} [soft]: nice {n}" for n in range(hard + 1, hard + soft + 1)]
    return "\n".join(lines)


def test_hard_criteria_are_capped_at_six() -> None:
    assert hf.MAX_HARD_CRITERIA == 6
    assert hf.hard_criteria_errors(_criteria(6, soft=4)) == []
    (error,) = hf.hard_criteria_errors(_criteria(7))
    assert "has 7 hard criteria (AC-1, AC-2, AC-3, AC-4, AC-5, AC-6, AC-7); at most 6 may be hard" in error
    assert "mark the others [soft]" in error
    assert hf.hard_criteria_errors("- unlabeled one\n" * 7) != []  # unlabeled criteria count as hard
    assert "At most 6 criteria may be `hard`" in hf.format_spec(HandoffKind.STRATEGY)


def test_relaxations_round_trip_and_parse_leniently() -> None:
    rule = Relaxation("AC-9", "GPT-4", 2, "the brief asks for simulated numbers, not a recomputation of each one")
    assert rule.line == "AC-9 (GPT-4, round 2): the brief asks for simulated numbers, not a recomputation of each one"
    parsed = hf.parse_relaxations([rule.line, "AC-6: hand-written reason", "AC-9 (CLA-1, round 3): later", "junk"])
    assert parsed == {"AC-9": rule, "AC-6": Relaxation("AC-6", "", 0, "hand-written reason")}
    criteria = hf.parse_acceptance_criteria(_criteria(2) + "\n- AC-9 [hard]: every number recomputed")
    assert [c.hard for c in hf.relaxed_criteria(criteria, parsed)] == [True, True, False]


def test_relax_is_a_ruling_of_the_adjudication_grammar() -> None:
    body = "## Summary\n\nRuled.\n\n## Rulings\n\n- GPT-4 [relax]: over-specified: the brief asks only for X\n"
    assert hf.validate_body(body, HandoffKind.ADJUDICATION) == []
    assert hf.parse_rulings("- GPT-4 [relax]: why") == [Ruling(id="GPT-4", ruling="relax", text="why")]
    assert cc.issues_to_fix(
        [Issue(id="GPT-4", severity="major", text="t", raised_by="chatgpt")],
        [hf.Response(id="GPT-4", stance="reject", text="no")],
        [Ruling(id="GPT-4", ruling="relax", text="why")],
    ) != []  # a relaxed issue is still worth improving, as a major one


@pytest.fixture
def with_ingestion(stage_env: StageEnv, sample_bodies: dict[str, str]) -> StageEnv:
    stage_env.put("01a-routing", HandoffKind.ROUTING, sample_bodies["routing"], from_="chatgpt")
    stage_env.put("01-ingestion", HandoffKind.INGESTION, sample_bodies["ingestion"], from_="gemini")
    return stage_env


def _strategy(sample: str, criteria: str) -> str:
    current = sample.split("## Acceptance Criteria\n\n", 1)[1].split("\n\n## Risks", 1)[0]
    return sample.replace(current, criteria)


def test_strategy_prompt_calibrates_the_criteria(with_ingestion: StageEnv, sample_bodies: dict[str, str]) -> None:
    with_ingestion.fakes.chatgpt.script(sample_bodies["strategy"])
    StrategyBackend().run_stage(with_ingestion.ctx("strategy", mode="mixed"))
    prompt = " ".join(prompt_of(with_ingestion.fakes.chatgpt.calls[0]).split())
    assert "At most 6 criteria are `hard`" in prompt
    assert "directly requires" in prompt and "within this run's budget" in prompt
    assert "Avoid absolute provenance or coverage demands" in prompt and "unless the request itself asks" in prompt
    assert "A hard criterion names how it is checked" in prompt
    # the budget the calibration refers to reaches the prompt: the run's and one work session's
    assert "This run's whole budget is $25.00: it pays for every model call of every stage" in prompt
    assert "including each cross-check loop (at most 2)" in prompt
    assert "Each Claude Code work session (an execution pass, or a cross-check's fix pass) may spend up to $8.00" in (
        prompt
    )
    prose = budget_note("prose", 38.0, with_ingestion.settings)
    assert prose.startswith("This run's whole budget is $38.00") and "Claude Code" not in prose
    assert "count toward the 6 hard criteria" in criteria_guidance("mixed", source_audit=True)
    assert "a soft criterion (hard only if the request asks" in criteria_guidance("mixed", source_audit=True)


def test_seven_hard_criteria_get_the_one_repair(with_ingestion: StageEnv, sample_bodies: dict[str, str]) -> None:
    seven = _strategy(sample_bodies["strategy"], _criteria(7))
    fixed = _strategy(sample_bodies["strategy"], _criteria(5, soft=2))
    with_ingestion.fakes.chatgpt.script(seven, fixed)

    output = StrategyBackend().run_stage(with_ingestion.ctx("strategy"))

    first, repair = with_ingestion.fakes.chatgpt.calls
    assert _is_repair(repair) and not _is_repair(first)
    assert "has 7 hard criteria" in prompt_of(repair) and "at most 6 may be hard" in prompt_of(repair)
    criteria = hf.parse_acceptance_criteria(output.notes[0].handoff)
    assert sum(c.hard for c in criteria) == 5
    assert output.notes[0].handoff.meta.cost_usd == pytest.approx(0.02)


def test_seven_hard_criteria_twice_fail_the_stage(with_ingestion: StageEnv, sample_bodies: dict[str, str]) -> None:
    seven = _strategy(sample_bodies["strategy"], _criteria(7))
    with_ingestion.fakes.chatgpt.script(seven, seven)
    with pytest.raises(HandoffInvalid, match="at most 6 may be hard"):
        StrategyBackend().run_stage(with_ingestion.ctx("strategy"))


def test_hand_edited_strategy_with_more_hard_criteria_stays_valid(sample_bodies: dict[str, str]) -> None:
    """Backward compatibility: the incident's 02-strategy has ten hard criteria; it still validates and parses."""
    body = _strategy(sample_bodies["strategy"], _criteria(10))
    assert hf.validate_body(body, HandoffKind.STRATEGY) == []
    assert len(hf.parse_acceptance_criteria(hf.split_sections(body)[2]["Acceptance Criteria"])) == 10


# ============================================================================= cross-check: relax and loop budget


def _critique(prefix: str, *issues: str) -> str:
    lines = "\n".join(f"- [{sev}] {prefix}-{n}: {text}" for n, (sev, text) in enumerate(
        (tuple(i.split(":", 1)) for i in issues), start=1
    ))
    return f"## Summary\n\nReviewed.\n\n## Issues\n\n{lines or 'None.'}\n"


OVERSPECIFIED = "- AC-3 [hard]: every computed number in every table is recomputed from saved outputs"
AC3_ISSUE = "critical:Unmet acceptance criterion AC-3: the tuning costs in Table 4 are not recomputed"


@pytest.fixture
def ready(stage_env: StageEnv, sample_bodies: dict[str, str]) -> StageEnv:
    criteria = sample_bodies["strategy"].split("## Acceptance Criteria\n\n", 1)[1].split("\n\n## Risks", 1)[0]
    body = _strategy(sample_bodies["strategy"], f"{criteria}\n{OVERSPECIFIED}")
    stage_env.put("02-strategy", HandoffKind.STRATEGY, body, from_="chatgpt")
    stage_env.put("03-execution", HandoffKind.EXECUTION, sample_bodies["execution"])
    stage_env.workspace_file("src/alloc.c", "void *tlsf_malloc(unsigned n);\n")
    stage_env.workspace_file("test/posix_test.log", "42 tests passed\n")
    return stage_env


def test_accepted_unmet_criterion_can_be_relaxed_and_stops_blocking(ready: StageEnv) -> None:
    ready.fakes.chatgpt.script(
        _critique("GPT", AC3_ISSUE),
        "## Summary\n\nRuled.\n\n## Rulings\n\n- GPT-1 [relax]: the brief asks that numbers come from simulations "
        "actually run, not that each table cell be recomputed\n",
    )
    ready.fakes.gemini.script(_critique("GEM"))
    ready.fakes.claude.script(
        _critique("CLA", "critical:Unmet acceptance criterion AC-3: stability counts not recomputed"),
        "## Summary\n\nAnswered.\n\n## Responses\n\n- GPT-1 [accept]: not done\n- CLA-1 [accept]: not done\n",
    )
    ready.fakes.claude_code.script({"fixed": [], "not_fixed": [{"id": "GPT-1", "reason": "too long"}], "summary": "s"})

    output = CrosscheckBackend().run_stage(ready.ctx("crosscheck", mode="code"))

    adjudication = prompt_of(ready.fakes.chatgpt.calls[1])
    assert "## Unmet acceptance criteria the author accepted" in adjudication
    assert "### GPT-1 [critical] (raised by chatgpt; an unmet acceptance criterion the author accepted)" in adjudication
    assert "`relax` applies only to an issue flagged as an unmet acceptance criterion" in adjudication
    assert ready.index.brief in adjudication
    assert "None: the author accepted every issue." in adjudication
    note = output.notes[-1].handoff
    assert output.loop_back is False and note.section("Verdict") == "PASS"
    assert output.index_updates == {
        "unresolved_critical": 0,
        "relaxed_criteria": [
            "AC-3 (GPT-1, round 1): the brief asks that numbers come from simulations actually run, not that each "
            "table cell be recomputed"
        ],
    }
    issues = note.section("Issues").splitlines()
    assert issues[0].startswith("- [major] GPT-1: Unmet acceptance criterion AC-3:")
    assert "maf: major, not critical: the adjudicator relaxed AC-3" in issues[0]
    assert issues[1].startswith("- [major] CLA-1:")  # the same criterion, raised by another critic
    assert note.section("Rulings").startswith("- GPT-1 [relax]: the brief asks")
    assert note.section(cc.RELAXED_SECTION).endswith(
        "- AC-3 (GPT-1, round 1): the brief asks that numbers come from simulations actually run, not that each "
        "table cell be recomputed"
    )
    assert "Relaxed this round (over-specified relative to the brief" in note.section("Summary")
    assert "Adjudicator ruling [relax]" in prompt_of(ready.fakes.claude_code.calls[0])  # the fixer is told
    assert hf.validate_handoff(note) == []


def test_relax_on_an_ordinary_issue_is_ignored_and_counts_as_fix(ready: StageEnv) -> None:
    ready.fakes.chatgpt.script(
        _critique("GPT", "critical:free() lacks a range check"),
        "## Summary\n\nRuled.\n\n## Rulings\n\n- GPT-1 [relax]: not important\n",
    )
    ready.fakes.gemini.script(_critique("GEM"))
    ready.fakes.claude.script(_critique("CLA"), "## Summary\n\nA.\n\n## Responses\n\n- GPT-1 [reject]: caller's job\n")
    ready.fakes.claude_code.script({"fixed": [], "not_fixed": [{"id": "GPT-1", "reason": "no"}], "summary": "s"})

    output = CrosscheckBackend().run_stage(ready.ctx("crosscheck", mode="code"))

    assert output.loop_back is True and "relaxed_criteria" not in output.index_updates
    note = output.notes[-1].handoff
    assert "GPT-1: free() lacks a range check" in note.section("Unresolved Critical")
    assert "`relax` ruling(s) on GPT-1 ignored" in note.section("Summary")


def test_a_criterion_relaxed_earlier_is_soft_for_the_critics_and_major_at_most(
    ready: StageEnv, sample_bodies: dict[str, str]
) -> None:
    ready.put("03-execution-r2", HandoffKind.EXECUTION, sample_bodies["execution"], round=2)
    ctx = ready.ctx("crosscheck", round=2, mode="code")
    ctx.index.relaxed_criteria = ["AC-3 (GPT-1, round 1): over-specified"]
    ready.fakes.chatgpt.script(_critique("GPT", AC3_ISSUE))
    ready.fakes.gemini.script(_critique("GEM"))
    ready.fakes.claude.script(_critique("CLA"), "## Summary\n\nA.\n\n## Responses\n\n- GPT-1 [accept]: no\n")
    ready.fakes.claude_code.script({"fixed": [], "not_fixed": [{"id": "GPT-1", "reason": "no"}], "summary": "s"})

    output = CrosscheckBackend().run_stage(ctx)

    critique_prompt = prompt_of(ready.fakes.gemini.calls[0])
    assert f"{OVERSPECIFIED.replace('[hard]', '[soft]')} (relaxed in round 1: over-specified" in critique_prompt
    assert len(ready.fakes.chatgpt.calls) == 1  # a major issue the author accepted needs no adjudication
    note = output.notes[-1].handoff
    assert note.section("Verdict") == "PASS" and note.section("Issues").startswith("- [major] GPT-1:")
    assert "relaxed_criteria" not in output.index_updates  # unchanged: the index keeps the earlier line
    assert note.section(cc.RELAXED_SECTION).endswith("- AC-3 (GPT-1, round 1): over-specified")


def _entry(stage: str, cost: float) -> LedgerEntry:
    return LedgerEntry(
        ts="2026-09-29T12:00:00Z", run_id=RUN_ID, stage=stage, agent="claude", provider="anthropic", model="m",
        cost_usd=cost, worst_case_usd=cost,
    )


def test_round_costs_split_the_ledger_into_execution_and_crosscheck_rounds() -> None:
    entries = [
        _entry("ingestion", 0.1), _entry("strategy", 0.1),
        _entry("execution", 0.06), _entry("execution", 10.28), _entry("execution", 1.13),  # timeout, resumed
        _entry("crosscheck", 3.0), _entry("crosscheck", 7.4),
        _entry("execution", 6.0), _entry("crosscheck", 5.6),
        _entry("final", 0.5), _entry("execution", 2.0),  # an extra round after final
    ]
    assert round_costs(entries) == pytest.approx([21.87, 11.6, 2.0])
    assert round_costs([]) == []


def test_loop_estimate_uses_the_last_round_and_reserves_final(ready: StageEnv) -> None:
    ctx = ready.ctx("crosscheck", round=2, mode="code")
    ctx.ledger = Ledger(RUN_ID, 38.0)
    for entry in (_entry("execution", 20.0), _entry("crosscheck", 2.0), _entry("execution", 6.0),
                  _entry("crosscheck", 6.33)):
        ctx.ledger.record(entry)
    estimate = cc.loop_estimate(ctx, "code", "prompt")
    assert isinstance(estimate, LoopEstimate) and estimate.basis == "round 2's spend"
    assert (estimate.round_usd, estimate.final_usd, estimate.remaining_usd) == (
        pytest.approx(12.33), pytest.approx(0.05 + 1.5), pytest.approx(3.67)
    )
    assert not estimate.affordable
    assert "another round (execution + cross-check) is estimated at $12.33 (round 2's spend)" in estimate.describe()
    assert "final needs up to $1.55, $13.88 in all; the run has $3.67 left" in estimate.describe()
    cheap = Ledger(RUN_ID, 25.0)
    for entry in (_entry("execution", 0.5), _entry("crosscheck", 0.1)):
        cheap.record(entry.model_copy(update={"worst_case_usd": entry.cost_usd * 4}))
    cheap.record(_entry("crosscheck", 0.2).model_copy(update={"provider": "claude_code", "purpose": "fixes"}))
    ctx.ledger = cheap
    floor = cc.loop_estimate(ctx, "code", "prompt")
    # an execution session ($3), the cross-check's non-Claude-Code calls at their worst case ($0.40), a fix session ($3)
    assert floor.round_usd == pytest.approx(6.4)
    assert floor.basis == (
        "the smallest round maf starts: an execution session $3.00, the cross-check's calls at their worst case $0.40 "
        "and a fix session $3.00; round 2 spent $0.80"
    )
    assert floor.affordable
    assert cc.loop_estimate(ctx, "prose", "prompt").round_usd == pytest.approx(0.8)
    small = Ledger(RUN_ID, 5.0)  # a small run's floor is a share of its cap: $1.25 per session
    small.record(_entry("execution", 0.5))
    ctx.ledger = small
    assert cc.loop_estimate(ctx, "code", "prompt").round_usd == pytest.approx(2.5)


def test_unaffordable_loop_goes_to_final_and_says_so(ready: StageEnv) -> None:
    ctx = ready.ctx("crosscheck", mode="code")
    ctx.ledger = Ledger(RUN_ID, 3.0)
    ready.fakes.chatgpt.script(_critique("GPT", "critical:free() lacks a range check"))
    ready.fakes.gemini.script(_critique("GEM"))
    ready.fakes.claude.script(_critique("CLA"), "## Summary\n\nA.\n\n## Responses\n\n- GPT-1 [accept]: yes\n")
    ready.fakes.claude_code.script({"fixed": [], "not_fixed": [{"id": "GPT-1", "reason": "no"}], "summary": "s"})
    ctx.settings = ctx.settings.model_copy(update={"claude_code_min_session_usd": 1.0})

    output = CrosscheckBackend().run_stage(ctx)

    note = output.notes[-1].handoff
    assert output.loop_back is False and note.meta.to == "final" and note.section("Verdict") == "LOOP"
    assert output.index_updates == {"unresolved_critical": 1, "loop_skipped": "budget"}
    summary = note.section("Summary")
    assert "Loop skipped: budget. Another round (execution + cross-check) is estimated at" in summary
    assert "ends `completed_with_issues` with 1 critical issue(s) unresolved" in summary
    assert f"maf resume {RUN_ID} --extra-round --budget USD" in summary


def test_affordable_loop_logs_its_estimate_and_clears_an_earlier_skip(ready: StageEnv) -> None:
    ctx = ready.ctx("crosscheck", mode="code")
    ctx.index.loop_skipped = "budget"  # an extra round after a budget skip, now with a raised budget
    ready.fakes.chatgpt.script(_critique("GPT", "critical:free() lacks a range check"))
    ready.fakes.gemini.script(_critique("GEM"))
    ready.fakes.claude.script(_critique("CLA"), "## Summary\n\nA.\n\n## Responses\n\n- GPT-1 [accept]: yes\n")
    ready.fakes.claude_code.script({"fixed": [], "not_fixed": [{"id": "GPT-1", "reason": "no"}], "summary": "s"})

    output = CrosscheckBackend().run_stage(ctx)

    assert output.loop_back is True and output.index_updates == {"unresolved_critical": 1, "loop_skipped": None}
    assert "Loop budget check: another round" in output.notes[-1].handoff.section("Summary")
    assert "so the run loops." in output.notes[-1].handoff.section("Summary")


# ============================================================================= final: relaxed criteria


FINAL_BODY = """## Summary

The allocator is complete.

## Deliverables

- `src/alloc.c`

## Verification

All suites pass.

## Acceptance

- AC-1 [met]: all suites pass
- AC-2 [met]: text=1804 bytes
- AC-3 [partial]: most tables are cross-checked

## Provenance

- [[02-strategy]]

## Limitations

None.
"""


def test_final_treats_a_relaxed_criterion_as_soft_and_lists_it(ready: StageEnv, sample_bodies: dict[str, str]) -> None:
    ready.put("04-crosscheck", HandoffKind.CROSSCHECK, sample_bodies["crosscheck"], from_="maf")
    ctx = ready.ctx("final", mode="prose")
    ctx.index.relaxed_criteria = ["AC-3 (GPT-1, round 1): the brief asks only for simulated numbers"]
    ready.workspace_file("document.md", "# Doc\n")
    ready.fakes.claude.script(FINAL_BODY)

    output = FinalBackend().run_stage(ctx)

    assert (output.index_updates["criteria_unmet"], output.index_updates["unmet_criteria"]) == (0, [])
    note = output.notes[0].handoff
    assert "[!warning]" not in note.section("Summary")
    assert (
        f"- AC-3 [partial]: {OVERSPECIFIED.split(': ', 1)[1]} (relaxed to soft: over-specified relative to the brief; "
        "see `## Relaxed Criteria`)\n  - Evidence: most tables are cross-checked"
    ) in note.section("Acceptance")
    assert note.section(final_mod.RELAXED_SECTION).endswith(
        f"- AC-3 [partial]: {OVERSPECIFIED.split(': ', 1)[1]}\n"
        "  - Relaxed (GPT-1, round 1): the brief asks only for simulated numbers"
    )
    assert "- AC-3 [partial]: every computed number in every table is recomputed from saved outputs (relaxed)" in (
        note.section("Limitations")
    )
    prompt = prompt_of(ready.fakes.claude.calls[0])
    assert "- AC-3 [soft]: every computed number" in prompt
    assert "- AC-3 (GPT-1, round 1): the brief asks only for simulated numbers" in prompt
    assert hf.validate_handoff(note) == []


def test_final_without_relaxations_is_unchanged(ready: StageEnv, sample_bodies: dict[str, str]) -> None:
    ready.put("04-crosscheck", HandoffKind.CROSSCHECK, sample_bodies["crosscheck"], from_="maf")
    ready.workspace_file("document.md", "# Doc\n")
    ready.fakes.claude.script(FINAL_BODY)
    output = FinalBackend().run_stage(ready.ctx("final", mode="prose"))
    note = output.notes[0].handoff
    assert output.index_updates["unmet_criteria"] == [f"AC-3 [partial]: {OVERSPECIFIED.split(': ', 1)[1]}"]
    assert final_mod.RELAXED_SECTION not in note.sections


def test_final_names_a_budget_skip_instead_of_the_loop_cap(ready: StageEnv, sample_bodies: dict[str, str]) -> None:
    looped = sample_bodies["crosscheck"].replace(
        "## Unresolved Critical\n\nNone.", "## Unresolved Critical\n\n- [critical] GPT-1: bug"
    )
    ready.put("04-crosscheck", HandoffKind.CROSSCHECK, looped.replace("PASS", "LOOP"), from_="maf")
    ctx = ready.ctx("final", mode="prose", unresolved_critical=1)
    ctx.index.loop_skipped = "budget"
    ready.workspace_file("document.md", "# Doc\n")
    ready.fakes.claude.script(FINAL_BODY.replace("- AC-3 [partial]", "- AC-3 [met]"))

    note = FinalBackend().run_stage(ctx).notes[0].handoff

    assert "after the cross-check, whose next loop was skipped for budget" in note.section("Summary")
    assert "loop cap" not in note.section("Summary")
    assert "Another cross-check loop was skipped because the run's budget could not pay for it" in prompt_of(
        ready.fakes.claude.calls[0]
    )


# ============================================================================= run.md


def _index(**kw: object) -> RunIndex:
    return RunIndex(
        run_id=RUN_ID, budget_usd=38.0, created="2026-09-29T13:42:53", updated="2026-09-29T17:30:50",
        workspace="/w", brief="b", **kw,
    )


def test_run_md_leaves_out_the_new_keys_until_they_are_used(stage_env: StageEnv) -> None:
    stage_env.vault.write_index(_index())
    text = stage_env.paths.run_md.read_text()
    assert "relaxed_criteria" not in text and "loop_skipped" not in text
    stage_env.vault.write_index(_index(relaxed_criteria=["AC-9 (GPT-4, round 2): r"], loop_skipped="budget"))
    loaded = stage_env.vault.read_index(RUN_ID)
    assert loaded.relaxed_criteria == ["AC-9 (GPT-4, round 2): r"] and loaded.loop_skipped == "budget"
    assert "- Relaxed criteria: AC-9" in stage_env.paths.run_md.read_text()


def test_run_md_names_a_budget_skip_and_a_partial_export() -> None:
    done = _index(status=RunStatus.COMPLETED_WITH_ISSUES, unresolved_critical=2, loop_skipped="budget",
                  handoffs=["04-crosscheck-r2"])
    assert describe_issues(done) == (
        "2 unresolved critical issue(s) after the cross-check (another loop was skipped: over budget)"
    )
    assert "> Another cross-check loop was skipped because the budget could not pay for it with 2 critical" in (
        render_run_body(done)
    )
    partial = _index(status=RunStatus.BUDGET_EXCEEDED, export_note=f"{PARTIAL_EXPORT_NOTE} after ...",
                     exported_at="2026-09-29T17:30:50")
    body = render_run_body(partial)
    assert body.split("## Status\n\n", 1)[1].startswith("> [!warning] Partial deliverables\n")
    assert "partial and unverified: no acceptance, clean-room, source-audit or lint gate checked them" in body
    assert "Partial deliverables" not in render_run_body(partial.model_copy(update={"status": RunStatus.RUNNING}))


# ============================================================================= end to end


@dataclass
class ResilienceScript(Script):
    """The allocator run with Claude Code misbehaving: the first ``timeouts`` execution sessions time out after writing
    part of the work, and with ``exhaust`` the first one stops at its budget cap instead (``exhaust`` is the cost it
    reports). ``strategies`` replaces the first strategy replies (one each)."""

    timeouts: int = 0
    exhaust: float = 0.0
    strategies: list[str] = field(default_factory=list)

    def reply(self, role: str, request: CompletionRequest) -> str | dict[str, object]:
        kind = _kind_of(request)
        if kind == "execution" and (self.timeouts or self.exhaust):
            self.requests.append((role, kind, _is_repair(request)))
            self.build()  # the session wrote part of the work before it was stopped
            if self.timeouts:
                self.timeouts -= 1
                raise _timeout(cost=min(2.0, request.max_budget_usd or 2.0))
            cost, self.exhaust = self.exhaust, 0.0
            raise _exhausted(cost)
        if kind == "strategy" and self.strategies:
            self.requests.append((role, kind, _is_repair(request)))
            return self.strategies.pop(0)
        return super().reply(role, request)


def _code_purposes(pipeline, run_id: str) -> list[tuple[str, str, str]]:  # type: ignore[no-untyped-def]
    ledger = pipeline.ledger_for(pipeline.status(run_id))
    return [(e.stage, e.purpose, e.error.split(":")[0]) for e in ledger.entries if e.provider == "claude_code"]


def test_e2e_timeout_then_continuation_completes(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    """(1) The first execution session times out; one continuation finishes it and the run completes."""
    script = ResilienceScript(sample_bodies, timeouts=1)
    pipeline, run_id = _start(settings, fake_providers, script)

    index = pipeline.run(run_id)

    assert index.status == RunStatus.COMPLETED, index.error
    assert index.handoffs == EXPECTED_HANDOFFS
    assert _code_purposes(pipeline, run_id) == [
        ("execution", "execution", "ClaudeCodeTimeout"),
        ("execution", "execution-continuation", ""),
        ("crosscheck", "fixes", ""),
        ("execution", "execution", ""),
        ("final", "cleanroom", ""),
    ]
    continued = prompt_of(fake_providers.claude_code.calls[1])
    assert continued.startswith("# Continuation: the previous session was cut off")
    assert "# Task: produce the definitive artifacts (round 1)" in continued
    execution = pipeline.vault.read_handoff(run_id, "03-execution")
    assert "maf: the first Claude Code session of this pass was stopped" in execution.section("Summary")
    assert execution.meta.cost_usd == pytest.approx(2.0 + 0.5)
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    assert len(ledger.entries) == _total_calls(fake_providers)
    notes = sum(pipeline.vault.read_handoff(run_id, name).meta.cost_usd for name in index.handoffs)
    assert notes == pytest.approx(ledger.spent_usd)  # the timed-out charge belongs to the execution note


def test_e2e_second_timeout_fails_with_partial_export(
    settings: Settings,
    fake_providers: FakeProviders,
    sample_bodies: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """(2) The session and its continuation both time out: FAILED with a clear message, both charged, and the partial
    workspace exported as unverified deliverables."""
    script = ResilienceScript(sample_bodies, timeouts=2)
    pipeline, run_id = _start(settings, fake_providers, script)
    monkeypatch.setattr(cli, "make_pipeline", lambda _settings: pipeline)
    argv = ["--vault", str(settings.vault_path), "--workspaces", str(settings.workspaces_path), "resume", run_id]

    assert cli.main(argv) == cli.EXIT_FAILED

    index = pipeline.status(run_id)
    assert (index.status, index.stage, index.round) == (RunStatus.FAILED, "execution", 1)
    assert index.error == (
        f"execution: ClaudeCodeTimeout: the Claude Code execution session and its one continuation both timed out "
        f"({TIMEOUT}; then {TIMEOUT}); their work stays in the workspace, and maf resume runs execution again"
    )
    assert _code_purposes(pipeline, run_id) == [
        ("execution", "execution", "ClaudeCodeTimeout"),
        ("execution", "execution-continuation", "ClaudeCodeTimeout"),
    ]
    paths = pipeline.vault.paths(run_id)
    assert (paths.deliverables / "src" / "alloc.c").is_file() and not (paths.deliverables / ".maf").exists()
    assert index.export_note and index.export_note.startswith(
        "partial export after the run stopped failed at execution: 2 file(s)"
    )
    assert "partial and unverified" in index.export_note
    assert "> [!warning] Partial deliverables" in paths.run_md.read_text()
    err = capsys.readouterr().err
    assert f"partial deliverables (unverified): {paths.deliverables}" in err
    assert "failed: execution: ClaudeCodeTimeout" in err


def test_e2e_budget_below_the_session_minimum_stops_before_spawning_and_resumes(
    settings: Settings,
    fake_providers: FakeProviders,
    sample_bodies: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """(3) $1.97 left for an execution session that needs at least $2 (its $3 minimum, at most its run's $2 budget
    with the share at 1): BUDGET_EXCEEDED at no cost, with the missing amount named; resuming with more budget
    completes the run."""
    script = ResilienceScript(sample_bodies)
    settings = settings.model_copy(update={"claude_code_min_session_share": 1.0})
    pipeline, run_id = _start(settings, fake_providers, script, budget_usd=2.0)
    monkeypatch.setattr(cli, "make_pipeline", lambda _settings: pipeline)
    base_argv = ["--vault", str(settings.vault_path), "--workspaces", str(settings.workspaces_path)]

    assert cli.main([*base_argv, "resume", run_id]) == cli.EXIT_FAILED

    stopped = pipeline.status(run_id)
    assert (stopped.status, stopped.stage, stopped.round) == (RunStatus.BUDGET_EXCEEDED, "execution", 1)
    assert fake_providers.claude_code.calls == []  # never spawned
    assert stopped.error and "would get $1.97, below the $2.00 a work session needs" in stopped.error
    assert "at least $1.03 more: resume with --budget 3.03 or higher" in stopped.error
    assert stopped.spent_usd == pytest.approx(0.03)  # ingestion and strategy only
    deliverables = pipeline.vault.paths(run_id).deliverables
    assert stopped.export_note is None and not any(deliverables.iterdir())  # nothing to export yet
    err = capsys.readouterr().err
    assert "budget exceeded: budget cap $2.00 too low for the next Claude Code session" in err
    assert f"raise the cap with: maf resume {run_id} --budget USD" in err

    assert cli.main([*base_argv, "resume", run_id, "--budget", "25"]) == cli.EXIT_OK

    resumed = pipeline.status(run_id)
    assert resumed.status == RunStatus.COMPLETED, resumed.error
    assert resumed.handoffs == EXPECTED_HANDOFFS and script.count("strategy") == 1


def test_e2e_budget_exhausted_under_a_clamp_is_budget_exceeded_with_partial_export(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    """(4) The execution session asks for $8, the run has $4.97 left, and the session stops at that cap: the run ends
    BUDGET_EXCEEDED (not FAILED), the workspace is exported as partial deliverables, and a resume completes it."""
    script = ResilienceScript(sample_bodies, exhaust=4.9)
    pipeline, run_id = _start(settings, fake_providers, script, budget_usd=5.0)

    stopped = pipeline.run(run_id)

    assert (stopped.status, stopped.stage) == (RunStatus.BUDGET_EXCEEDED, "execution"), stopped.error
    assert stopped.error and "stopped at its $4.97 cap, which the run's remaining budget had cut from $8.00" in (
        stopped.error
    )
    assert fake_providers.claude_code.calls[0].max_budget_usd == pytest.approx(4.97)
    paths = pipeline.vault.paths(run_id)
    assert (paths.deliverables / "src" / "alloc.c").read_text() == "/* tlsf pass 0 */\n"
    assert stopped.export_note and stopped.export_note.startswith(
        "partial export after the run stopped budget_exceeded at execution"
    )
    post = frontmatter.load(paths.run_md)
    assert post["status"] == "budget_exceeded" and "> [!warning] Partial deliverables" in post.content
    assert _assert_ledger_mirrored(pipeline, run_id).entries[-1].cost_usd == pytest.approx(4.9)

    resumed = pipeline.resume(run_id, budget_usd=30.0)

    assert resumed.status == RunStatus.COMPLETED, resumed.error
    assert resumed.export_note and resumed.export_note.startswith("final: ")
    assert "Partial deliverables" not in paths.run_md.read_text()


def test_e2e_unclamped_budget_exhaustion_still_fails_and_exports(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    script = ResilienceScript(sample_bodies, exhaust=8.0)
    pipeline, run_id = _start(settings, fake_providers, script)

    stopped = pipeline.run(run_id)

    assert stopped.status == RunStatus.FAILED and stopped.error.startswith("execution: ClaudeCodeBudgetExhausted")
    assert stopped.export_note and stopped.export_note.startswith("partial export after the run stopped failed")


def test_e2e_partial_export_error_never_masks_the_failure(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = ResilienceScript(sample_bodies, timeouts=2)
    pipeline, run_id = _start(settings, fake_providers, script)

    def broken(*_a: object, **_k: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(final_mod, "export_run", broken)
    stopped = pipeline.run(run_id)

    assert stopped.status == RunStatus.FAILED and "both timed out" in (stopped.error or "")
    assert stopped.export_note is None and pipeline.status(run_id).error == stopped.error


def test_e2e_loop_skipped_for_budget_goes_to_final_with_issues(
    settings: Settings,
    fake_providers: FakeProviders,
    sample_bodies: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """(5) GPT-1 stays unfixed. Round 1 leaves $3.92, and another round plus final needs $4.30: the LOOP is skipped,
    final runs (clean room included) and the run ends completed_with_issues, naming the budget, not the loop cap."""
    script = ResilienceScript(sample_bodies, stubborn=True)
    pipeline, run_id = _start(settings, fake_providers, script, budget_usd=5.0)
    monkeypatch.setattr(cli, "make_pipeline", lambda _settings: pipeline)
    argv = ["--vault", str(settings.vault_path), "--workspaces", str(settings.workspaces_path), "resume", run_id]

    assert cli.main(argv) == cli.EXIT_WITH_ISSUES

    index = pipeline.status(run_id)
    assert index.status == RunStatus.COMPLETED_WITH_ISSUES, index.error
    assert (index.round, index.unresolved_critical, index.loop_skipped, index.criteria_unmet) == (1, 1, "budget", 0)
    assert script.count("execution") == 1 and script.count("final") == 1 and script.count("cleanroom") == 1
    crosscheck = pipeline.vault.read_handoff(run_id, "04-crosscheck")
    assert crosscheck.section("Verdict") == "LOOP" and crosscheck.meta.to == "final"
    summary = crosscheck.section("Summary")
    assert "Loop skipped: budget. Another round (execution + cross-check) is estimated at $2.75 (the smallest" in (
        summary
    )
    assert "an execution session $1.25, the cross-check's calls at their worst case $0.25 and a fix session $1.25" in (
        summary
    )
    assert "final needs up to $1.55, $4.30 in all; the run has $3.92 left" in summary
    final = pipeline.vault.read_handoff(run_id, "05-final")
    assert "whose next loop was skipped for budget" in final.section("Summary")
    assert "- [critical] GPT-1: stress test double-frees" in final.section("Limitations")
    run_md = pipeline.vault.paths(run_id).run_md.read_text()
    assert "loop_skipped: budget" in run_md and "Another cross-check loop was skipped" in run_md
    last = capsys.readouterr().err.strip().splitlines()[-1]
    assert last.startswith(
        "completed with issues: 1 unresolved critical issue(s) after the cross-check (another loop was skipped: "
        "over budget)"
    )
    assert last.endswith(f"one more pass: maf resume {run_id} --extra-round --budget USD)")
    assert _assert_ledger_mirrored(pipeline, run_id).spent_usd <= 5.0


def test_e2e_seven_hard_criteria_get_one_strategy_repair(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    """(6) The first strategy sets seven hard criteria; the repair names the cap and the run completes."""
    seven = _strategy(sample_bodies["strategy"], _criteria(7))
    script = ResilienceScript(sample_bodies, strategies=[seven])
    pipeline, run_id = _start(settings, fake_providers, script)

    index = pipeline.run(run_id)

    assert index.status == RunStatus.COMPLETED, index.error
    assert [(role, repair) for role, kind, repair in script.requests if kind == "strategy"] == [
        ("chatgpt", False), ("chatgpt", True),
    ]
    repair = next(r for r in fake_providers.chatgpt.calls if _is_repair(r))
    assert "has 7 hard criteria" in prompt_of(repair) and "At most 6 criteria may be `hard`" in prompt_of(repair)
    assert pipeline.vault.read_handoff(run_id, "02-strategy").meta.cost_usd == pytest.approx(0.02)


@dataclass
class RelaxScript(Script):
    """Round 1: ChatGPT reports the over-specified AC-3 unmet, the author accepts, the adjudicator relaxes it, and the
    fixer cannot meet it. The final report calls AC-3 partial."""

    def answer(self, role: str, kind: str) -> str | dict[str, object]:
        if kind == "strategy":
            criteria = self.bodies["strategy"].split("## Acceptance Criteria\n\n", 1)[1].split("\n\n## Risks", 1)[0]
            return _strategy(self.bodies["strategy"], f"{criteria}\n{OVERSPECIFIED}")
        if kind == "critique":
            if role == "chatgpt" and self.executions == 1:
                return _critique("GPT", AC3_ISSUE)
            return CLEAN_CRITIQUE
        if kind == "rebuttal":
            return "## Summary\n\nAnswered.\n\n## Responses\n\n- GPT-1 [accept]: Table 4 is not recomputed\n"
        if kind == "adjudication":
            return (
                "## Summary\n\nRuled.\n\n## Rulings\n\n- GPT-1 [relax]: the brief asks for a tested allocator; "
                "recomputing every table number is beyond it\n"
            )
        if kind == "fix_report":
            return {"fixed": [], "not_fixed": [{"id": "GPT-1", "reason": "no recomputation harness"}], "summary": "s"}
        return super().answer(role, kind)


def test_e2e_relaxed_criterion_does_not_block_and_is_listed(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    """(7) The adjudicator relaxes the over-specified AC-3: no loop, and final reports it partial without blocking."""
    script = RelaxScript(sample_bodies, acceptance=ACCEPTANCE_MET + "- AC-3 [partial]: two tables are cross-checked\n")
    pipeline, run_id = _start(settings, fake_providers, script)

    index = pipeline.run(run_id)

    assert index.status == RunStatus.COMPLETED, index.error
    assert (index.round, index.unresolved_critical, index.criteria_unmet) == (1, 0, 0)
    assert script.count("execution") == 1  # the relaxed criterion did not loop
    line = "AC-3 (GPT-1, round 1): the brief asks for a tested allocator; recomputing every table number is beyond it"
    assert index.relaxed_criteria == [line]
    vault = pipeline.vault
    crosscheck = vault.read_handoff(run_id, "04-crosscheck")
    assert crosscheck.section("Verdict") == "PASS"
    assert crosscheck.section("Issues").startswith("- [major] GPT-1: Unmet acceptance criterion AC-3:")
    assert crosscheck.section(cc.RELAXED_SECTION).endswith(f"- {line}")
    final = vault.read_handoff(run_id, "05-final")
    assert "(relaxed to soft: over-specified relative to the brief" in final.section("Acceptance")
    assert final.section(final_mod.RELAXED_SECTION).endswith(
        f"- AC-3 [partial]: {OVERSPECIFIED.split(': ', 1)[1]}\n  - Relaxed (GPT-1, round 1): the brief asks for a "
        "tested allocator; recomputing every table number is beyond it"
    )
    assert "(relaxed)" in final.section("Limitations")
    post = frontmatter.load(vault.paths(run_id).run_md)
    assert post["relaxed_criteria"] == [line] and "- Relaxed criteria: AC-3" in post.content
    out, err = io.StringIO(), io.StringIO()
    assert cli._report(pipeline, index, out, err) == cli.EXIT_OK
    assert err.getvalue() == (
        "criteria relaxed to soft as over-specified (not verified as written): AC-3; see ## Relaxed Criteria in "
        f"{vault.paths(run_id).note('05-final')}\n"
    )
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    assert len(ledger.entries) == _total_calls(fake_providers)


def test_defaults_match_the_incident_fixes() -> None:
    settings = Settings()
    assert settings.claude_code_timeout_s == 5400.0 and settings.claude_code_min_session_usd == 3.0
    assert settings.claude_code_min_session_share == 0.25
    assert settings.bash_timeout_s == 4050.0


# ============================================================================= review fixes: hints, floors, reserves


def test_session_floor_is_a_share_of_a_small_run_and_the_hint_solves_for_the_cap() -> None:
    settings = Settings()
    assert base.session_floor(settings, 38.0, 8.0) == 3.0
    assert base.session_floor(settings, 5.0, 8.0) == 1.25  # the $5 MCP default: a quarter of the run
    assert base.session_floor(settings, 38.0, 2.0) == 2.0  # never more than the session's own budget
    assert base.session_floor(settings, 5.0, 8.0, minimum_usd=0.0) == 0.0  # an explicit minimum (the repair)
    # c - committed >= min(fixed, share * c): the incident's $36.61 committed needs the full $3 on top
    assert base.needed_cap(36.61, 3.0, 0.25) == pytest.approx(39.61)
    assert base.needed_cap(3.0, 3.0, 0.25) == pytest.approx(4.0)  # at $4 the floor is $1 and $1 is left
    assert base.needed_cap(5.28, 3.0, None) == pytest.approx(8.28)


@pytest.mark.parametrize(("share", "hint"), [(0.25, 7.04), (1.0, 8.28)])
def test_hinted_cap_gives_the_session_its_minimum_when_less_than_one_turn_is_left(
    stage_env: StageEnv, share: float, hint: float
) -> None:
    """$1 left against Opus's $2.28 turn headroom: the hint must cover the uncovered headroom too, and resuming with
    exactly the hinted cap starts the session."""
    fake = HeadroomFake(name="claude_code", agent="claude", headroom=2.28)
    stage_env.fakes.claude_code = fake
    ctx = stage_env.ctx("execution", mode="code")
    ctx.settings = ctx.settings.model_copy(update={"claude_code_min_session_share": share})
    spent = _entry("execution", 3.0)
    ctx.ledger = Ledger(RUN_ID, 4.0)
    ctx.ledger.record(spent)

    with pytest.raises(SessionBudgetTooSmall) as info:
        work_session(ctx, _code_request(8.0), purpose="execution")

    assert info.value.budget_usd == 0.0 and info.value.needed_cap_usd == pytest.approx(hint)
    assert f"at least ${hint - 4.0:.2f} more: resume with --budget {hint:.2f} or higher" in str(info.value)
    ctx.ledger = Ledger(RUN_ID, info.value.needed_cap_usd)
    ctx.ledger.record(spent)
    fake.script("ok")
    assert work_session(ctx, _code_request(8.0), purpose="execution").text == "ok"
    floor = base.session_floor(ctx.settings, info.value.needed_cap_usd, 8.0)
    assert fake.calls[-1].max_budget_usd >= floor - 1e-9


def test_crosscheck_overhead_counts_the_latest_crosschecks_non_claude_code_worst_cases() -> None:
    def worst(stage: str, cost: float, provider: str = "anthropic", purpose: str = "") -> LedgerEntry:
        return _entry(stage, cost).model_copy(
            update={"worst_case_usd": cost * 2, "provider": provider, "purpose": purpose}
        )

    entries = [
        worst("execution", 5.0, "claude_code"), worst("crosscheck", 1.0), worst("crosscheck", 0.5, "gemini"),
        worst("crosscheck", 4.0, "claude_code", "fixes"),
        worst("execution", 0.06, "claude_code", "preflight"),  # a resumed round 2: no cross-check yet
    ]
    assert base.crosscheck_overhead(entries) == pytest.approx(3.0)  # round 1's critiques and audit at worst case
    assert base.crosscheck_overhead(entries[:1]) == 0.0
    assert [len(r) for r in base.round_entries(entries)] == [4, 1]


def _stubborn_crosscheck(ready: StageEnv, ledger: Ledger, **settings: object) -> base.StageContext:
    """A cross-check whose one critical issue the author accepts, so the fix pass gets it."""
    ctx = ready.ctx("crosscheck", mode="code")
    ctx.ledger = ledger
    if settings:
        ctx.settings = ctx.settings.model_copy(update=settings)
    ready.fakes.chatgpt.script(_critique("GPT", "critical:free() lacks a range check"))
    ready.fakes.gemini.script(_critique("GEM"))
    ready.fakes.claude.script(_critique("CLA"), "## Summary\n\nA.\n\n## Responses\n\n- GPT-1 [accept]: yes\n")
    return ctx


def test_fix_session_keeps_finals_reserve(ready: StageEnv) -> None:
    ctx = _stubborn_crosscheck(ready, Ledger(RUN_ID, 6.0), claude_code_min_session_usd=1.0)
    ready.fakes.claude_code.script({"fixed": [], "not_fixed": [{"id": "GPT-1", "reason": "no"}], "summary": "s"})

    output = CrosscheckBackend().run_stage(ctx)

    (fix,) = ready.fakes.claude_code.calls
    reserve = 0.05 + 1.5  # the final report's worst case (fake) and the clean room's budget
    assert fix.max_budget_usd == pytest.approx(6.0 - 0.04 - reserve)  # after three critiques and the rebuttal
    assert output.notes[-1].handoff.section("Applied Fixes") != ""


def test_fix_pass_that_cannot_keep_finals_reserve_is_skipped_and_the_crosscheck_goes_on(ready: StageEnv) -> None:
    """The fix pass needs $0.51 (a quarter of the $2.05 run) but would get $0.46 after keeping final's $1.55: it is
    skipped (nothing spawned, nothing spent), the issue stays open, and the unaffordable LOOP goes to final instead of
    stopping the run."""
    ctx = _stubborn_crosscheck(ready, Ledger(RUN_ID, 2.05), claude_code_min_session_usd=1.0)

    output = CrosscheckBackend().run_stage(ctx)

    assert ready.fakes.claude_code.calls == []
    note = output.notes[-1].handoff
    assert note.section("Applied Fixes").endswith("fix pass skipped: budget")
    assert "GPT-1: free() lacks a range check" in note.section("Unresolved Critical")
    summary = note.section("Summary")
    assert "Fix pass skipped: budget. The run has $2.01 left; after one turn of headroom ($0.00) and $1.55 kept " in (
        summary
    )
    assert "a fix session would get $0.46, below its $0.51 minimum. The 1 issue(s) to fix stay open." in summary
    assert "Loop skipped: budget" in summary and note.meta.to == "final" and output.loop_back is False
    assert output.index_updates == {"unresolved_critical": 1, "loop_skipped": "budget"}
    assert note.meta.cost_usd == 0.0 and note.meta.model == "none"


def test_fix_session_out_of_its_cut_budget_counts_nothing_fixed(ready: StageEnv) -> None:
    ctx = _stubborn_crosscheck(ready, Ledger(RUN_ID, 6.0), claude_code_min_session_usd=1.0)
    ready.workspace_file("src/alloc.c", "/* half fixed */\n")
    ready.fakes.claude_code.script(_exhausted(4.3))

    output = CrosscheckBackend().run_stage(ctx)

    note = output.notes[-1].handoff
    assert "the fix session ran out of its $4.41 budget before reporting" in note.section("Applied Fixes")
    assert "stopped at its $4.41 cap (cut to keep final affordable)" in note.section("Summary")
    assert note.meta.cost_usd == pytest.approx(4.3) and output.loop_back is False  # $1.65 left: no other round


def test_refused_fix_continuation_keeps_the_timed_out_charge(ready: StageEnv) -> None:
    ctx = _stubborn_crosscheck(ready, Ledger(RUN_ID, 6.0), claude_code_min_session_usd=1.0)
    ready.fakes.claude_code.script(_timeout(cost=3.5))  # leaves $2.45: under final's reserve plus the $1 minimum

    output = CrosscheckBackend().run_stage(ctx)

    assert len(ready.fakes.claude_code.calls) == 1
    note = output.notes[-1].handoff
    assert "The first fix session timed out and was charged $3.50; its continuation was not started." in (
        note.section("Summary")
    )
    assert note.meta.cost_usd == pytest.approx(3.5) and note.meta.model == ctx.model("claude_code")
    assert sum(e.cost_usd for e in ctx.ledger.entries if e.provider == "claude_code") == pytest.approx(3.5)


# ----------------------------------------------------------------------------- end to end


@dataclass
class CleanScript(ResilienceScript):
    """Round 1 passes: every critic is clean."""

    def answer(self, role: str, kind: str) -> str | dict[str, object]:
        return CLEAN_CRITIQUE if kind == "critique" else super().answer(role, kind)


def test_e2e_resuming_with_exactly_the_hinted_budget_in_a_new_process_completes(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    """The hint covers the sandbox preflight a resumed process pays again: before, following it literally was
    refused again every time, short by one more preflight."""
    settings = settings.model_copy(update={"claude_code_min_session_usd": 1.0, "claude_code_min_session_share": 1.0})
    script = CleanScript(sample_bodies)
    pipeline, code = _sandboxed_pipeline(settings, fake_providers, script)
    run_id = pipeline.create(BRIEF, budget_usd=1.2).run_id

    stopped = pipeline.run(run_id)

    assert (stopped.status, stopped.stage) == (RunStatus.BUDGET_EXCEEDED, "execution"), stopped.error
    assert code.calls == []  # refused before the preflight too
    preflight = preflight_budget_usd(settings.model_for("claude_code", "execution"))
    match = re.search(r"resume with --budget (\d+\.\d\d) or higher", stopped.error or "")
    assert match and float(match[1]) == pytest.approx(0.03 + preflight + 1.0, abs=0.01)
    assert f"first pays up to ${preflight:.2f} for the sandbox preflight" in (stopped.error or "")

    resumed = pipeline.resume(run_id, budget_usd=float(match[1]))  # a new _advance: an unverified sandbox again

    assert resumed.status == RunStatus.COMPLETED, resumed.error
    assert code.preflights == 1 and _code_kinds(script) == ["preflight", "execution", "cleanroom"]
    _assert_ledger_mirrored(pipeline, run_id)


def test_e2e_mcp_default_budget_still_starts_a_code_session_with_the_real_turn_headroom(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    """ChatGPT's default $5 run with Opus's $2.28 turn headroom: the session's minimum is a quarter of the run
    ($1.25), so execution starts with $2.69 instead of the run stopping before any work. The fix pass that cannot keep
    final's reserve is skipped, and the run still ends with a final report and a clean room."""
    fake_providers.claude_code = HeadroomFake(name="claude_code", agent="claude", headroom=2.28, cost_per_call=0.5)
    script = ResilienceScript(sample_bodies)
    pipeline, run_id = _start(settings, fake_providers, script, budget_usd=DEFAULT_MCP_MAX_BUDGET_USD)

    index = pipeline.run(run_id)

    first = fake_providers.claude_code.calls[0]
    assert _kind_of(first) == "execution" and first.max_budget_usd == pytest.approx(5.0 - 0.03 - 2.28)
    assert index.status == RunStatus.COMPLETED_WITH_ISSUES, index.error
    assert (index.round, index.loop_skipped) == (1, "budget")
    assert script.count("fix_report") == 0 and script.count("cleanroom") == 1 and script.count("final") == 1
    crosscheck = pipeline.vault.read_handoff(run_id, "04-crosscheck")
    assert "Fix pass skipped: budget." in crosscheck.section("Summary")
    assert _assert_ledger_mirrored(pipeline, run_id).spent_usd <= 5.0


@dataclass
class CappedPriceFake(HeadroomFake):
    """Claude Code whose execution sessions cost the next of ``prices`` (at most the session's cap), as a session
    that works until its budget is gone would."""

    prices: list[float] = field(default_factory=list)

    def complete(self, request: CompletionRequest) -> CompletionResult:
        result = super().complete(request)
        if _kind_of(request) != "execution" or not self.prices:
            return result
        return result.model_copy(update={"cost_usd": min(self.prices.pop(0), request.max_budget_usd or 0.0)})


def test_e2e_a_dearer_second_round_still_reaches_final(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    """$12 run, a stubborn critical issue. Round 1 is cheap, so the loop is approved; round 2's execution would spend
    its whole $8. It holds back the next cross-check, a fix session and final, the fix pass keeps final's reserve, and
    the run ends completed_with_issues with a final report instead of budget_exceeded at crosscheck round 2."""
    fake_providers.claude_code = CappedPriceFake(
        name="claude_code", agent="claude", headroom=0.0, cost_per_call=0.5, prices=[0.5, 8.0]
    )
    script = ResilienceScript(sample_bodies, stubborn=True)
    pipeline, run_id = _start(settings, fake_providers, script, budget_usd=12.0)

    index = pipeline.run(run_id)

    assert index.status == RunStatus.COMPLETED_WITH_ISSUES, index.error
    assert (index.round, index.loop_skipped) == (2, "budget")
    executions = [r for r in fake_providers.claude_code.calls if _kind_of(r) == "execution"]
    assert executions[0].max_budget_usd == 8.0
    # round 2: $10.92 left, minus the cross-check's calls at worst case ($0.25), a $3 fix session and final's $1.55
    assert executions[1].max_budget_usd == pytest.approx(10.92 - 0.25 - 3.0 - 1.55)
    assert script.count("fix_report") == 2 and script.count("final") == 1 and script.count("cleanroom") == 1
    assert _assert_ledger_mirrored(pipeline, run_id).spent_usd <= 12.0


def test_e2e_extra_round_after_a_budget_skip_is_exactly_one_pass(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    """A budget skip at round 1 leaves round 2 below the loop cap; `--extra-round` must still run one pass, as the
    CLI hint promises, even when the raised budget would pay for more."""
    script = ResilienceScript(sample_bodies, stubborn=True)
    pipeline, run_id = _start(settings, fake_providers, script, budget_usd=5.0)
    skipped = pipeline.run(run_id)
    assert (skipped.status, skipped.round, skipped.loop_skipped) == (RunStatus.COMPLETED_WITH_ISSUES, 1, "budget")

    again = pipeline.resume(run_id, extra_round=True, budget_usd=40.0)

    assert again.status == RunStatus.COMPLETED_WITH_ISSUES, again.error
    assert (again.round, again.loop_skipped, script.count("execution")) == (2, None, 2)
    crosscheck = pipeline.vault.read_handoff(run_id, "04-crosscheck-r2")
    assert crosscheck.meta.to == "final" and crosscheck.section("Verdict") == "LOOP"
    assert "Extra round (`maf resume --extra-round` pays for one pass): the run goes to final" in (
        crosscheck.section("Summary")
    )
    assert "Loop budget check" not in crosscheck.section("Summary")
    assert again.handoffs[-1] == "05-final"


def test_e2e_failed_extra_round_keeps_the_verified_deliverables(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    """Final exported and gated the deliverables; an extra round whose execution times out twice must not replace
    them with its half-edited workspace (which stays in workspaces/ for maf resume)."""
    unmet = ACCEPTANCE_MET.replace("- AC-2 [met]: `arm-none-eabi-size` reports text=1804", "- AC-2 [unmet]: 2210 bytes")
    script = ResilienceScript(sample_bodies, acceptance=unmet)
    pipeline, run_id = _start(settings, fake_providers, script)
    done = pipeline.run(run_id)
    assert done.status == RunStatus.COMPLETED_WITH_ISSUES, done.error
    paths = pipeline.vault.paths(run_id)
    alloc = paths.deliverables / "src" / "alloc.c"
    assert alloc.read_text() == "/* tlsf pass 2 */\n" and (done.export_note or "").startswith("final: ")

    script.timeouts, script.executions = 2, 98
    failed = pipeline.resume(run_id, extra_round=True)

    assert (failed.status, failed.stage, failed.round) == (RunStatus.FAILED, "execution", 3), failed.error
    assert (paths.workspace / "src" / "alloc.c").read_text() == "/* tlsf pass 98 */\n"
    assert alloc.read_text() == "/* tlsf pass 2 */\n"
    assert failed.export_note == done.export_note and failed.exported_at == done.exported_at
    assert "Partial deliverables" not in paths.run_md.read_text()
