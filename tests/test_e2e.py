"""End-to-end offline runs: the real Pipeline, stage backends, vault and ledger, driven by scripted fakes.

The script answers by request content (which handoff kind the prompt asks for), so the tests exercise the
whole flow: ingestion -> strategy -> execution -> crosscheck (LOOP) -> execution r2 -> crosscheck r2 (PASS)
-> final. No network, no subprocess, zero cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import frontmatter
import pytest
from conftest import FakeProviders

from maf.config import Settings
from maf.handoff import HandoffKind, validate_handoff
from maf.ledger import Ledger
from maf.pipeline import Pipeline
from maf.providers import CompletionRequest
from maf.types import RunStatus

BRIEF = "Design a portable O(1) small-memory allocator."
BROKEN_BODY = "## Summary\n\nI forgot the other sections.\n"
PREFIX = {"chatgpt": "GPT", "gemini": "GEM", "claude": "CLA"}
TRIAGE = {
    "summary": "Build a portable O(1) allocator.",
    "execution_mode": "code",
    "gemini_instructions": "Survey O(1) allocators (TLSF, segregated fits).",
    "search_queries": ["TLSF allocator worst case"],
    "deliverable": "alloc.c with POSIX, FreeRTOS and QEMU tests",
}
CLEAN_CRITIQUE = "## Summary\n\nNo problems found.\n\n## Issues\n\nNone.\n"


class Crash(BaseException):
    """Stands in for the process dying mid-stage (not caught by the pipeline's ``except Exception``)."""


def _kind_of(request: CompletionRequest) -> str:
    if request.json_schema is not None:
        return request.schema_name
    prompt = request.messages[-1].content
    for kind in HandoffKind:
        if f"## Output format: {kind.value} handoff" in prompt:
            return kind.value
    raise AssertionError(f"cannot tell what this request wants: {prompt[:200]!r}")


def _is_repair(request: CompletionRequest) -> bool:
    return request.messages[-1].content.startswith("Your previous ")


@dataclass
class Script:
    """Scripted behaviour of all four agents for one allocator run.

    Round 1: ChatGPT raises a critical issue the fixer cannot fix (LOOP); round 2: every critic is clean (PASS).
    ``broken`` maps a kind to how many leading replies for it are invalid bodies. ``crash_at`` names a
    ``(kind, n)`` whose n-th request raises ``Crash``.
    """

    bodies: dict[str, str]
    broken: dict[str, int] = field(default_factory=dict)
    crash_at: tuple[str, int] | None = None
    workspace: Path | None = None
    requests: list[tuple[str, str, bool]] = field(default_factory=list)  # (agent role, kind, is_repair)
    executions: int = 0

    def install(self, fakes: FakeProviders) -> None:
        fakes.chatgpt.default = lambda r: self.reply("chatgpt", r)
        fakes.gemini.default = lambda r: self.reply("gemini", r)
        fakes.claude.default = lambda r: self.reply("claude", r)
        fakes.claude_code.default = lambda r: self.reply("claude_code", r)

    def count(self, kind: str) -> int:
        return sum(1 for _, k, _ in self.requests if k == kind)

    def reply(self, role: str, request: CompletionRequest) -> str | dict[str, object]:
        kind = _kind_of(request)
        self.requests.append((role, kind, _is_repair(request)))
        if self.crash_at == (kind, self.count(kind)):
            raise Crash(f"simulated crash during {kind} #{self.count(kind)}")
        if self.broken.get(kind, 0) >= self.count(kind):
            return BROKEN_BODY
        return self.answer(role, kind)

    def answer(self, role: str, kind: str) -> str | dict[str, object]:
        if kind == "triage":
            return TRIAGE
        if kind == "execution":
            self.executions += 1
            self.build()
            return self.bodies["execution"]
        if kind == "critique":
            if self.executions == 1 and role == "chatgpt":
                return "## Summary\n\nOne blocker.\n\n## Issues\n\n- [critical] GPT-1: stress test double-frees\n"
            if self.executions == 1 and role == "gemini":
                return "## Summary\n\nStyle only.\n\n## Issues\n\n- [minor] GEM-1: rename tlsf_map\n"
            return CLEAN_CRITIQUE
        if kind == "rebuttal":
            return "## Summary\n\nAnswered.\n\n## Responses\n\n- GPT-1 [accept]: will fix\n- GEM-1 [reject]: name is standard\n"
        if kind == "adjudication":
            return "## Summary\n\nRuled.\n\n## Rulings\n\n- GEM-1 [wontfix]: the name follows the paper\n"
        if kind == "fix_report":
            return {"fixed": [], "not_fixed": [{"id": "GPT-1", "reason": "needs a redesign"}], "summary": "tried"}
        return self.bodies[kind]

    def build(self) -> None:
        """What Claude Code would leave in the workspace: the artifacts the execution note lists."""
        assert self.workspace is not None
        (self.workspace / "src").mkdir(parents=True, exist_ok=True)
        (self.workspace / "src" / "alloc.c").write_text(f"/* tlsf pass {self.executions} */\n", encoding="utf-8")
        (self.workspace / "test").mkdir(exist_ok=True)
        (self.workspace / "test" / "posix_test.log").write_text("42 passed\n", encoding="utf-8")


def _pipeline(settings: Settings, fakes: FakeProviders) -> Pipeline:
    ticks = iter(range(100_000))
    return Pipeline(
        settings,
        providers_factory=fakes.factory(),
        clock=lambda: datetime(2026, 9, 28, 12, 0) + timedelta(seconds=next(ticks)),
    )


def _start(settings: Settings, fakes: FakeProviders, script: Script, **create_kw: object) -> tuple[Pipeline, str]:
    script.install(fakes)
    pipeline = _pipeline(settings, fakes)
    run_id = pipeline.create(BRIEF, **create_kw).run_id  # type: ignore[arg-type]
    script.workspace = pipeline.vault.paths(run_id).workspace
    return pipeline, run_id


def _total_calls(fakes: FakeProviders) -> int:
    return sum(len(f.calls) for f in fakes.all())


def _assert_ledger_mirrored(pipeline: Pipeline, run_id: str) -> Ledger:
    index = pipeline.status(run_id)
    ledger = pipeline.ledger_for(index)
    assert index.spent_usd == pytest.approx(ledger.spent_usd)
    assert index.spend_by_agent == pytest.approx(ledger.by_agent())
    assert index.spend_by_provider == pytest.approx(ledger.by_provider())
    assert index.spent_usd <= index.budget_usd
    return ledger


EXPECTED_HANDOFFS = [
    "01a-routing",
    "01-ingestion",
    "02-strategy",
    "03-execution",
    "04a-critique-chatgpt",
    "04a-critique-gemini",
    "04a-critique-claude",
    "04b-rebuttal",
    "04c-adjudication",
    "04-crosscheck",
    "03-execution-r2",
    "04a-critique-chatgpt-r2",
    "04a-critique-gemini-r2",
    "04a-critique-claude-r2",
    "04b-rebuttal-r2",
    "04c-adjudication-r2",
    "04-crosscheck-r2",
    "05-final",
]


def test_full_run_with_one_loop(settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]) -> None:
    script = Script(sample_bodies)
    pipeline, run_id = _start(settings, fake_providers, script)
    progress: list[str] = []

    index = pipeline.run(run_id, progress=lambda _i, msg: progress.append(msg))

    assert index.status == RunStatus.COMPLETED, index.error
    assert (index.stage, index.round, index.mode) == ("final", 2, "code")
    assert index.unresolved_critical == 0
    assert index.completed_stages == [
        "ingestion", "strategy", "execution", "crosscheck", "execution", "crosscheck", "final",
    ]
    assert index.handoffs == EXPECTED_HANDOFFS
    assert any("run completed" in msg for msg in progress)

    # Every handoff exists, parses, validates as its kind, and carries this run's metadata.
    vault = pipeline.vault
    for name in index.handoffs:
        handoff = vault.read_handoff(run_id, name)
        assert validate_handoff(handoff) == [], name
        assert handoff.meta.run_id == run_id
        expected_round = 2 if name.endswith("-r2") or name == "05-final" else 1  # final carries the last round
        assert handoff.meta.round == expected_round, name
        assert f"maf/{handoff.meta.stage.value}" in handoff.meta.tags
    assert vault.read_handoff(run_id, "04-crosscheck").section("Verdict").strip() == "LOOP"
    assert vault.read_handoff(run_id, "04-crosscheck-r2").section("Verdict").strip() == "PASS"
    assert "GPT-1" in vault.read_handoff(run_id, "04-crosscheck").section("Unresolved Critical")
    # Round 2 execution saw the unresolved issue from round 1.
    r2_prompt = fake_providers.claude_code.calls[2].messages[-1].content
    assert "GPT-1" in r2_prompt and "stress test double-frees" in r2_prompt
    # Final deliverables were copied from the latest execution pass.
    paths = vault.paths(run_id)
    assert (paths.deliverables / "src" / "alloc.c").read_text() == "/* tlsf pass 2 */\n"
    assert "[[04-crosscheck-r2]]" in vault.read_handoff(run_id, "05-final").section("Provenance")

    # run.md: frontmatter is the index; the body links every handoff and carries the cost table.
    post = frontmatter.load(paths.run_md)
    assert post["status"] == "completed" and post["run_id"] == run_id
    for name in EXPECTED_HANDOFFS:
        assert f"[[{name}]]" in post.content
    assert f"| **Total** | **{index.spent_usd:.4f}** |" in post.content
    assert f"| claude | {index.spend_by_agent['claude']:.4f} |" in post.content

    # Ledger: one entry per provider call, totals exact and mirrored into run.md.
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    assert len(ledger.entries) == _total_calls(fake_providers)
    code_calls = len(fake_providers.claude_code.calls)
    assert code_calls == 3  # execution, fix pass, execution r2 (round 2 has no issues, so no fix pass)
    assert ledger.by_agent() == pytest.approx(
        {
            "chatgpt": 0.01 * len(fake_providers.chatgpt.calls),
            "gemini": 0.01 * len(fake_providers.gemini.calls),
            "claude": 0.01 * len(fake_providers.claude.calls) + 0.50 * code_calls,
        }
    )
    assert set(ledger.by_stage()) == {"ingestion", "strategy", "execution", "crosscheck", "final"}
    # Each note's cost is what its own calls spent, so the notes add up to the ledger total.
    note_costs = sum(vault.read_handoff(run_id, name).meta.cost_usd for name in index.handoffs)
    assert note_costs == pytest.approx(ledger.spent_usd)


def test_budget_stop_then_resume_with_higher_budget(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    script = Script(sample_bodies)
    # Ingestion + strategy spend $0.03 and execution $0.50; three concurrent critiques each reserve a
    # $0.05 worst case, so they cannot all fit under $0.60.
    pipeline, run_id = _start(settings, fake_providers, script, budget_usd=0.60)

    stopped = pipeline.run(run_id)

    assert stopped.status == RunStatus.BUDGET_EXCEEDED
    assert (stopped.stage, stopped.round) == ("crosscheck", 1)
    assert stopped.error and "budget" in stopped.error.lower()
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    # The refused calls never reached a provider; everything that ran was recorded.
    assert len(ledger.entries) == _total_calls(fake_providers)
    assert ledger.spent_usd <= 0.60
    assert "04-crosscheck" not in stopped.handoffs
    assert stopped.handoffs[-1] == "03-execution"
    # run() leaves a stopped run alone; continuing it is an explicit resume.
    assert pipeline.run(run_id).status == RunStatus.BUDGET_EXCEEDED

    resumed = pipeline.resume(run_id, budget_usd=5.0)

    assert resumed.status == RunStatus.COMPLETED, resumed.error
    assert resumed.budget_usd == 5.0
    assert resumed.handoffs == EXPECTED_HANDOFFS
    assert script.count("triage") == 1 and script.count("strategy") == 1  # earlier stages were not repeated
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    assert len(ledger.entries) == _total_calls(fake_providers)


def test_resume_after_crash_repeats_only_the_current_stage(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    script = Script(sample_bodies, crash_at=("execution", 2))
    pipeline, run_id = _start(settings, fake_providers, script)

    with pytest.raises(Crash):
        pipeline.run(run_id)

    # On disk the run is still RUNNING at execution round 2, with round 1 fully persisted.
    crashed = pipeline.status(run_id)
    assert crashed.status == RunStatus.RUNNING
    assert (crashed.stage, crashed.round) == ("execution", 2)
    assert crashed.handoffs[-1] == "04-crosscheck"
    spent_before = pipeline.ledger_for(crashed).spent_usd
    calls_before = _total_calls(fake_providers)
    # The crashed call's spend is unknown, so it is recorded at its reserved worst case.
    crashed_entries = pipeline.ledger_for(crashed).entries
    assert len(crashed_entries) == calls_before
    assert crashed_entries[-1].error.startswith("Crash") and crashed_entries[-1].cost_usd == crashed_entries[-1].worst_case_usd

    # A fresh process: new Pipeline, ledger reloaded from ledger.jsonl.
    script.crash_at = None
    index = _pipeline(settings, fake_providers).run(run_id)

    assert index.status == RunStatus.COMPLETED, index.error
    assert index.handoffs == EXPECTED_HANDOFFS
    assert script.count("triage") == 1 and script.count("strategy") == 1
    assert script.count("execution") == 3  # round 1, the crashed round 2, the repeated round 2
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    assert ledger.spent_usd > spent_before
    assert len(ledger.entries) == _total_calls(fake_providers)
    for name in index.handoffs:
        assert validate_handoff(pipeline.vault.read_handoff(run_id, name)) == [], name


def test_one_repair_then_success(settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]) -> None:
    script = Script(sample_bodies, broken={"strategy": 1})
    pipeline, run_id = _start(settings, fake_providers, script)

    index = pipeline.run(run_id)

    assert index.status == RunStatus.COMPLETED, index.error
    assert [(role, repair) for role, kind, repair in script.requests if kind == "strategy"] == [
        ("chatgpt", False),
        ("chatgpt", True),
    ]
    repair_prompt = next(r for r in fake_providers.chatgpt.calls if _is_repair(r)).messages[-1].content
    assert "missing required section '## Options Considered'" in repair_prompt
    assert BROKEN_BODY.strip() in repair_prompt
    strategy = pipeline.vault.read_handoff(run_id, "02-strategy")
    assert validate_handoff(strategy) == []
    assert strategy.meta.cost_usd == pytest.approx(0.02)  # the original call plus its repair


def test_second_invalid_body_stops_the_run(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    script = Script(sample_bodies, broken={"strategy": 2})
    pipeline, run_id = _start(settings, fake_providers, script)

    index = pipeline.run(run_id)

    assert index.status == RunStatus.FAILED
    assert index.stage == "strategy"
    assert index.error and index.error.startswith("strategy:")
    assert script.count("strategy") == 2  # exactly one repair, then stop
    assert fake_providers.claude_code.calls == []  # nothing after strategy ran
    assert index.handoffs == ["01a-routing", "01-ingestion"]
    assert not pipeline.vault.has_note(run_id, "02-strategy")
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    assert len(ledger.entries) == _total_calls(fake_providers)  # both failed attempts were still billed

    # Once the agent behaves, resume continues from strategy without repeating ingestion.
    script.broken = {}
    resumed = pipeline.resume(run_id)
    assert resumed.status == RunStatus.COMPLETED, resumed.error
    assert script.count("triage") == 1
    assert resumed.handoffs == EXPECTED_HANDOFFS
