"""End-to-end offline runs: the real Pipeline, stage backends, vault and ledger, driven by scripted fakes.

The script answers by request content (which handoff kind the prompt asks for), so the tests exercise the
whole flow: ingestion -> strategy -> execution -> crosscheck (LOOP) -> execution r2 -> crosscheck r2 (PASS)
-> final (workspace export, clean-room reproduction, acceptance verdicts). Variants cover the loop cap
(``completed_with_issues``), unmet acceptance criteria and a failed clean room (``completed_with_issues`` too),
``maf export``, non-retryable provider errors (fail fast) and the Claude Code sandbox preflight (``SandboxedFake``:
the real ``ClaudeCodeProvider.preflight`` over a fake transport). The integration section runs the demo gaps end to
end: a fix-round file that must ship and rebuild, a document citing a non-source that the source audit raises, and
a pipeline wikilink that the lint raises. No network, no subprocess, zero cost.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import frontmatter
import pytest
from conftest import FakeProviders
from test_stages_base import SandboxedFake

from maf import cli
from maf import lint as _lint
from maf.config import Settings
from maf.handoff import HandoffKind, validate_handoff
from maf.ledger import Ledger
from maf.pipeline import Pipeline, ProvidersFactory
from maf.providers import CompletionRequest, ProviderError, Providers
from maf.providers.claude_code import PREFLIGHT_FILE
from maf.stages.final import CLEANROOM_SECTION, ISSUES_TAG, REPRO_EXIT, REPRO_LOG, cleanroom_dir
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
ACCEPTANCE_MET = (
    "## Acceptance\n\n"
    "- AC-1 [met]: the POSIX, FreeRTOS and QEMU suites pass (see [[03-execution]])\n"
    "- AC-2 [met]: `arm-none-eabi-size` reports text=1804\n"
)
"""Verdicts for the two acceptance criteria of the sample strategy."""


SANDBOX_ERROR = "Sandbox is required but failed to initialize: Failed to create bridge sockets after 5 attempts"
THESIS_REBUTTAL = (
    "## Summary\n\nAnswered.\n\n## Responses\n\n"
    "- GPT-1 [accept]: will fix the stress test\n"
    "  - Ref 20 gets ISBN 978-0-12-409210-6\n"
    "  - Ref 21 gets DOI 10.1088/0029-5515/39/12/301\n"
    "\n"
    "- GEM-1 [reject]: name is standard\n"
)


class Crash(BaseException):
    """Stands in for the process dying mid-stage (not caught by the pipeline's ``except Exception``)."""


class SandboxDown(ProviderError):
    """Stands in for ``maf.providers.base.SandboxUnavailable``: an infrastructure failure, never retryable."""

    def __init__(self, message: str = SANDBOX_ERROR, *, cost_usd: float = 0.0) -> None:
        super().__init__(message, provider="claude_code", cost_usd=cost_usd, retryable=False)


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
    ``(kind, n)`` whose n-th request raises ``Crash``. ``errors`` maps a kind to an exception its next request
    raises (once). ``stubborn`` keeps the critical issue coming every round (the loop cap is reached);
    ``rebuttal`` replaces the default rebuttal body. ``sandbox_down`` makes the sandbox preflight report the CLI's
    sandbox error instead of the probe file's digest. ``acceptance`` is the final note's ``## Acceptance`` section;
    the clean-room session exits with ``cleanroom_exit`` and reports ``cleanroom_missing``, or, with
    ``cleanroom_needs``, acts like a build that needs those files: exit 0 when all are in the clean room, else 2
    naming the missing ones. Every fix pass also creates ``tests/fix_added.c``, a file no execution note lists;
    every execution also writes ``extra_files`` (workspace path to content).
    """

    bodies: dict[str, str]
    broken: dict[str, int] = field(default_factory=dict)
    crash_at: tuple[str, int] | None = None
    errors: dict[str, BaseException] = field(default_factory=dict)
    stubborn: bool = False
    sandbox_down: bool = False
    rebuttal: str = "## Summary\n\nAnswered.\n\n## Responses\n\n- GPT-1 [accept]: will fix\n- GEM-1 [reject]: name is standard\n"
    acceptance: str = ACCEPTANCE_MET
    cleanroom_exit: int = 0
    cleanroom_missing: list[str] = field(default_factory=list)
    cleanroom_needs: tuple[str, ...] = ()
    extra_files: dict[str, str] = field(default_factory=dict)
    cleanroom_seen: set[str] = field(default_factory=set)
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
        if kind in self.errors:
            raise self.errors.pop(kind)
        if self.broken.get(kind, 0) >= self.count(kind):
            return BROKEN_BODY
        return self.answer(role, kind)

    def answer(self, role: str, kind: str) -> str | dict[str, object]:
        if kind == "triage":
            return TRIAGE
        if kind == "preflight":
            assert self.workspace is not None
            probe = (self.workspace / PREFLIGHT_FILE).read_bytes()
            return {"digest": SANDBOX_ERROR if self.sandbox_down else hashlib.sha256(probe).hexdigest()}
        if kind == "execution":
            self.executions += 1
            self.build()
            return self.bodies["execution"]
        if kind == "critique":
            first = self.executions == 1 or self.stubborn
            if first and role == "chatgpt":
                return "## Summary\n\nOne blocker.\n\n## Issues\n\n- [critical] GPT-1: stress test double-frees\n"
            if first and role == "gemini":
                return "## Summary\n\nStyle only.\n\n## Issues\n\n- [minor] GEM-1: rename tlsf_map\n"
            return CLEAN_CRITIQUE
        if kind == "rebuttal":
            return self.rebuttal
        if kind == "adjudication":
            return "## Summary\n\nRuled.\n\n## Rulings\n\n- GEM-1 [wontfix]: the name follows the paper\n"
        if kind == "fix_report":
            assert self.workspace is not None
            (self.workspace / "tests").mkdir(exist_ok=True)
            (self.workspace / "tests" / "fix_added.c").write_text("/* added by a fix pass */\n", encoding="utf-8")
            return {"fixed": [], "not_fixed": [{"id": "GPT-1", "reason": "needs a redesign"}], "summary": "tried"}
        if kind == "cleanroom":
            return self.cleanroom()
        if kind == "final":
            return f"{self.bodies['final'].rstrip()}\n\n{self.acceptance}"
        return self.bodies[kind]

    def cleanroom(self) -> dict[str, object]:
        """What the clean-room session does: run the command in the fresh copy and record its exit code."""
        assert self.workspace is not None
        room = cleanroom_dir(self.workspace)
        assert (room / "src" / "alloc.c").is_file() and not (room / ".maf").exists()
        self.cleanroom_seen = {p.relative_to(room).as_posix() for p in room.rglob("*") if p.is_file()}
        exit_code, missing = self.cleanroom_exit, list(self.cleanroom_missing)
        if self.cleanroom_needs:
            missing = [rel for rel in self.cleanroom_needs if not (room / rel).is_file()]
            exit_code = 2 if missing else 0
        log = "".join(f"make: *** No rule to make target '{rel}'\n" for rel in missing) + f"make: exit {exit_code}\n"
        (room / REPRO_LOG).write_text(log, encoding="utf-8")
        (room / REPRO_EXIT).write_text(f"{exit_code}\n", encoding="utf-8")
        return {"command": "make all", "exit_code": exit_code, "missing_paths": missing, "log_tail": log}

    def build(self) -> None:
        """What Claude Code would leave in the workspace: the artifacts the execution note lists."""
        assert self.workspace is not None
        (self.workspace / "src").mkdir(parents=True, exist_ok=True)
        (self.workspace / "src" / "alloc.c").write_text(f"/* tlsf pass {self.executions} */\n", encoding="utf-8")
        (self.workspace / "test").mkdir(exist_ok=True)
        (self.workspace / "test" / "posix_test.log").write_text("42 passed\n", encoding="utf-8")
        for rel, content in self.extra_files.items():
            (self.workspace / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.workspace / rel).write_text(content, encoding="utf-8")


def _pipeline(settings: Settings, fakes: FakeProviders, factory: ProvidersFactory | None = None) -> Pipeline:
    ticks = iter(range(100_000))
    return Pipeline(
        settings,
        providers_factory=factory or fakes.factory(),
        clock=lambda: datetime(2026, 9, 28, 12, 0) + timedelta(seconds=next(ticks)),
    )


def _sandboxed_pipeline(settings: Settings, fakes: FakeProviders, script: Script) -> tuple[Pipeline, SandboxedFake]:
    """A pipeline whose claude_code fake runs the real sandbox preflight. Like ``build_providers``, the factory binds
    Claude Code to the run's workspace and hands every ``_advance`` (one process's pass) an unverified sandbox."""
    code = SandboxedFake(name="claude_code", agent="claude", cost_per_call=0.50, worst_case_usd=2.0)
    fakes.claude_code = code
    script.install(fakes)

    def factory(_settings: Settings, workspace: Path) -> Providers:
        code.workspace = script.workspace = workspace
        code.sandbox_verified = False
        return fakes.as_providers()

    return _pipeline(settings, fakes, factory), code


def _code_kinds(script: Script) -> list[str]:
    return [kind for role, kind, _ in script.requests if role == "claude_code"]


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
    # The deliverables are the workspace tree after the latest pass, including the file the round-1 fix pass added
    # (no execution note lists it), and nothing from .maf/ (prompt copies, the clean room).
    paths = vault.paths(run_id)
    assert (paths.deliverables / "src" / "alloc.c").read_text() == "/* tlsf pass 2 */\n"
    assert (paths.deliverables / "tests" / "fix_added.c").is_file()
    assert not (paths.deliverables / ".maf").exists()
    final = vault.read_handoff(run_id, "05-final")
    assert "[[04-crosscheck-r2]]" in final.section("Provenance")
    assert f"`runs/{run_id}/deliverables/tests/`" in final.section("Deliverables")
    # Acceptance: both strategy criteria met, and the clean room rebuilt the export.
    acceptance = final.section("Acceptance")
    assert "- AC-1 [met]: All tests pass on POSIX, FreeRTOS and QEMU." in acceptance
    assert "- AC-2 [met]:" in acceptance and "- clean-room [met]:" in acceptance
    assert final.section(CLEANROOM_SECTION).startswith("- Result: passed\n- Command: `make all`")
    assert (index.criteria_unmet, index.unmet_criteria) == (0, [])
    assert index.export_note and index.export_note.startswith("final: ")

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
    assert code_calls == 4  # execution, fix pass, execution r2 (round 2 has no issues, so no fix pass), clean room
    assert [(e.stage, e.purpose) for e in ledger.entries if e.provider == "claude_code"][-1] == ("final", "cleanroom")
    assert ledger.by_agent() == pytest.approx(
        {
            "chatgpt": 0.01 * len(fake_providers.chatgpt.calls),
            "gemini": 0.01 * len(fake_providers.gemini.calls),
            "claude": 0.01 * len(fake_providers.claude.calls) + 0.50 * code_calls,
        }
    )
    assert set(ledger.by_stage()) == {"ingestion", "strategy", "execution", "crosscheck", "final"}
    # Each note's cost is what its own calls spent (05-final includes the clean room), so the notes add up to the
    # ledger total.
    note_costs = sum(vault.read_handoff(run_id, name).meta.cost_usd for name in index.handoffs)
    assert note_costs == pytest.approx(ledger.spent_usd)


def test_budget_stop_then_resume_with_higher_budget(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    script = Script(sample_bodies)
    # Ingestion + strategy spend $0.03 and execution $0.50; three concurrent critiques each reserve a
    # $0.05 worst case, so they cannot all fit under $0.60. (The budget is far below the work-session minimum, which
    # is not what this test is about: tests/test_budget_resilience.py covers it.)
    settings = settings.model_copy(update={"claude_code_min_session_usd": 0.1})
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


# --------------------------------------------------------------------------- loop cap and fail-fast infra errors


def test_loop_cap_ends_completed_with_issues(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    """GPT-1 stays unfixed in every round: final still runs, but the run says completed_with_issues everywhere.
    The rebuttal uses indented sub-bullets (the thesis-run shape) and needs no repair."""
    script = Script(sample_bodies, stubborn=True, rebuttal=THESIS_REBUTTAL)
    pipeline, run_id = _start(settings, fake_providers, script)

    index = pipeline.run(run_id)

    assert index.status == RunStatus.COMPLETED_WITH_ISSUES, index.error
    assert (index.stage, index.round, index.unresolved_critical, index.error) == ("final", 3, 1, None)
    assert script.count("execution") == 3 and script.count("final") == 1
    assert not any(repair for _, _, repair in script.requests)  # sub-bullets under responses parse as-is
    vault = pipeline.vault
    for name in index.handoffs:
        assert validate_handoff(vault.read_handoff(run_id, name)) == [], name

    last = vault.read_handoff(run_id, "04-crosscheck-r3")
    assert last.section("Verdict") == "LOOP" and last.meta.to == "final"
    assert "Loop cap reached" in last.section("Summary")
    assert "GPT-1: stress test double-frees" in last.section("Unresolved Critical")

    final = vault.read_handoff(run_id, "05-final")
    assert final.section("Summary").startswith("> [!warning] Run status: completed_with_issues")
    assert "[[04-crosscheck-r3]]" in final.section("Summary")
    assert "- [critical] GPT-1: stress test double-frees" in final.section("Limitations")
    assert ISSUES_TAG in final.meta.tags
    final_prompt = next(r for r in fake_providers.claude.calls if "## Output format: final handoff" in r.messages[-1].content)
    assert "ends with status `completed_with_issues`" in final_prompt.messages[-1].content

    post = frontmatter.load(vault.paths(run_id).run_md)
    assert (post["status"], post["unresolved_critical"]) == ("completed_with_issues", 1)
    status_block = post.content.split("## Status\n\n", 1)[1]
    assert status_block.startswith("> [!warning] Completed with 1 unresolved critical issue(s)")
    assert "[[04-crosscheck-r3]]" in status_block.split("\n\n", 1)[0]
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    assert len(ledger.entries) == _total_calls(fake_providers)


def test_resume_cli_on_loop_capped_run_exits_2_and_refuses_without_extra_round(
    settings: Settings,
    fake_providers: FakeProviders,
    sample_bodies: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = Script(sample_bodies, stubborn=True)
    pipeline, run_id = _start(settings, fake_providers, script)
    monkeypatch.setattr(cli, "make_pipeline", lambda _settings: pipeline)
    argv = ["--vault", str(settings.vault_path), "--workspaces", str(settings.workspaces_path), "resume", run_id]

    assert cli.main(argv) == 2  # runs the pending run to the loop cap
    captured = capsys.readouterr()
    assert captured.out.strip().splitlines()[-1] == str(pipeline.vault.paths(run_id).note("05-final"))
    last = captured.err.strip().splitlines()[-1]
    assert last.startswith("completed with issues: 1 unresolved critical issue(s)")
    assert str(pipeline.vault.paths(run_id).note("04-crosscheck-r3")) in last

    calls = _total_calls(fake_providers)
    assert cli.main(argv) == 2  # terminal: refused, nothing spent
    assert "nothing to resume" in capsys.readouterr().err
    assert _total_calls(fake_providers) == calls

    script.stubborn = False  # the extra pass comes back clean
    assert cli.main([*argv, "--extra-round"]) == 0
    index = pipeline.status(run_id)
    assert (index.status, index.round, index.unresolved_critical) == (RunStatus.COMPLETED, 4, 0)
    assert index.handoffs[-2:] == ["04-crosscheck-r4", "05-final"]
    r4_prompt = [r for r in fake_providers.claude_code.calls if _kind_of(r) == "execution"][-1].messages[-1].content
    assert "GPT-1" in r4_prompt  # the extra execution pass saw the round-3 unresolved issue
    assert "[!warning]" not in pipeline.vault.read_handoff(run_id, "05-final").section("Summary")


def test_non_retryable_provider_error_in_execution_fails_fast(
    settings: Settings,
    fake_providers: FakeProviders,
    sample_bodies: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A broken Claude Code sandbox in execution round 1 ends the run as FAILED at once: no critiques, no
    loop-back, the error in run.md and on the CLI, and the partial spend recorded."""
    script = Script(sample_bodies, errors={"execution": SandboxDown(cost_usd=0.37)})
    pipeline, run_id = _start(settings, fake_providers, script)
    monkeypatch.setattr(cli, "make_pipeline", lambda _settings: pipeline)
    argv = ["--vault", str(settings.vault_path), "--workspaces", str(settings.workspaces_path), "resume", run_id]

    assert cli.main(argv) == cli.EXIT_FAILED == 1

    index = pipeline.status(run_id)
    assert index.status == RunStatus.FAILED
    assert (index.stage, index.round) == ("execution", 1)
    assert index.error == f"execution: SandboxDown: {SANDBOX_ERROR}"
    assert len(fake_providers.claude_code.calls) == 1  # never retried
    assert script.count("critique") == 0 and script.count("fix_report") == 0
    assert index.handoffs == ["01a-routing", "01-ingestion", "02-strategy"]
    assert "crosscheck" not in index.completed_stages
    # Spend: the failed call's partial cost is in the ledger and mirrored into run.md.
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    failed_entry = ledger.entries[-1]
    assert (failed_entry.provider, failed_entry.cost_usd) == ("claude_code", pytest.approx(0.37))
    assert failed_entry.error == f"SandboxDown: {SANDBOX_ERROR}"
    assert index.spend_by_provider["claude_code"] == pytest.approx(0.37)
    post = frontmatter.load(pipeline.vault.paths(run_id).run_md)
    assert post["status"] == "failed" and post["error"] == index.error
    assert f"- Error: execution: SandboxDown: {SANDBOX_ERROR}" in post.content
    last = capsys.readouterr().err.strip().splitlines()[-1]
    assert last == f"failed: execution: SandboxDown: {SANDBOX_ERROR} (spent ${index.spent_usd:.4f} of $25.00)"

    # Once the environment is fixed, resume continues from execution round 1 without repeating earlier stages.
    assert cli.main(argv) == 0
    resumed = pipeline.status(run_id)
    assert resumed.status == RunStatus.COMPLETED, resumed.error
    assert resumed.handoffs == EXPECTED_HANDOFFS
    assert script.count("triage") == 1 and script.count("strategy") == 1
    assert _assert_ledger_mirrored(pipeline, run_id).spent_usd > index.spent_usd


def test_non_retryable_provider_error_in_fix_pass_never_loops_back(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    script = Script(sample_bodies, errors={"fix_report": SandboxDown(cost_usd=0.05)})
    pipeline, run_id = _start(settings, fake_providers, script)

    index = pipeline.run(run_id)

    assert index.status == RunStatus.FAILED
    assert (index.stage, index.round) == ("crosscheck", 1)
    assert index.error == f"crosscheck: SandboxDown: {SANDBOX_ERROR}"
    assert script.count("execution") == 1 and script.count("fix_report") == 1
    assert "03-execution-r2" not in index.handoffs and "04-crosscheck" not in index.handoffs
    assert index.unresolved_critical == 0
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    assert len(ledger.entries) == _total_calls(fake_providers)  # the critiques and the failed fix pass were billed
    assert ledger.entries[-1].error == f"SandboxDown: {SANDBOX_ERROR}"


# --------------------------------------------------------------------------- Claude Code sandbox preflight


def test_code_run_preflights_the_sandbox_exactly_once(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    """Happy path with one loop: one preflight before the first Claude Code session; the fix pass and execution
    round 2 reuse the verified sandbox. The preflight is metered (execution/preflight) but belongs to no note."""
    script = Script(sample_bodies)
    pipeline, code = _sandboxed_pipeline(settings, fake_providers, script)
    run_id = pipeline.create(BRIEF).run_id

    index = pipeline.run(run_id)

    assert index.status == RunStatus.COMPLETED, index.error
    assert index.handoffs == EXPECTED_HANDOFFS
    assert _code_kinds(script) == ["preflight", "execution", "fix_report", "execution", "cleanroom"]
    assert code.preflights == 1 and code.sandbox_verified is True
    assert not (pipeline.vault.paths(run_id).workspace / PREFLIGHT_FILE).exists()
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    assert len(ledger.entries) == _total_calls(fake_providers)
    assert [(e.stage, e.purpose, e.cost_usd) for e in ledger.entries if e.provider == "claude_code"] == [
        ("execution", "preflight", 0.04),
        ("execution", "execution", 0.50),
        ("crosscheck", "fixes", 0.50),
        ("execution", "execution", 0.50),
        ("final", "cleanroom", 0.50),
    ]
    note_costs = sum(pipeline.vault.read_handoff(run_id, name).meta.cost_usd for name in index.handoffs)
    assert note_costs == pytest.approx(ledger.spent_usd - 0.04)


def test_failed_sandbox_preflight_fails_the_run_at_execution_round_1(
    settings: Settings,
    fake_providers: FakeProviders,
    sample_bodies: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The 2026-09-28 incident: the sandbox cannot start. The preflight catches it, so the run is FAILED before any
    execution session, critique or loop is paid for; only the cheap preflight is spent (and recorded)."""
    script = Script(sample_bodies, sandbox_down=True)
    pipeline, code = _sandboxed_pipeline(settings, fake_providers, script)
    run_id = pipeline.create(BRIEF).run_id
    workspace = pipeline.vault.paths(run_id).workspace
    monkeypatch.setattr(cli, "make_pipeline", lambda _settings: pipeline)
    argv = ["--vault", str(settings.vault_path), "--workspaces", str(settings.workspaces_path), "resume", run_id]

    assert cli.main(argv) == cli.EXIT_FAILED

    index = pipeline.status(run_id)
    assert index.status == RunStatus.FAILED
    assert (index.stage, index.round, index.mode) == ("execution", 1, "code")
    assert index.error is not None
    assert index.error.startswith("execution: SandboxUnavailable: Claude Code sandbox preflight failed")
    assert SANDBOX_ERROR in index.error
    assert _code_kinds(script) == ["preflight"]  # no execution session, no retry
    assert script.count("critique") == 0 and script.count("fix_report") == 0
    assert index.handoffs == ["01a-routing", "01-ingestion", "02-strategy"]
    assert "execution" not in index.completed_stages and "crosscheck" not in index.completed_stages
    assert not (workspace / PREFLIGHT_FILE).exists()
    assert not (workspace / ".maf" / "execution-r1.md").exists()  # stopped before the prompt was even written
    # Spend: ingestion and strategy ($0.03) plus the preflight, metered as a normal execution/preflight entry.
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    assert len(ledger.entries) == _total_calls(fake_providers)
    preflight = ledger.entries[-1]
    assert (preflight.stage, preflight.purpose, preflight.provider, preflight.error) == (
        "execution", "preflight", "claude_code", "",
    )
    assert index.spent_usd == pytest.approx(0.03 + 0.04)
    assert index.spend_by_provider["claude_code"] == pytest.approx(0.04)
    post = frontmatter.load(pipeline.vault.paths(run_id).run_md)
    assert post["status"] == "failed" and post["error"] == index.error
    last = capsys.readouterr().err.strip().splitlines()[-1]
    assert last.startswith("failed: execution: SandboxUnavailable: Claude Code sandbox preflight failed")
    assert last.endswith("(spent $0.0700 of $25.00)")

    # Once the sandbox works, a new process checks it again and the run completes from execution round 1.
    script.sandbox_down = False
    assert cli.main(argv) == cli.EXIT_OK
    resumed = pipeline.status(run_id)
    assert resumed.status == RunStatus.COMPLETED, resumed.error
    assert resumed.handoffs == EXPECTED_HANDOFFS
    assert script.count("triage") == 1 and script.count("strategy") == 1
    assert code.preflights == 2
    assert _code_kinds(script) == ["preflight", "preflight", "execution", "fix_report", "execution", "cleanroom"]
    assert len(_assert_ledger_mirrored(pipeline, run_id).entries) == _total_calls(fake_providers)


def test_cli_run_to_the_loop_cap_exits_2_as_completed_with_issues(
    settings: Settings,
    fake_providers: FakeProviders,
    sample_bodies: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``maf run`` in code mode where GPT-1 is never fixed: three execution passes and three fix passes on one
    verified sandbox, then final, status completed_with_issues and exit code 2."""
    script = Script(sample_bodies, stubborn=True)
    pipeline, code = _sandboxed_pipeline(settings, fake_providers, script)
    monkeypatch.setattr(cli, "make_pipeline", lambda _settings: pipeline)
    base = ["--vault", str(settings.vault_path), "--workspaces", str(settings.workspaces_path)]

    assert cli.main([*base, "run", BRIEF]) == cli.EXIT_WITH_ISSUES == 2

    out, err = capsys.readouterr()
    run_id = out.splitlines()[0]
    index = pipeline.status(run_id)
    assert index.status == RunStatus.COMPLETED_WITH_ISSUES, index.error
    assert (index.stage, index.round, index.unresolved_critical, index.error) == ("final", 3, 1, None)
    assert index.handoffs[-2:] == ["04-crosscheck-r3", "05-final"]
    assert _code_kinds(script) == ["preflight"] + ["execution", "fix_report"] * 3 + ["cleanroom"]
    paths = pipeline.vault.paths(run_id)
    assert out.strip().splitlines()[-1] == str(paths.note("05-final"))
    last = err.strip().splitlines()[-1]
    assert last.startswith("completed with issues: 1 unresolved critical issue(s) after the cross-check loop cap")
    assert str(paths.note("04-crosscheck-r3")) in last and f"maf resume {run_id} --extra-round" in last
    assert frontmatter.load(paths.run_md)["status"] == "completed_with_issues"
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    assert len(ledger.entries) == _total_calls(fake_providers)

    assert cli.main([*base, "status", run_id]) == cli.EXIT_OK
    assert "unresolved critical: 1" in capsys.readouterr().out


def test_process_resumed_into_crosscheck_verifies_the_sandbox_before_the_critiques(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    """A budget stop at crosscheck, then ``resume``: the new process has not seen execution's preflight, so the
    crosscheck stage runs its own (ledger stage crosscheck) before paying for critiques; round 2 reuses it."""
    script = Script(sample_bodies)
    settings = settings.model_copy(update={"claude_code_min_session_usd": 0.1})  # as in the budget-stop test
    pipeline, code = _sandboxed_pipeline(settings, fake_providers, script)
    # $0.03 ingestion + strategy, $0.04 preflight, $0.50 execution; the three $0.05 critique reservations cannot fit.
    run_id = pipeline.create(BRIEF, budget_usd=0.64).run_id

    stopped = pipeline.run(run_id)
    assert stopped.status == RunStatus.BUDGET_EXCEEDED and (stopped.stage, stopped.round) == ("crosscheck", 1)
    assert _code_kinds(script) == ["preflight", "execution"]
    before_resume = len(script.requests)

    resumed = pipeline.resume(run_id, budget_usd=5.0)

    assert resumed.status == RunStatus.COMPLETED, resumed.error
    assert resumed.handoffs == EXPECTED_HANDOFFS
    assert _code_kinds(script) == ["preflight", "execution", "preflight", "fix_report", "execution", "cleanroom"]
    first_after_resume = [(role, kind) for role, kind, _ in script.requests[before_resume : before_resume + 2]]
    assert first_after_resume[0] == ("claude_code", "preflight")  # verified before any critique of the resumed pass
    assert first_after_resume[1][1] == "critique"
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    assert len(ledger.entries) == _total_calls(fake_providers)
    assert [(e.stage, e.purpose) for e in ledger.entries if e.provider == "claude_code"] == [
        ("execution", "preflight"),
        ("execution", "execution"),
        ("crosscheck", "preflight"),
        ("crosscheck", "fixes"),
        ("execution", "execution"),
        ("final", "cleanroom"),
    ]


# --------------------------------------------------------------------------- acceptance gate, clean room, maf export


def test_failed_clean_room_ends_completed_with_issues_and_exits_2(
    settings: Settings,
    fake_providers: FakeProviders,
    sample_bodies: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The allocator-demo gap: the exported tree does not rebuild (a file is missing). Every strategy criterion is
    met and nothing critical is open, yet the run ends completed_with_issues, in run.md, 05-final and the CLI."""
    script = Script(sample_bodies, cleanroom_exit=2, cleanroom_missing=["tests/host_sizes.c"])
    pipeline, code = _sandboxed_pipeline(settings, fake_providers, script)
    monkeypatch.setattr(cli, "make_pipeline", lambda _settings: pipeline)
    base = ["--vault", str(settings.vault_path), "--workspaces", str(settings.workspaces_path)]

    assert cli.main([*base, "run", BRIEF]) == cli.EXIT_WITH_ISSUES

    out, err = capsys.readouterr()
    run_id = out.splitlines()[0]
    index = pipeline.status(run_id)
    assert index.status == RunStatus.COMPLETED_WITH_ISSUES, index.error
    assert (index.unresolved_critical, index.criteria_unmet) == (0, 1)
    assert index.unmet_criteria == [
        "clean-room [unmet]: Clean-room reproduction: the reproduction command the deliverables document succeeds "
        "in a fresh copy of the exported deliverables."
    ]
    paths = pipeline.vault.paths(run_id)
    last = err.strip().splitlines()[-1]
    assert last.startswith(f"completed with issues: 1 acceptance criterion not met (clean-room); see {paths.note('05-final')}")
    assert "unresolved critical" not in last and f"maf resume {run_id} --extra-round" in last

    final = pipeline.vault.read_handoff(run_id, "05-final")
    assert final.section("Summary").startswith(
        "> [!warning] Run status: completed_with_issues\n"
        "> 1 acceptance criterion is not met (clean-room); see Acceptance and Limitations."
    )
    assert ISSUES_TAG in final.meta.tags
    cleanroom = final.section(CLEANROOM_SECTION)
    assert "- Result: failed: `make all` exited with 2 in the clean room; the exported deliverables lack " in cleanroom
    assert "`tests/host_sizes.c`" in cleanroom and "make: exit 2" in cleanroom
    assert "- clean-room [unmet]:" in final.section("Acceptance")
    assert "- clean-room [unmet]: Clean-room reproduction" in final.section("Limitations")
    final_prompt = next(r for r in fake_providers.claude.calls if _kind_of(r) == "final").messages[-1].content
    assert "## Clean-room reproduction (run by maf)" in final_prompt and "exited with 2" in final_prompt

    post = frontmatter.load(paths.run_md)
    assert (post["status"], post["criteria_unmet"], post["unresolved_critical"]) == ("completed_with_issues", 1, 0)
    callout = post.content.split("## Status\n\n", 1)[1].split("\n\n", 1)[0]
    assert callout.startswith("> [!warning] Completed with 1 acceptance criterion not met\n")
    assert "> - clean-room [unmet]: Clean-room reproduction" in callout
    assert code.preflights == 1  # the clean room reused the sandbox verified by execution
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    assert len(ledger.entries) == _total_calls(fake_providers)

    assert cli.main([*base, "status", run_id]) == cli.EXIT_OK
    status = capsys.readouterr().out
    assert "criteria unmet: 1\n  - clean-room [unmet]: Clean-room reproduction" in status
    assert "unresolved critical" not in status


def test_unmet_acceptance_criterion_ends_completed_with_issues_until_an_extra_round_meets_it(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    """The thesis-demo gap: the final report itself says a hard criterion is unmet, so the run must not be completed."""
    unmet = ACCEPTANCE_MET.replace("- AC-2 [met]: `arm-none-eabi-size` reports text=1804", "- AC-2 [unmet]: 2210 bytes")
    script = Script(sample_bodies, acceptance=unmet)
    pipeline, run_id = _start(settings, fake_providers, script)

    index = pipeline.run(run_id)

    assert index.status == RunStatus.COMPLETED_WITH_ISSUES, index.error
    assert (index.unresolved_critical, index.criteria_unmet) == (0, 1)
    assert index.unmet_criteria == ["AC-2 [unmet]: `.text` < 2048 bytes on Cortex-M3 at `-Os`."]
    final = pipeline.vault.read_handoff(run_id, "05-final")
    assert "- AC-2 [unmet]: `.text` < 2048 bytes on Cortex-M3 at `-Os`.\n  - Evidence: 2210 bytes" in final.section(
        "Acceptance"
    )
    assert "- AC-2 [unmet]: `.text` < 2048 bytes" in final.section("Limitations")

    script.acceptance = ACCEPTANCE_MET
    again = pipeline.resume(run_id, extra_round=True)
    assert again.status == RunStatus.COMPLETED, again.error
    assert (again.round, again.criteria_unmet, again.unmet_criteria) == (3, 0, [])
    assert again.handoffs[-1] == "05-final"
    # The extra execution pass was told which criterion final found unmet, with final's evidence.
    r3 = [r for r in fake_providers.claude_code.calls if _kind_of(r) == "execution"][-1].messages[-1].content
    unmet_block = r3.split("## Acceptance criteria not met at the last final report ([[05-final]])", 1)[1]
    assert "- AC-2 [unmet]: `.text` < 2048 bytes" in unmet_block and "Evidence: 2210 bytes" in unmet_block
    assert "[[05-final]]" in pipeline.vault.read_handoff(run_id, "03-execution-r3").meta.inputs


def test_maf_export_reexports_an_existing_run_from_its_workspace(
    settings: Settings,
    fake_providers: FakeProviders,
    sample_bodies: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``maf export`` after the run: files added to the workspace since final ship, stale ones go, run.md notes it,
    and no provider is called."""
    script = Script(sample_bodies)
    pipeline, run_id = _start(settings, fake_providers, script)
    assert pipeline.run(run_id).status == RunStatus.COMPLETED
    paths = pipeline.vault.paths(run_id)
    (paths.workspace / "logs").mkdir()
    (paths.workspace / "logs" / "host_sizes.log").write_text("sizes\n", encoding="utf-8")
    (paths.workspace / "tests" / "fix_added.c").unlink()
    calls = _total_calls(fake_providers)
    monkeypatch.setattr(cli, "make_pipeline", lambda _settings: pipeline)
    base = ["--vault", str(settings.vault_path), "--workspaces", str(settings.workspaces_path)]

    assert cli.main([*base, "export", run_id]) == cli.EXIT_OK

    out = capsys.readouterr().out
    assert out.startswith("exported 3 file(s)") and str(paths.deliverables) in out
    assert (paths.deliverables / "logs" / "host_sizes.log").read_text() == "sizes\n"
    assert not (paths.deliverables / "tests").exists()
    assert _total_calls(fake_providers) == calls
    index = pipeline.status(run_id)
    assert index.status == RunStatus.COMPLETED
    assert index.export_note == "maf export: 3 file(s), 34 B"
    post = frontmatter.load(paths.run_md)
    assert f"- Deliverables exported: {index.exported_at.isoformat()} (maf export: 3 file(s), 34 B)" in post.content  # type: ignore[union-attr]



# --------------------------------------------------------------------------- integration: export, sources, lint


CLEANROOM_UNMET = (
    "clean-room [unmet]: Clean-room reproduction: the reproduction command the deliverables document succeeds in a "
    "fresh copy of the exported deliverables."
)


@pytest.mark.parametrize(
    ("needs", "status", "unmet"),
    [
        (("tests/fix_added.c",), RunStatus.COMPLETED, []),
        (("tests/fix_added.c", "build/config.h"), RunStatus.COMPLETED_WITH_ISSUES, [CLEANROOM_UNMET]),
    ],
    ids=["fix-round-file-ships", "build-output-is-not-shipped"],
)
def test_clean_room_rebuilds_exactly_the_exported_tree(
    settings: Settings,
    fake_providers: FakeProviders,
    sample_bodies: dict[str, str],
    needs: tuple[str, ...],
    status: RunStatus,
    unmet: list[str],
) -> None:
    """The allocator-demo gap, end to end. The round-1 fix pass creates ``tests/fix_added.c``, which no execution
    note lists; the build also leaves ``build/config.h``, which is build output. The clean room is a fresh copy of
    what was exported, and its "make all" needs the listed files: the fix-round file ships, so a build that needs
    only it passes; one that also needs build output fails, and the run is ``completed_with_issues`` for it."""
    script = Script(sample_bodies, cleanroom_needs=needs, extra_files={"build/config.h": "#define N 4\n"})
    pipeline, code = _sandboxed_pipeline(settings, fake_providers, script)
    run_id = pipeline.create(BRIEF).run_id

    index = pipeline.run(run_id)

    assert index.status == status, index.error
    vault, paths = pipeline.vault, pipeline.vault.paths(run_id)
    listed = [vault.read_handoff(run_id, n).section("Artifacts") for n in ("03-execution", "03-execution-r2")]
    assert not any("fix_added.c" in artifacts for artifacts in listed)
    assert "- created: `tests/fix_added.c`" in vault.read_handoff(run_id, "04-crosscheck").section("Changed Files")
    assert (paths.deliverables / "tests" / "fix_added.c").is_file()
    assert not (paths.deliverables / "build").exists() and (paths.workspace / "build" / "config.h").is_file()
    assert "tests/fix_added.c" in script.cleanroom_seen and "build/config.h" not in script.cleanroom_seen
    assert (index.criteria_unmet, index.unmet_criteria, index.unresolved_critical) == (len(unmet), unmet, 0)
    final = vault.read_handoff(run_id, "05-final")
    cleanroom = final.section(CLEANROOM_SECTION)
    if unmet:
        assert cleanroom.startswith("- Result: failed: `make all` exited with 2 in the clean room; the exported ")
        assert "- Missing paths: `build/config.h`" in cleanroom and "No rule to make target 'build/config.h'" in cleanroom
        assert final.section("Summary").startswith("> [!warning] Run status: completed_with_issues")
        assert frontmatter.load(paths.run_md)["criteria_unmet"] == 1
    else:
        assert cleanroom.startswith("- Result: passed\n- Command: `make all`")
        assert "- clean-room [met]:" in final.section("Acceptance") and ISSUES_TAG not in final.meta.tags
    assert _code_kinds(script) == ["preflight", "execution", "fix_report", "execution", "cleanroom"]
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    note_costs = sum(vault.read_handoff(run_id, name).meta.cost_usd for name in index.handoffs)
    assert note_costs == pytest.approx(ledger.spent_usd - 0.04)  # every note's spend, minus the preflight


DOC_TRIAGE = {
    "summary": "A short survey of O(1) allocators.",
    "execution_mode": "prose",
    "gemini_instructions": "Find the TLSF paper and its worst-case bounds.",
    "search_queries": ["TLSF allocator Masmano 2004"],
    "deliverable": "A two-page survey with verified references",
}
DOC_STRATEGY = """## Summary

A short survey built on the TLSF paper.

## Options Considered

1. Survey. 2. Tutorial.

## Chosen Strategy

Survey: it fits two pages.

## Execution Brief

Describe TLSF and cite only verified sources.

## Acceptance Criteria

- AC-1 [hard]: Every reference passes the source audit.
- AC-2 [soft]: The survey states the worst-case cost of malloc and free.

## Risks

- Few primary sources.
"""
MASMANO = (
    '1. M. Masmano, I. Ripoll, A. Crespo, J. Real, "TLSF: a New Dynamic Memory Allocator for Real-Time Systems", '
    "ECRTS 2004, pp. 79-86. doi:10.1109/EMRTS.2004.1311009"
)
LEAKY_SURVEY = f"""#### Design

TLSF finds a free block in constant time with two bitmap levels [1]. Its worst-case bound of 168 cycles is taken
from [2].

#### References

{MASMANO}
2. Gemini, survey of O(1) allocators (AI-generated summary), 2026.
"""
CLEAN_SURVEY = f"""#### Design

TLSF finds a free block in constant time with two bitmap levels [1]. We measure no cycle bound ourselves, so none
is given.

#### References

{MASMANO}
"""


def _prose_execution(survey: str) -> str:
    return (
        "## Summary\n\nA short survey.\n\n## Artifacts\n\n### O(1) Allocators\n\n" + survey
        + "\n## Implementation Notes\n\nWritten from the verified sources.\n\n## Verification\n\n"
        "Every citation checked against the ingestion excerpts.\n\n## Known Limitations\n\nNone.\n"
    )


@dataclass
class DocScript:
    """A prose run whose first document cites an AI summary as reference [2]. The source audit (Gemini) marks it
    ``internal_note``, so SRC-1 is critical. With ``fix_pass`` the round-1 fixer replaces the citation; otherwise
    it reports SRC-1 not fixed, the run loops, and execution round 2 writes the clean document."""

    bodies: dict[str, str]
    fix_pass: bool
    requests: list[tuple[str, str]] = field(default_factory=list)  # (role, kind)
    prompts: dict[str, list[str]] = field(default_factory=dict)  # kind -> prompts, in order
    workspace: Path | None = None
    executions: int = 0

    def install(self, fakes: FakeProviders) -> None:
        for role in ("chatgpt", "gemini", "claude", "claude_code"):
            getattr(fakes, role).default = lambda r, role=role: self.reply(role, r)

    def count(self, kind: str) -> int:
        return len(self.prompts.get(kind, []))

    def reply(self, role: str, request: CompletionRequest) -> str | dict[str, object]:
        kind = _kind_of(request)
        self.requests.append((role, kind))
        self.prompts.setdefault(kind, []).append(request.messages[-1].content)
        if kind == "triage":
            return DOC_TRIAGE
        if kind == "strategy":
            return DOC_STRATEGY
        if kind == "execution":
            self.executions += 1
            return _prose_execution(LEAKY_SURVEY if self.executions == 1 else CLEAN_SURVEY)
        if kind == "source_audit":
            leaky = "Gemini, survey of O(1) allocators" in request.messages[-1].content
            return self.audit(leaky)
        if kind == "critique":
            return CLEAN_CRITIQUE
        if kind == "rebuttal":
            return "## Summary\n\nAnswered.\n\n## Responses\n\n- SRC-1 [accept]: cite only the verified TLSF paper\n"
        if kind == "fix_report":
            assert self.workspace is not None
            current = (self.workspace / "document.md").read_text(encoding="utf-8")
            if self.fix_pass:
                fixed = current.split("## Design", 1)[0] + CLEAN_SURVEY.replace("####", "##")
                return {"fixed": ["SRC-1"], "not_fixed": [], "summary": "Replaced [2].", "document": fixed}
            reason = "no verified source gives a cycle bound; the claim needs rewriting in execution"
            return {"fixed": [], "not_fixed": [{"id": "SRC-1", "reason": reason}], "summary": "", "document": current}
        if kind == "final":
            acceptance = (
                "## Acceptance\n\n- AC-1 [met]: the source audit verified the one reference (see [[04-crosscheck]])\n"
                "- AC-2 [partial]: malloc and free are O(1); no cycle bound is given\n"
            )
            return f"{self.bodies['final'].rstrip()}\n\n{acceptance}"
        return self.bodies[kind]

    @staticmethod
    def audit(leaky: bool) -> dict[str, object]:
        refs: list[dict[str, object]] = [
            {
                "document": "document.md",
                "reference": MASMANO,
                "verdict": "verified",
                "claims_checked": ["constant time with two bitmap levels"],
                "finding": "opened https://doi.org/10.1109/EMRTS.2004.1311009: Section 3 describes the bitmaps",
                "correction": "",
            }
        ]
        if leaky:
            refs.append({
                "document": "document.md",
                "reference": "2. Gemini, survey of O(1) allocators (AI-generated summary), 2026.",
                "verdict": "internal_note",
                "claims_checked": ["worst-case bound of 168 cycles"],
                "finding": "an AI model's summary, not a publication; no record of the 168-cycle bound was found",
                "correction": "cite the TLSF paper for what it states, and drop the cycle bound",
            })
        return {"references": refs, "unaudited": 0, "summary": f"Audited {len(refs)} reference(s)."}


@pytest.mark.parametrize("fix_pass", [True, False], ids=["fixed-by-the-fix-pass", "fixed-by-execution-round-2"])
def test_document_citing_an_internal_note_is_audited_fixed_and_completes(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str], fix_pass: bool
) -> None:
    """The thesis-demo gap, end to end: a document cites a non-source (an AI summary). Gemini's web audit raises it as
    the critical SRC-1, which the debate accepts and the loop fixes (in the fix pass, or in execution round 2 after a
    LOOP). The next audit verifies the remaining reference: after the fix pass it audits the revised document before
    the verdict (no fixer's claim is taken on trust), in round 2 it is that round's audit. Final's ``source-audit``
    gate reads the latest audit of the shipped text, and the run completes."""
    script = DocScript(sample_bodies, fix_pass=fix_pass)
    script.install(fake_providers)
    pipeline = _pipeline(settings, fake_providers)
    run_id = pipeline.create("Write a short survey of O(1) allocators.").run_id
    script.workspace = pipeline.vault.paths(run_id).workspace

    index = pipeline.run(run_id)

    assert index.status == RunStatus.COMPLETED, index.error
    assert (index.mode, index.round, index.unresolved_critical, index.criteria_unmet) == (
        "prose", 1 if fix_pass else 2, 0, 0,
    )
    vault, paths = pipeline.vault, pipeline.vault.paths(run_id)
    first = vault.read_handoff(run_id, "04-crosscheck")
    issue = first.section("Issues")
    assert issue.startswith("- [critical] SRC-1: `document.md`: the reference '2. Gemini, survey of O(1) allocators")
    assert "is an internal pipeline note, not a citable source [internal_note]" in issue
    assert "> - SRC-1 [internal_note] `document.md` 2. Gemini" in first.section("Source Audit")
    assert "LINT" not in issue  # the document's Markdown is clean: one defect, one issue
    fix_prompt = script.prompts["fix_report"][0]
    assert "## Verified sources from the ingestion report" in fix_prompt and "[S1] TLSF" in fix_prompt
    audits = [e for e in pipeline.ledger_for(index).entries if e.purpose == "source_audit"]
    assert [e.stage for e in audits] == ["crosscheck"] * 2
    if fix_pass:
        assert first.section("Verdict") == "PASS" and first.section("Applied Fixes") == "- SRC-1: fixed"
        assert script.count("source_audit") == 2 and "03-execution-r2" not in index.handoffs
        assert "Gemini, survey" not in script.prompts["source_audit"][1]  # the re-audit saw the revised document
        assert first.section("Source Audit").split("\n\n")[2].startswith(
            "Source audit after the fix pass: Gemini checked 1 reference(s) in `document.md` with web search: 1 verified"
        )
    else:
        assert first.section("Verdict") == "LOOP" and "SRC-1" in first.section("Unresolved Critical")
        assert "SRC-1" in script.prompts["execution"][1] and "168 cycles" in script.prompts["execution"][1]
        second = vault.read_handoff(run_id, "04-crosscheck-r2")
        assert second.section("Verdict") == "PASS" and second.section("Issues") == "None."
        assert second.section("Source Audit").startswith("Source audit: Gemini checked 1 reference(s) in `document.md`")
        assert script.count("rebuttal") == 1  # round 2 raised nothing, so nothing was debated
    shipped = (paths.deliverables / "document.md").read_text(encoding="utf-8")
    assert "Gemini, survey" not in shipped and "Masmano" in shipped
    final = vault.read_handoff(run_id, "05-final")
    assert "- AC-1 [met]: Every reference passes the source audit." in final.section("Acceptance")
    assert "- source-audit [met]: Source audit:" in final.section("Acceptance")  # maf's own verdict agrees
    assert "- lint [met]:" in final.section("Acceptance")
    assert "- AC-2 [partial]:" in final.section("Acceptance") and "(soft criterion)" in final.section("Acceptance")
    assert "[!warning]" not in final.section("Summary")  # a soft criterion does not block
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    assert len(ledger.entries) == _total_calls(fake_providers)
    note_costs = sum(vault.read_handoff(run_id, name).meta.cost_usd for name in index.handoffs)
    assert note_costs == pytest.approx(ledger.spent_usd)  # the audits are billed to their cross-check notes


@dataclass
class ReattributingScript(DocScript):
    """The thesis replay: the audit finds a value credited to a paper that does not contain it (a major
    ``unsupported_claim``); the fixer, without web access, re-attributes it from memory and reports SRC-1 fixed. The
    re-audit of the revised document finds the new attribution unsupported too."""

    def reply(self, role: str, request: CompletionRequest) -> str | dict[str, object]:
        kind = _kind_of(request)
        if kind == "execution":
            self.requests.append((role, kind))
            self.prompts.setdefault(kind, []).append(request.messages[-1].content)
            return _prose_execution(CLEAN_SURVEY.replace("so none\nis given", "the bound is 168 cycles [1]"))
        if kind == "source_audit":
            self.requests.append((role, kind))
            self.prompts.setdefault(kind, []).append(request.messages[-1].content)
            revised = "Rodriguez" in request.messages[-1].content
            reference = "2. J. Rodriguez, Bounded allocators, RTSS 2009." if revised else MASMANO
            return {
                "references": [{
                    "document": "document.md", "reference": reference, "verdict": "unsupported_claim",
                    "claims_checked": ["the bound is 168 cycles"], "finding": "the paper gives no cycle bound",
                    "correction": "drop the 168-cycle bound",
                }],
                "unaudited": 0, "summary": "One attribution is unsupported.",
            }
        if kind == "rebuttal":
            self.requests.append((role, kind))
            return "## Summary\n\nAnswered.\n\n## Responses\n\n- SRC-1 [accept]: re-attribute the bound\n"
        if kind == "fix_report":
            self.requests.append((role, kind))
            self.prompts.setdefault(kind, []).append(request.messages[-1].content)
            assert self.workspace is not None
            current = (self.workspace / "document.md").read_text(encoding="utf-8")
            revised = current.replace("168 cycles [1]", "168 cycles [2]") + (
                "2. J. Rodriguez, Bounded allocators, RTSS 2009.\n"
            )
            return {"fixed": ["SRC-1"], "not_fixed": [], "summary": "Re-attributed [2].", "document": revised}
        return super().reply(role, request)


def test_a_re_attribution_from_memory_never_ends_completed(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str]
) -> None:
    """Before, the one audit ran before the fix pass: the fixer's new attribution was never checked, the verdict was
    PASS, final read "SRC-1: fixed" and the run ended ``completed``. Now the revised text is audited again (a major
    finding: no loop), and final's ``source-audit`` gate, reading that audit of the shipped text, is unmet, so the
    run ends ``completed_with_issues`` even though final's model called AC-1 met."""
    script = ReattributingScript(sample_bodies, fix_pass=True)
    script.install(fake_providers)
    pipeline = _pipeline(settings, fake_providers)
    run_id = pipeline.create("Write a short survey of O(1) allocators.").run_id
    script.workspace = pipeline.vault.paths(run_id).workspace

    index = pipeline.run(run_id)

    assert index.status == RunStatus.COMPLETED_WITH_ISSUES, index.error
    assert (index.round, index.unresolved_critical, index.criteria_unmet) == (1, 0, 1)
    assert index.unmet_criteria[0].startswith("source-audit [unmet]: Source audit:")
    crosscheck = pipeline.vault.read_handoff(run_id, "04-crosscheck")
    assert crosscheck.section("Verdict") == "PASS"
    assert [line.split(":")[0] for line in crosscheck.section("Issues").splitlines()] == [
        "- [major] SRC-1", "- [major] SRC-2",
    ]
    assert "Rodriguez" in script.prompts["source_audit"][1]
    final = pipeline.vault.read_handoff(run_id, "05-final")
    assert "- AC-1 [met]:" in final.section("Acceptance")  # the model's verdict, from "SRC-1: fixed"
    assert "`document.md` has 1 reference(s) the audit did not verify" in final.section("Acceptance")
    assert final.section("Summary").startswith("> [!warning] Run status: completed_with_issues")


LEAKY_README = "# Allocator\n\nThe design follows the survey in [[01-ingestion]].\n\nBuild with `make all`.\n"
CLEAN_README = "# Allocator\n\nThe design is a two-level segregated fit.\n\nBuild with `make all`.\n"


@dataclass
class LintScript(Script):
    """The allocator run whose README links a pipeline note. No critic raises anything: the only issue is maf's
    LINT-1. ``honest`` fixers remove the link; otherwise the fixer claims LINT-1 fixed and leaves it, maf's re-lint
    catches that, the run loops, and execution round 2 writes the clean README."""

    honest: bool = True

    def build(self) -> None:
        super().build()
        assert self.workspace is not None
        (self.workspace / "README.md").write_text(LEAKY_README if self.executions == 1 else CLEAN_README)

    def answer(self, role: str, kind: str) -> str | dict[str, object]:
        if kind == "critique":
            return CLEAN_CRITIQUE
        if kind == "rebuttal":
            return "## Summary\n\nAnswered.\n\n## Responses\n\n- LINT-1 [accept]: cite the paper instead\n"
        if kind == "fix_report":
            assert self.workspace is not None
            if self.honest:
                (self.workspace / "README.md").write_text(CLEAN_README)
            return {"fixed": ["LINT-1"], "not_fixed": [], "summary": "Replaced the link."}
        return super().answer(role, kind)


@pytest.mark.parametrize("honest", [True, False], ids=["fixed-by-the-fix-pass", "false-claim-then-round-2"])
def test_pipeline_wikilink_in_a_deliverable_is_one_lint_issue_until_fixed(
    settings: Settings, fake_providers: FakeProviders, sample_bodies: dict[str, str], honest: bool
) -> None:
    """A deliverable links ``[[01-ingestion]]``: execution lists it under ``## Lint``, the cross-check raises it once
    (LINT-1, critical; critics see it only under the automated checks, never the execution note's stale list), and it
    counts as fixed only when maf's re-lint no longer finds it."""
    script = LintScript(sample_bodies, honest=honest)
    pipeline, run_id = _start(settings, fake_providers, script)

    index = pipeline.run(run_id)

    assert index.status == RunStatus.COMPLETED, index.error
    assert (index.round, index.unresolved_critical) == (1 if honest else 2, 0)
    vault, paths = pipeline.vault, pipeline.vault.paths(run_id)
    execution = vault.read_handoff(run_id, "03-execution")
    assert "- [critical] pipeline-wikilink README.md:3: link to the internal pipeline note `[[01-ingestion]]`" in (
        execution.section("Lint")
    )
    first = vault.read_handoff(run_id, "04-crosscheck")
    assert first.section("Issues").splitlines() == [
        "- [critical] LINT-1: `README.md` line 3: pipeline-wikilink: link to the internal pipeline note "
        "`[[01-ingestion]]`; a deliverable must stand alone, so cite the original source instead",
    ]
    critique_prompts = [r.messages[-1].content for f in fake_providers.all() for r in f.calls if _kind_of(r) == "critique"]
    assert len(critique_prompts) == (3 if honest else 6)
    first_round = critique_prompts[0]
    assert "- [critical] LINT-1: `README.md` line 3: pipeline-wikilink" in first_round.split("## Automated checks\n\n")[1]
    assert "## Lint\n" not in first_round and "maf lint found these critical and major problems" not in first_round
    if honest:
        assert first.section("Applied Fixes") == "- LINT-1: fixed" and first.section("Verdict") == "PASS"
    else:
        assert first.section("Applied Fixes") == (
            "- LINT-1: not fixed - maf lint still reports pipeline-wikilink in `README.md` after the fix pass"
        )
        assert first.section("Verdict") == "LOOP" and "LINT-1" in first.section("Unresolved Critical")
        r2 = [r for r in fake_providers.claude_code.calls if _kind_of(r) == "execution"][1].messages[-1].content
        assert "LINT-1" in r2 and "pipeline-wikilink" in r2
        assert "Lint" not in vault.read_handoff(run_id, "03-execution-r2").sections
        assert vault.read_handoff(run_id, "04-crosscheck-r2").section("Issues") == "None."
    assert (paths.deliverables / "README.md").read_text() == CLEAN_README
    assert _lint.lint_deliverables(paths.deliverables) == []
    assert "Lint" not in vault.read_handoff(run_id, "05-final").section("Limitations")
    ledger = _assert_ledger_mirrored(pipeline, run_id)
    assert len(ledger.entries) == _total_calls(fake_providers)
