"""Pipeline state machine tests: scripted stage backends over a real temp vault and ledger, fake providers."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from maf.config import Settings
from maf.handoff import HandoffInvalid, HandoffKind, HandoffMeta, build_handoff
from maf.ledger import metered_call
from maf.pipeline import ALLOWED_INDEX_UPDATES, REVIEW_NOTE_REL, Pipeline, Step, next_step
from maf.providers import CompletionRequest, ProviderError
from maf.stages.base import NoteOut, StageContext, StageOutput
from maf.stages.final import CLEANROOM_ROOT, REPRO_EXIT, REPRO_LOG, cleanroom_dir
from maf.types import STAGE_ORDER, AgentName, RunStatus, StageName
from maf.vault import ExportError, ExportTooLarge, RunIndex, note_name

StageFn = Callable[[StageContext], StageOutput]


def _index(**kw: object) -> RunIndex:
    now = datetime(2026, 9, 28, 12, 0)
    base: dict[str, object] = dict(
        run_id="2026-09-28-x", budget_usd=25.0, created=now, updated=now, workspace="/tmp/ws", brief="x"
    )
    base.update(kw)
    return RunIndex.model_validate(base)


def _out(loop_back: bool = False) -> StageOutput:
    return StageOutput(notes=[], loop_back=loop_back)


# --------------------------------------------------------------------------- next_step (pure)


@pytest.mark.parametrize(
    ("finished", "review", "round_", "loop_back", "max_loops", "expected"),
    [
        ("ingestion", False, 1, False, 2, Step("strategy", 1, RunStatus.RUNNING)),
        ("strategy", False, 1, False, 2, Step("execution", 1, RunStatus.RUNNING)),
        ("strategy", True, 1, False, 2, Step("execution", 1, RunStatus.AWAITING_REVIEW)),
        ("execution", False, 2, False, 2, Step("crosscheck", 2, RunStatus.RUNNING)),
        ("crosscheck", False, 1, True, 2, Step("execution", 2, RunStatus.RUNNING)),
        ("crosscheck", False, 2, True, 2, Step("execution", 3, RunStatus.RUNNING)),
        ("crosscheck", False, 3, True, 2, Step("final", 3, RunStatus.RUNNING)),
        ("crosscheck", False, 1, False, 2, Step("final", 1, RunStatus.RUNNING)),
        ("crosscheck", False, 1, True, 0, Step("final", 1, RunStatus.RUNNING)),
        ("final", False, 2, False, 2, Step(None, 2, RunStatus.COMPLETED)),
        ("final", True, 1, True, 2, Step(None, 1, RunStatus.COMPLETED)),
    ],
)
def test_next_step(
    finished: StageName, review: bool, round_: int, loop_back: bool, max_loops: int, expected: Step
) -> None:
    assert next_step(_index(review=review, round=round_), finished, _out(loop_back), max_loops) == expected


def test_next_step_ignores_loop_back_outside_crosscheck() -> None:
    assert next_step(_index(), "execution", _out(True), 2) == Step("crosscheck", 1, RunStatus.RUNNING)


def test_next_step_final_with_unresolved_critical_is_completed_with_issues() -> None:
    assert next_step(_index(round=3, unresolved_critical=2), "final", _out(), 2) == Step(
        None, 3, RunStatus.COMPLETED_WITH_ISSUES
    )
    assert next_step(_index(round=3, unresolved_critical=0), "final", _out(), 2) == Step(None, 3, RunStatus.COMPLETED)


def test_next_step_final_with_unmet_criteria_is_completed_with_issues() -> None:
    assert next_step(_index(criteria_unmet=1), "final", _out(), 2) == Step(None, 1, RunStatus.COMPLETED_WITH_ISSUES)


def test_completed_with_issues_is_terminal_and_finished() -> None:
    status = RunStatus.COMPLETED_WITH_ISSUES
    assert status.value == "completed_with_issues"
    assert status.terminal and status.finished
    assert RunStatus.COMPLETED.finished and not RunStatus.FAILED.finished and not RunStatus.BUDGET_EXCEEDED.finished


def test_allowed_index_updates() -> None:
    assert ALLOWED_INDEX_UPDATES == {
        "mode", "unresolved_critical", "criteria_unmet", "unmet_criteria", "exported_at", "export_note"
    }


# --------------------------------------------------------------------------- scripted backends


def _note(ctx: StageContext, kind: HandoffKind, body: str, *, frm: str = "claude", agent: AgentName | None = None) -> NoteOut:
    meta = HandoffMeta.model_validate(
        {
            "run_id": ctx.run_id,
            "stage": kind,
            "from": frm,
            "to": "next",
            "created": ctx.now,
            "model": "fake-model",
            "cost_usd": 0.0,
            "round": ctx.index.round,
        }
    )
    return NoteOut(note_name(kind, ctx.index.round, agent), build_handoff(body, meta))


@dataclass
class ScriptedBackend:
    """A stage backend whose behavior per call is scripted; records every context it receives."""

    name: StageName
    default: StageFn
    script: list[StageFn] = field(default_factory=list)
    contexts: list[StageContext] = field(default_factory=list)

    def run_stage(self, ctx: StageContext) -> StageOutput:
        self.contexts.append(ctx)
        fn = self.script.pop(0) if self.script else self.default
        return fn(ctx)

    @property
    def calls(self) -> int:
        return len(self.contexts)


def _defaults(bodies: dict[str, str]) -> dict[StageName, StageFn]:
    def ingestion(ctx: StageContext) -> StageOutput:
        return StageOutput(
            notes=[
                _note(ctx, HandoffKind.ROUTING, bodies["routing"], frm="chatgpt"),
                _note(ctx, HandoffKind.INGESTION, bodies["ingestion"], frm="gemini"),
            ],
            index_updates={"mode": "code"},
        )

    def strategy(ctx: StageContext) -> StageOutput:
        return StageOutput(notes=[_note(ctx, HandoffKind.STRATEGY, bodies["strategy"], frm="chatgpt")])

    def execution(ctx: StageContext) -> StageOutput:
        return StageOutput(notes=[_note(ctx, HandoffKind.EXECUTION, bodies["execution"])])

    def crosscheck(ctx: StageContext) -> StageOutput:
        return StageOutput(
            notes=[_note(ctx, HandoffKind.CROSSCHECK, bodies["crosscheck"], frm="maf")],
            index_updates={"unresolved_critical": 0},
        )

    def final(ctx: StageContext) -> StageOutput:
        return StageOutput(notes=[_note(ctx, HandoffKind.FINAL, bodies["final"])])

    return {"ingestion": ingestion, "strategy": strategy, "execution": execution, "crosscheck": crosscheck, "final": final}


def _defaults_for(h: "Harness") -> dict[StageName, StageFn]:
    return _defaults(h.bodies)


def _loop(ctx: StageContext) -> StageOutput:
    body = (
        "## Summary\n\nOne critical issue left.\n\n## Issues\n\n- [critical] GPT-1: broken\n\n## Rulings\n\nNone.\n\n"
        "## Applied Fixes\n\nNone.\n\n## Unresolved Critical\n\n- GPT-1\n\n## Verdict\n\nLOOP\n"
    )
    return StageOutput(
        notes=[_note(ctx, HandoffKind.CROSSCHECK, body, frm="maf")],
        index_updates={"unresolved_critical": 1},
        loop_back=True,
    )


class Harness:
    def __init__(self, settings: Settings, fake_providers, sample_bodies: dict[str, str]) -> None:  # type: ignore[no-untyped-def]
        self.settings = settings
        self.fakes = fake_providers
        self.bodies = sample_bodies
        defaults = _defaults(sample_bodies)
        self.backends = {stage: ScriptedBackend(stage, defaults[stage]) for stage in STAGE_ORDER}
        self.ticks = 0
        self.pipeline = self.make_pipeline()

    def clock(self) -> datetime:
        self.ticks += 1
        return datetime(2026, 9, 28, 12, 0) + timedelta(seconds=self.ticks)

    def make_pipeline(self, **kw: object) -> Pipeline:
        options: dict[str, object] = dict(
            providers_factory=self.fakes.factory(), backends=self.backends, clock=self.clock
        )
        options.update(kw)
        return Pipeline(self.settings, **options)  # type: ignore[arg-type]

    def __getitem__(self, stage: StageName) -> ScriptedBackend:
        return self.backends[stage]


@pytest.fixture
def h(settings: Settings, fake_providers, sample_bodies: dict[str, str]) -> Harness:  # type: ignore[no-untyped-def]
    return Harness(settings, fake_providers, sample_bodies)


def _metered(role: str, purpose: str = "call") -> StageFn:
    """A stage fn that makes one metered call and then behaves like the default for that stage."""

    def fn(ctx: StageContext) -> StageOutput:
        request = CompletionRequest.simple(ctx.model(role), "hi", max_output_tokens=100)  # type: ignore[arg-type]
        metered_call(ctx.ledger, ctx.providers.for_role(role), request, stage=ctx.stage, purpose=purpose)  # type: ignore[arg-type]
        return StageOutput(notes=[])

    return fn


# --------------------------------------------------------------------------- create


def test_create_writes_pending_run_without_model_calls(h: Harness, tmp_path: Path) -> None:
    src = tmp_path / "spec.txt"
    src.write_text("requirements", encoding="utf-8")
    index = h.pipeline.create("Portable allocator", [src], budget_usd=7.5, tier="max", review=True)

    assert index.status == RunStatus.PENDING
    assert index.stage == "ingestion"
    assert index.round == 1
    assert (index.budget_usd, index.tier, index.review) == (7.5, "max", True)
    assert index.run_id.startswith("2026-09-28-portable-allocator")
    paths = h.pipeline.vault.paths(index.run_id)
    assert paths.run_md.is_file()
    assert (paths.workspace / "inputs" / "spec.txt").read_text(encoding="utf-8") == "requirements"
    assert index.input_files and index.input_files[0].endswith("spec.txt")
    assert h.pipeline.status(index.run_id) == index
    assert all(not fake.calls for fake in h.fakes.all())
    assert all(b.calls == 0 for b in h.backends.values())


def test_create_uses_settings_defaults(h: Harness) -> None:
    index = h.pipeline.create("brief")
    assert (index.budget_usd, index.tier, index.review) == (h.settings.budget_usd, "default", False)


def test_create_missing_file_creates_nothing(h: Harness, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        h.pipeline.create("brief", [tmp_path / "nope.pdf"])
    assert not h.pipeline.vault.runs_dir.exists() or not any(h.pipeline.vault.runs_dir.iterdir())
    assert h.pipeline.list_runs() == []


@pytest.mark.parametrize(
    ("brief", "budget"), [("   ", None), ("ok", 0.0), ("ok", -1.0), ("ok", float("nan")), ("ok", float("inf"))]
)
def test_create_rejects_bad_arguments(h: Harness, brief: str, budget: float | None) -> None:
    with pytest.raises(ValueError):
        h.pipeline.create(brief, budget_usd=budget)


def test_create_same_brief_twice_gets_unique_ids(h: Harness) -> None:
    a = h.pipeline.create("same brief")
    b = h.pipeline.create("same brief")
    assert a.run_id != b.run_id
    assert {r.run_id for r in h.pipeline.list_runs()} == {a.run_id, b.run_id}


def test_pipeline_requires_every_backend(h: Harness) -> None:
    partial = {k: v for k, v in h.backends.items() if k != "final"}
    with pytest.raises(ValueError, match="final"):
        h.make_pipeline(backends=partial)


# --------------------------------------------------------------------------- run


def test_happy_path_completes(h: Harness) -> None:
    run_id = h.pipeline.create("Portable allocator").run_id
    messages: list[str] = []
    index = h.pipeline.run(run_id, progress=lambda _i, m: messages.append(m))

    assert index.status == RunStatus.COMPLETED
    assert index.stage == "final"
    assert index.error is None
    assert index.mode == "code"
    assert index.completed_stages == list(STAGE_ORDER)
    assert index.handoffs == ["01a-routing", "01-ingestion", "02-strategy", "03-execution", "04-crosscheck", "05-final"]
    assert h.pipeline.status(run_id) == index
    for name in index.handoffs:
        assert h.pipeline.vault.has_note(run_id, name)
    assert [b.calls for b in h.backends.values()] == [1, 1, 1, 1, 1]
    assert any("run completed" in m for m in messages)
    # Stages get a snapshot, not the live index.
    assert h["strategy"].contexts[0].index.mode == "code"
    assert h["strategy"].contexts[0].stage == "strategy"


def test_run_on_completed_run_is_a_noop(h: Harness) -> None:
    run_id = h.pipeline.create("x").run_id
    first = h.pipeline.run(run_id)
    assert h.pipeline.run(run_id) == first
    assert h.pipeline.resume(run_id, note="ignored") == first
    assert h["final"].calls == 1


def test_run_missing_raises(h: Harness) -> None:
    with pytest.raises(FileNotFoundError):
        h.pipeline.run("2026-01-01-nope")


def test_crosscheck_loops_are_capped(h: Harness) -> None:
    h["crosscheck"].default = _loop
    run_id = h.pipeline.create("x").run_id
    messages: list[str] = []
    index = h.pipeline.run(run_id, progress=lambda _i, m: messages.append(m))

    assert index.status == RunStatus.COMPLETED_WITH_ISSUES
    assert index.error is None
    assert any("run completed with issues: 1 unresolved critical issue(s)" in m for m in messages)
    assert h["execution"].calls == 3  # 1 + max_crosscheck_loops
    assert [c.index.round for c in h["execution"].contexts] == [1, 2, 3]
    assert [c.index.round for c in h["crosscheck"].contexts] == [1, 2, 3]
    assert h["final"].contexts[0].index.round == 3
    assert index.round == 3
    assert index.unresolved_critical == 1
    assert "03-execution-r2" in index.handoffs and "04-crosscheck-r3" in index.handoffs
    assert index.handoffs[-1] == "05-final"
    # run.md says so in the frontmatter and opens ## Status with a callout linking the last cross-check.
    run_md = h.pipeline.vault.paths(run_id).run_md.read_text(encoding="utf-8")
    assert "\nstatus: completed_with_issues\n" in run_md
    status_block = run_md.split("## Status\n\n", 1)[1]
    assert status_block.startswith("> [!warning] Completed with 1 unresolved critical issue(s)\n")
    assert "[[04-crosscheck-r3]]" in status_block.split("\n\n", 1)[0]
    # Terminal: run() and a plain resume() leave it alone.
    assert h.pipeline.run(run_id) == index
    assert h.pipeline.resume(run_id, note="ignored", budget_usd=99.0) == index
    assert not h.pipeline.review_note_path(run_id).exists()
    assert (h["execution"].calls, h["final"].calls) == (3, 1)


def test_extra_round_runs_one_more_pass_then_final(h: Harness) -> None:
    h["crosscheck"].default = _loop
    run_id = h.pipeline.create("x").run_id
    assert h.pipeline.run(run_id).status == RunStatus.COMPLETED_WITH_ISSUES

    h["crosscheck"].default = _defaults_for(h)["crosscheck"]  # the extra pass resolves everything
    index = h.pipeline.resume(run_id, extra_round=True, note="focus on GPT-1", budget_usd=30.0)

    assert index.status == RunStatus.COMPLETED, index.error
    assert (index.round, index.unresolved_critical, index.budget_usd) == (4, 0, 30.0)
    assert [c.index.round for c in h["execution"].contexts] == [1, 2, 3, 4]
    assert h["execution"].contexts[-1].review_note == "focus on GPT-1"
    assert h["final"].calls == 2
    # 05-final moved to the end, after the extra pass's notes.
    assert index.handoffs[-3:] == ["03-execution-r4", "04-crosscheck-r4", "05-final"]
    assert index.handoffs.count("05-final") == 1
    assert index.completed_stages[-3:] == ["execution", "crosscheck", "final"]


def test_extra_round_that_leaves_issues_open_ends_with_issues_again(h: Harness) -> None:
    h["crosscheck"].default = _loop
    run_id = h.pipeline.create("x").run_id
    h.pipeline.run(run_id)
    index = h.pipeline.resume(run_id, extra_round=True)
    assert index.status == RunStatus.COMPLETED_WITH_ISSUES
    assert index.round == 4
    assert h["execution"].calls == 4  # exactly one extra pass: the loop cap still applies
    assert h["final"].calls == 2


@pytest.mark.parametrize("stopped", ["completed", "failed", "awaiting_review"])
def test_extra_round_only_applies_to_completed_with_issues(h: Harness, stopped: str) -> None:
    if stopped == "failed":
        h["strategy"].script = [lambda ctx: (_ for _ in ()).throw(KeyError("boom"))]
    run_id = h.pipeline.create("x", review=stopped == "awaiting_review").run_id
    index = h.pipeline.run(run_id)
    assert index.status.value == stopped
    with pytest.raises(ValueError, match="completed_with_issues"):
        h.pipeline.resume(run_id, extra_round=True)
    assert h.pipeline.status(run_id) == index  # nothing was written


def test_unmet_criteria_from_final_end_completed_with_issues(h: Harness) -> None:
    default_final = _defaults_for(h)["final"]

    def unmet(ctx: StageContext) -> StageOutput:
        output = default_final(ctx)
        output.index_updates = {
            "criteria_unmet": 1,
            "unmet_criteria": ["clean-room [unmet]: Clean-room reproduction"],
            "exported_at": ctx.now,
            "export_note": "final: 3 file(s), 1.0 kB",
        }
        return output

    h["final"].script = [unmet]
    run_id = h.pipeline.create("x").run_id
    messages: list[str] = []

    index = h.pipeline.run(run_id, progress=lambda _i, m: messages.append(m))

    assert index.status == RunStatus.COMPLETED_WITH_ISSUES
    assert (index.unresolved_critical, index.criteria_unmet) == (0, 1)
    assert index.unmet_criteria == ["clean-room [unmet]: Clean-room reproduction"]
    assert index.export_note == "final: 3 file(s), 1.0 kB" and index.exported_at is not None
    assert messages[-1].startswith("final done; run completed with issues: 1 acceptance criterion not met (clean-room)")
    assert h.pipeline.status(run_id) == index

    # One more pass whose final meets everything clears the fields and completes the run.
    again = h.pipeline.resume(run_id, extra_round=True)
    assert again.status == RunStatus.COMPLETED, again.error
    assert (again.round, again.criteria_unmet, again.unmet_criteria) == (2, 0, [])


def test_single_loop_then_pass(h: Harness) -> None:
    h["crosscheck"].script = [_loop]
    index = h.pipeline.run(h.pipeline.create("x").run_id)
    assert index.status == RunStatus.COMPLETED
    assert h["execution"].calls == 2
    assert index.round == 2
    assert index.unresolved_critical == 0
    assert index.handoffs[-4:] == ["04-crosscheck", "03-execution-r2", "04-crosscheck-r2", "05-final"]


def test_zero_loops_goes_straight_to_final(h: Harness) -> None:
    h.settings.max_crosscheck_loops = 0
    h["crosscheck"].default = _loop
    index = h.pipeline.run(h.pipeline.create("x").run_id)
    assert index.status == RunStatus.COMPLETED_WITH_ISSUES
    assert h["execution"].calls == 1


# --------------------------------------------------------------------------- review gate


def test_review_gate_pauses_then_resume_uses_edited_strategy(h: Harness) -> None:
    run_id = h.pipeline.create("x", review=True).run_id
    index = h.pipeline.run(run_id)
    assert index.status == RunStatus.AWAITING_REVIEW
    assert index.stage == "execution"
    assert h["execution"].calls == 0
    assert h.pipeline.run(run_id) == index  # run() does not pass the gate

    note_path = h.pipeline.vault.paths(run_id).note("02-strategy")
    text = note_path.read_text(encoding="utf-8")
    note_path.write_text(text.replace("## Risks", "## Risks\n\n- USER EDIT: use a bitmap.\n", 1), encoding="utf-8")

    index = h.pipeline.resume(run_id, note="Prefer TLSF over buddy.")
    assert index.status == RunStatus.COMPLETED
    ctx = h["execution"].contexts[0]
    assert ctx.review_note == "Prefer TLSF over buddy."
    assert "USER EDIT" in ctx.read("02-strategy").section("Risks")
    assert h["crosscheck"].contexts[0].review_note == "Prefer TLSF over buddy."
    stored = h.pipeline.vault.paths(run_id).workspace / REVIEW_NOTE_REL
    assert stored.read_text(encoding="utf-8").strip() == "Prefer TLSF over buddy."
    assert h["strategy"].calls == 1


def test_review_gate_invalid_edit_fails_then_fix_and_resume(h: Harness) -> None:
    run_id = h.pipeline.create("x", review=True).run_id
    h.pipeline.run(run_id)
    note_path = h.pipeline.vault.paths(run_id).note("02-strategy")
    good = note_path.read_text(encoding="utf-8")
    note_path.write_text(good.split("## Risks")[0], encoding="utf-8")

    index = h.pipeline.resume(run_id)
    assert index.status == RunStatus.FAILED
    assert "Risks" in (index.error or "")
    assert index.stage == "execution"
    assert h["execution"].calls == 0

    note_path.write_text(good, encoding="utf-8")
    index = h.pipeline.resume(run_id)
    assert index.status == RunStatus.COMPLETED
    assert index.error is None


def test_review_gate_missing_strategy_note_fails(h: Harness) -> None:
    run_id = h.pipeline.create("x", review=True).run_id
    h.pipeline.run(run_id)
    h.pipeline.vault.paths(run_id).note("02-strategy").unlink()
    index = h.pipeline.resume(run_id)
    assert index.status == RunStatus.FAILED
    assert "missing" in (index.error or "")


def test_review_note_persists_across_resumes(h: Harness) -> None:
    def flaky(ctx: StageContext) -> StageOutput:
        raise RuntimeError("flaky")

    h["execution"].script = [flaky, flaky]
    run_id = h.pipeline.create("x").run_id
    assert h.pipeline.run(run_id).status == RunStatus.FAILED
    assert h["execution"].contexts[-1].review_note is None
    assert h.pipeline.resume(run_id, note="be careful").status == RunStatus.FAILED
    assert h["execution"].contexts[-1].review_note == "be careful"
    # A later resume (new Pipeline) without --note still passes the stored note.
    assert h.make_pipeline().resume(run_id).status == RunStatus.COMPLETED
    assert h["execution"].contexts[-1].review_note == "be careful"


# --------------------------------------------------------------------------- errors and budget


def test_budget_exceeded_then_resume_with_higher_budget(h: Harness) -> None:
    h.fakes.chatgpt.default = "ok"
    h.fakes.chatgpt.worst_case_usd = 3.0
    h.fakes.chatgpt.cost_per_call = 1.0
    default = h["strategy"].default
    h["strategy"].default = lambda ctx: (_metered("chatgpt")(ctx), default(ctx))[1]
    run_id = h.pipeline.create("x", budget_usd=2.0).run_id

    index = h.pipeline.run(run_id)
    assert index.status == RunStatus.BUDGET_EXCEEDED
    assert index.stage == "strategy"
    assert "budget cap" in (index.error or "")
    assert index.spent_usd == 0.0
    assert h.fakes.chatgpt.calls == []

    index = h.pipeline.resume(run_id, budget_usd=10.0)
    assert index.status == RunStatus.COMPLETED
    assert index.budget_usd == 10.0
    assert index.spent_usd == pytest.approx(1.0)
    assert index.spend_by_agent["chatgpt"] == pytest.approx(1.0)
    assert index.spend_by_provider["openai"] == pytest.approx(1.0)
    assert h.pipeline.ledger_for(index).spent_usd == pytest.approx(1.0)


def test_ledger_totals_mirrored_after_each_stage(h: Harness) -> None:
    for stage, role in (("ingestion", "gemini"), ("execution", "claude_code"), ("final", "claude")):
        default = h[stage].default
        h[stage].default = lambda ctx, d=default, r=role: (_metered(r)(ctx), d(ctx))[1]  # type: ignore[misc]
    for fake in h.fakes.all():
        fake.default = "ok"
    snapshots: list[float] = []
    index = h.pipeline.run(
        h.pipeline.create("x").run_id, progress=lambda i, m: snapshots.append(i.spent_usd) if "done" in m else None
    )
    assert index.status == RunStatus.COMPLETED
    assert index.spend_by_agent == pytest.approx({"chatgpt": 0.0, "gemini": 0.01, "claude": 0.51})
    assert index.spend_by_provider["claude_code"] == pytest.approx(0.5)
    assert index.spent_usd == pytest.approx(0.52)
    assert snapshots == pytest.approx([0.01, 0.01, 0.51, 0.51, 0.52])
    run_md = h.pipeline.vault.paths(index.run_id).run_md.read_text(encoding="utf-8")
    assert "0.52" in run_md


def test_handoff_invalid_fails_run(h: Harness) -> None:
    def bad(ctx: StageContext) -> StageOutput:
        raise HandoffInvalid(HandoffKind.EXECUTION, ["missing section 'Verification'"])

    h["execution"].script = [bad]
    run_id = h.pipeline.create("x").run_id
    index = h.pipeline.run(run_id)
    assert index.status == RunStatus.FAILED
    assert index.stage == "execution"
    assert "Verification" in (index.error or "")
    # Notes from completed stages survive; resume repeats only the failed stage.
    assert index.handoffs == ["01a-routing", "01-ingestion", "02-strategy"]
    assert h.pipeline.resume(run_id).status == RunStatus.COMPLETED
    assert (h["ingestion"].calls, h["strategy"].calls, h["execution"].calls) == (1, 1, 2)


def test_provider_error_records_partial_spend(h: Harness) -> None:
    h.fakes.claude_code.script(ProviderError("hit budget", provider="claude_code", cost_usd=0.75))
    h["execution"].script = [_metered("claude_code")]
    index = h.pipeline.run(h.pipeline.create("x").run_id)
    assert index.status == RunStatus.FAILED
    assert index.error == "execution: ProviderError: hit budget"
    assert index.spent_usd == pytest.approx(0.75)
    assert index.spend_by_agent["claude"] == pytest.approx(0.75)


class SandboxDown(ProviderError):
    """Stands in for ``maf.providers.base.SandboxUnavailable``: a non-retryable infrastructure failure."""


@pytest.mark.parametrize("retryable", [False, True])
def test_provider_error_in_execution_fails_fast_without_crosscheck(h: Harness, retryable: bool) -> None:
    """Claude Code is never retried, so even a retryable-flagged error ends the run at once."""
    h.fakes.claude_code.script(
        SandboxDown("Sandbox is required but failed to initialize", provider="claude_code", cost_usd=0.2, retryable=retryable)
    )
    h["execution"].script = [_metered("claude_code")]
    index = h.pipeline.run(h.pipeline.create("x").run_id)
    assert index.status == RunStatus.FAILED
    assert (index.stage, index.round) == ("execution", 1)
    assert index.error == "execution: SandboxDown: Sandbox is required but failed to initialize"
    assert index.spent_usd == pytest.approx(0.2)
    assert len(h.fakes.claude_code.calls) == 1
    assert (h["crosscheck"].calls, h["final"].calls) == (0, 0)
    assert index.handoffs == ["01a-routing", "01-ingestion", "02-strategy"]


def test_real_sandbox_unavailable_fails_fast(h: Harness) -> None:
    """The providers' own ``SandboxUnavailable`` (when present) takes the same generic non-retryable path."""
    import maf.providers.base as provider_base

    sandbox_error = getattr(provider_base, "SandboxUnavailable", None)
    if sandbox_error is None:
        pytest.skip("maf.providers.base.SandboxUnavailable not available")
    h.fakes.claude_code.script(sandbox_error("bridge sockets failed", provider="claude_code", cost_usd=0.3))
    h["execution"].script = [_metered("claude_code")]
    index = h.pipeline.run(h.pipeline.create("x").run_id)
    assert index.status == RunStatus.FAILED
    assert index.error == "execution: SandboxUnavailable: bridge sockets failed"
    assert index.spent_usd == pytest.approx(0.3)
    assert h["crosscheck"].calls == 0


def test_provider_error_in_crosscheck_never_loops_back(h: Harness) -> None:
    def broken_fix_pass(ctx: StageContext) -> StageOutput:
        raise SandboxDown("bridge sockets", provider="claude_code", cost_usd=0.1)

    h["crosscheck"].script = [broken_fix_pass]
    h["crosscheck"].default = _loop
    index = h.pipeline.run(h.pipeline.create("x").run_id)
    assert index.status == RunStatus.FAILED
    assert (index.stage, index.round) == ("crosscheck", 1)
    assert index.error == "crosscheck: SandboxDown: bridge sockets"
    assert (h["execution"].calls, h["crosscheck"].calls, h["final"].calls) == (1, 1, 0)
    assert index.unresolved_critical == 0


def test_unexpected_exception_fails_with_type(h: Harness) -> None:
    h["crosscheck"].script = [lambda ctx: (_ for _ in ()).throw(KeyError("boom"))]
    index = h.pipeline.run(h.pipeline.create("x").run_id)
    assert index.status == RunStatus.FAILED
    assert index.error == "crosscheck: KeyError: 'boom'"
    assert h.pipeline.status(index.run_id).status == RunStatus.FAILED


def test_backend_returning_wrong_type_fails(h: Harness) -> None:
    h["strategy"].script = [lambda ctx: None]  # type: ignore[list-item,return-value]
    index = h.pipeline.run(h.pipeline.create("x").run_id)
    assert index.status == RunStatus.FAILED
    assert "TypeError" in (index.error or "")


def test_disallowed_index_updates_are_ignored(h: Harness) -> None:
    default = h["ingestion"].default

    def sneaky(ctx: StageContext) -> StageOutput:
        out = default(ctx)
        out.index_updates.update({"budget_usd": 1000.0, "status": "completed", "stage": "final"})
        return out

    h["ingestion"].script = [sneaky]
    run_id = h.pipeline.create("x", review=True).run_id
    index = h.pipeline.run(run_id)
    assert index.status == RunStatus.AWAITING_REVIEW
    assert index.budget_usd == 25.0
    assert index.mode == "code"


def test_invalid_index_update_value_fails(h: Harness) -> None:
    h["ingestion"].script = [lambda ctx: StageOutput(notes=[], index_updates={"mode": "poetry"})]
    index = h.pipeline.run(h.pipeline.create("x").run_id)
    assert index.status == RunStatus.FAILED
    assert index.stage == "ingestion"
    assert "mode" in (index.error or "")


def test_providers_factory_failure_fails_run(h: Harness) -> None:
    def broken(_settings: Settings, _ws: Path):  # type: ignore[no-untyped-def]
        raise RuntimeError("no claude binary")

    pipeline = h.make_pipeline(providers_factory=broken)
    index = pipeline.run(pipeline.create("x").run_id)
    assert index.status == RunStatus.FAILED
    assert index.error == "providers: RuntimeError: no claude binary"
    assert h["ingestion"].calls == 0


def test_providers_built_once_per_run_with_workspace(h: Harness) -> None:
    seen: list[Path] = []

    def factory(settings: Settings, workspace: Path):  # type: ignore[no-untyped-def]
        seen.append(workspace)
        return h.fakes.as_providers()

    pipeline = h.make_pipeline(providers_factory=factory)
    index = pipeline.run(pipeline.create("x").run_id)
    assert index.status == RunStatus.COMPLETED
    assert seen == [Path(index.workspace)]


def test_progress_callback_errors_do_not_break_run(h: Harness) -> None:
    def explode(_i: RunIndex, _m: str) -> None:
        raise RuntimeError("observer bug")

    assert h.pipeline.run(h.pipeline.create("x").run_id, progress=explode).status == RunStatus.COMPLETED


def test_keyboard_interrupt_is_recorded_and_reraised(h: Harness) -> None:
    def interrupt(ctx: StageContext) -> StageOutput:
        raise KeyboardInterrupt

    h["execution"].script = [interrupt]
    run_id = h.pipeline.create("x").run_id
    with pytest.raises(KeyboardInterrupt):
        h.pipeline.run(run_id)
    index = h.pipeline.status(run_id)
    assert index.status == RunStatus.FAILED
    assert "interrupted" in (index.error or "")
    assert h.pipeline.resume(run_id).status == RunStatus.COMPLETED  # guard was released


# --------------------------------------------------------------------------- crash recovery and concurrency


def test_crashed_running_run_continues_from_recorded_stage(h: Harness) -> None:
    run_id = h.pipeline.create("x", review=True).run_id
    h.pipeline.run(run_id)  # stops at the gate after strategy
    index = h.pipeline.status(run_id)
    index.status = RunStatus.RUNNING  # simulate a crash mid-execution in another process
    index.review = False
    h.pipeline.vault.write_index(index)

    fresh = h.make_pipeline()
    result = fresh.run(run_id)
    assert result.status == RunStatus.COMPLETED
    assert (h["ingestion"].calls, h["strategy"].calls, h["execution"].calls) == (1, 1, 1)


def test_second_concurrent_run_of_same_id_is_rejected(h: Harness) -> None:
    entered, release = threading.Event(), threading.Event()
    default = h["strategy"].default

    def blocking(ctx: StageContext) -> StageOutput:
        entered.set()
        assert release.wait(5)
        return default(ctx)

    h["strategy"].script = [blocking]
    run_id = h.pipeline.create("x").run_id
    results: list[RunIndex] = []
    worker = threading.Thread(target=lambda: results.append(h.pipeline.run(run_id)))
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(RuntimeError, match="already running"):
            h.pipeline.run(run_id)
        with pytest.raises(RuntimeError, match="already running"):
            h.make_pipeline().resume(run_id)  # the guard is process-wide, not per instance
        other = h.pipeline.create("another")
        assert h.pipeline.run(other.run_id).status == RunStatus.COMPLETED  # different runs are fine
    finally:
        release.set()
        worker.join(5)
    assert results and results[0].status == RunStatus.COMPLETED


# --------------------------------------------------------------------------- export (maf export)


def test_export_rewrites_deliverables_and_notes_it_without_model_calls(h: Harness) -> None:
    run_id = h.pipeline.run(h.pipeline.create("x").run_id).run_id  # scripted stages: mode code, completed
    paths = h.pipeline.vault.paths(run_id)
    (paths.workspace / "src").mkdir()
    (paths.workspace / "src" / "alloc.c").write_text("int x;\n")
    (paths.workspace / ".maf").mkdir(exist_ok=True)
    (paths.workspace / ".maf" / "prompt.md").write_text("p")
    before = h.pipeline.status(run_id)

    index, export = h.pipeline.export(run_id)

    assert (paths.deliverables / "src" / "alloc.c").read_text() == "int x;\n"
    assert not (paths.deliverables / ".maf").exists()
    assert export.files == 1 and export.tree is not None and export.tree.files == ("src/alloc.c",)
    assert index.export_note == "maf export: 1 file(s), 7 B"
    assert index.exported_at == index.updated and index.updated > before.updated
    assert index.model_copy(update={"exported_at": None, "export_note": None, "updated": before.updated}) == before
    assert h.pipeline.status(run_id) == index
    assert all(not f.calls for f in h.fakes.all())


def test_export_honours_settings_excludes_and_cap(h: Harness) -> None:
    run_id = h.pipeline.run(h.pipeline.create("x").run_id).run_id
    workspace = h.pipeline.vault.paths(run_id).workspace
    (workspace / "a.log").write_text("log")
    (workspace / "b.c").write_text("c" * 100)
    h.settings = h.settings.model_copy(update={"export_exclude": ("*.log",), "export_max_mb": 0.001})
    pipeline = h.make_pipeline()
    assert pipeline.export(run_id)[1].tree.files == ("b.c",)  # type: ignore[union-attr]
    h.settings = h.settings.model_copy(update={"export_max_mb": 0.00001})
    with pytest.raises(ExportTooLarge):
        h.make_pipeline().export(run_id)
    assert h.pipeline.status(run_id).export_note == "maf export: 1 file(s), 100 B; excluded: a.log"  # the refused export wrote nothing


def test_export_refuses_runs_without_a_mode_and_busy_runs(h: Harness) -> None:
    fresh = h.pipeline.create("x").run_id
    with pytest.raises(ExportError, match="no execution mode"):
        h.pipeline.export(fresh)
    assert h.pipeline.status(fresh).exported_at is None
    with pytest.raises(FileNotFoundError):
        h.pipeline.export("2026-09-28-nope")

    done = h.pipeline.run(h.pipeline.create("y").run_id).run_id
    with h.make_pipeline()._guard(done):
        with pytest.raises(RuntimeError, match="already running"):
            h.pipeline.export(done)


def test_notes_are_persisted_before_run_md(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    vault = h.pipeline.vault
    real_write_handoff, real_write_index = vault.write_handoff, vault.write_index

    def write_handoff(run_id: str, name: str, handoff):  # type: ignore[no-untyped-def]
        order.append(f"note:{name}")
        return real_write_handoff(run_id, name, handoff)

    def write_index(index: RunIndex) -> None:
        order.append(f"index:{index.stage}:{len(index.handoffs)}")
        real_write_index(index)

    monkeypatch.setattr(vault, "write_handoff", write_handoff)
    monkeypatch.setattr(vault, "write_index", write_index)
    h.pipeline.run(h.pipeline.create("x").run_id)
    start = order.index("note:01a-routing")
    assert order[start : start + 3] == ["note:01a-routing", "note:01-ingestion", "index:strategy:2"]


def test_failed_index_write_repeats_only_current_stage(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = h.pipeline.create("x").run_id
    vault = h.pipeline.vault
    real = vault.write_index
    calls = {"n": 0}

    def flaky(index: RunIndex) -> None:
        if index.stage == "crosscheck" and index.status == RunStatus.RUNNING and calls["n"] == 0:
            calls["n"] += 1
            raise OSError("disk full")
        real(index)

    monkeypatch.setattr(vault, "write_index", flaky)
    index = h.pipeline.run(run_id)
    assert index.status == RunStatus.FAILED
    assert index.stage == "execution"
    assert "disk full" in (index.error or "")
    assert h.pipeline.resume(run_id).status == RunStatus.COMPLETED
    assert h["execution"].calls == 2
    assert h.pipeline.status(run_id).handoffs.count("03-execution") == 1


def test_list_runs_and_ledger_for(h: Harness) -> None:
    a = h.pipeline.create("first")
    b = h.pipeline.create("second")
    runs = h.pipeline.list_runs()
    assert [r.run_id for r in runs] == [b.run_id, a.run_id]
    ledger = h.pipeline.ledger_for(a)
    assert ledger.cap_usd == a.budget_usd
    assert ledger.path == h.pipeline.vault.paths(a.run_id).ledger
    assert ledger.spent_usd == 0.0


@pytest.mark.parametrize("budget", [0, -1.0, float("nan"), float("inf")])
def test_resume_rejects_nonpositive_or_non_finite_budget(h: Harness, budget: float) -> None:
    run_id = h.pipeline.create("x").run_id
    with pytest.raises(ValueError):
        h.pipeline.resume(run_id, budget_usd=budget)


# --------------------------------------------------------------------------- end to end with the real stage backends


def _kind_of(request: CompletionRequest) -> str:
    """The handoff kind a generation prompt asks for (``format_spec`` heading), or the schema name."""
    if request.json_schema is not None:
        return request.schema_name
    prompt = request.messages[-1].content
    for kind in HandoffKind:
        if f"## Output format: {kind.value} handoff" in prompt:
            return kind.value
    raise AssertionError(f"cannot tell what this request wants: {prompt[:200]!r}")


ACCEPTANCE = "## Acceptance\n\n- AC-1 [met]: all suites pass\n- AC-2 [met]: text=1804\n"


def _cleanroom(settings: Settings, exit_code: int = 0) -> dict[str, object]:
    """The clean-room session of the (single) run under ``settings.workspaces_path``."""
    (workspace,) = [p for p in settings.workspaces_path.iterdir() if p.name != CLEANROOM_ROOT]
    (cleanroom_dir(workspace) / REPRO_LOG).write_text("make all\n")
    (cleanroom_dir(workspace) / REPRO_EXIT).write_text(f"{exit_code}\n")
    return {"command": "make all", "exit_code": exit_code, "missing_paths": [], "log_tail": "ok"}


def _script_real_run(fakes, bodies: dict[str, str], *, unfixed_rounds: int, settings: Settings) -> dict[str, list[str]]:  # type: ignore[no-untyped-def]
    """Script all four fakes by request content. GPT-1 (critical) stays unfixed for ``unfixed_rounds`` rounds; the
    final note meets both criteria and the clean room rebuilds the export."""
    seen: dict[str, list[str]] = {"chatgpt": [], "gemini": [], "claude": [], "claude_code": []}
    fix_rounds = {"n": 0}
    triage = {
        "summary": "Build a portable O(1) allocator.",
        "execution_mode": "code",
        "gemini_instructions": "Survey O(1) allocators.",
        "search_queries": ["TLSF allocator"],
        "deliverable": "alloc.c with tests",
    }

    def critique(prefix: str) -> str:
        severity = "critical" if prefix == "GPT" else "minor"
        return f"## Summary\n\nReviewed.\n\n## Issues\n\n- [{severity}] {prefix}-1: issue from {prefix}\n"

    def chatgpt(req: CompletionRequest) -> str | dict[str, object]:
        kind = _kind_of(req)
        seen["chatgpt"].append(kind)
        if kind == "triage":
            return triage
        if kind == "critique":
            return critique("GPT")
        if kind == "adjudication":
            return "## Summary\n\nRuled.\n\n## Rulings\n\n- GEM-1 [wontfix]: cosmetic\n"
        return bodies[kind]

    def gemini(req: CompletionRequest) -> str:
        kind = _kind_of(req)
        seen["gemini"].append(kind)
        return critique("GEM") if kind == "critique" else bodies[kind]

    def claude(req: CompletionRequest) -> str:
        kind = _kind_of(req)
        seen["claude"].append(kind)
        if kind == "critique":
            return critique("CLA")
        if kind == "rebuttal":
            return (
                "## Summary\n\nAnswered.\n\n## Responses\n\n- GPT-1 [accept]: will fix\n"
                "- GEM-1 [reject]: cosmetic\n- CLA-1 [accept]: will fix\n"
            )
        if kind == "final":
            return f"{bodies['final'].rstrip()}\n\n{ACCEPTANCE}"
        return bodies[kind]

    def claude_code(req: CompletionRequest) -> str | dict[str, object]:
        kind = _kind_of(req)
        seen["claude_code"].append(kind)
        if kind == "fix_report":
            fix_rounds["n"] += 1
            if fix_rounds["n"] <= unfixed_rounds:
                return {"fixed": ["CLA-1"], "not_fixed": [{"id": "GPT-1", "reason": "needs redesign"}], "summary": "partial"}
            return {"fixed": ["GPT-1", "CLA-1"], "not_fixed": [], "summary": "all fixed"}
        if kind == "cleanroom":
            return _cleanroom(settings)
        return bodies[kind]

    fakes.chatgpt.default = chatgpt
    fakes.gemini.default = gemini
    fakes.claude.default = claude
    fakes.claude_code.default = claude_code
    return seen


def _real_pipeline(settings: Settings, fakes) -> Pipeline:  # type: ignore[no-untyped-def]
    ticks = iter(range(10_000))
    return Pipeline(
        settings,
        providers_factory=fakes.factory(),
        clock=lambda: datetime(2026, 9, 28, 12, 0) + timedelta(seconds=next(ticks)),
    )


def _seed_workspace(pipeline: Pipeline, run_id: str) -> None:
    workspace = pipeline.vault.paths(run_id).workspace
    (workspace / "src").mkdir(parents=True, exist_ok=True)
    (workspace / "src" / "alloc.c").write_text("/* tlsf */\n", encoding="utf-8")
    (workspace / "test").mkdir(exist_ok=True)
    (workspace / "test" / "posix_test.log").write_text("42 passed\n", encoding="utf-8")


def test_end_to_end_with_real_backends(settings: Settings, fake_providers, sample_bodies: dict[str, str]) -> None:  # type: ignore[no-untyped-def]
    seen = _script_real_run(fake_providers, sample_bodies, unfixed_rounds=0, settings=settings)
    pipeline = _real_pipeline(settings, fake_providers)
    run_id = pipeline.create("Portable small-memory allocator").run_id
    _seed_workspace(pipeline, run_id)

    index = pipeline.run(run_id)
    assert index.status == RunStatus.COMPLETED, index.error
    assert index.mode == "code"
    assert index.round == 1
    assert index.unresolved_critical == 0
    assert index.handoffs[:3] == ["01a-routing", "01-ingestion", "02-strategy"]
    assert index.handoffs[-2:] == ["04-crosscheck", "05-final"]
    for name in ("04a-critique-chatgpt", "04a-critique-gemini", "04a-critique-claude", "04b-rebuttal", "04c-adjudication"):
        assert name in index.handoffs
        assert pipeline.vault.has_note(run_id, name)
    crosscheck = pipeline.vault.read_handoff(run_id, "04-crosscheck")
    assert crosscheck.section("Verdict").strip() == "PASS"
    assert seen["chatgpt"][:2] == ["triage", "strategy"]
    assert seen["claude_code"] == ["execution", "fix_report", "cleanroom"]
    assert (pipeline.vault.paths(run_id).deliverables / "src" / "alloc.c").is_file()
    # Every call went through the ledger and is mirrored into run.md.
    calls = sum(len(f.calls) for f in fake_providers.all())
    assert calls == len(pipeline.ledger_for(index).entries)
    assert index.spent_usd == pytest.approx(pipeline.ledger_for(index).spent_usd)
    assert index.spend_by_agent["claude"] > 0 and index.spend_by_agent["gemini"] > 0


def test_end_to_end_crosscheck_loop_with_real_backends(settings: Settings, fake_providers, sample_bodies: dict[str, str]) -> None:  # type: ignore[no-untyped-def]
    seen = _script_real_run(fake_providers, sample_bodies, unfixed_rounds=1, settings=settings)
    pipeline = _real_pipeline(settings, fake_providers)
    run_id = pipeline.create("Portable small-memory allocator").run_id
    _seed_workspace(pipeline, run_id)

    index = pipeline.run(run_id)
    assert index.status == RunStatus.COMPLETED, index.error
    assert index.round == 2
    assert pipeline.vault.read_handoff(run_id, "04-crosscheck").section("Verdict").strip() == "LOOP"
    assert pipeline.vault.read_handoff(run_id, "04-crosscheck-r2").section("Verdict").strip() == "PASS"
    assert "03-execution-r2" in index.handoffs
    assert seen["claude_code"] == ["execution", "fix_report", "execution", "fix_report", "cleanroom"]


def test_end_to_end_review_gate_with_real_backends(settings: Settings, fake_providers, sample_bodies: dict[str, str]) -> None:  # type: ignore[no-untyped-def]
    _script_real_run(fake_providers, sample_bodies, unfixed_rounds=0, settings=settings)
    pipeline = _real_pipeline(settings, fake_providers)
    run_id = pipeline.create("Portable small-memory allocator", review=True).run_id
    _seed_workspace(pipeline, run_id)

    assert pipeline.run(run_id).status == RunStatus.AWAITING_REVIEW
    index = pipeline.resume(run_id, note="Keep code size under 1.5 KB.")
    assert index.status == RunStatus.COMPLETED, index.error
    execution_prompt = fake_providers.claude_code.calls[0].messages[-1].content
    assert "Keep code size under 1.5 KB." in execution_prompt
