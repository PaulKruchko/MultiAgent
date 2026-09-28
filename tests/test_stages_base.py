"""Tests for ``maf.stages.base``: metered calls, the generate-validate-repair loop, prompt inputs.

Also hosts the shared ``StageEnv`` helper and ``stage_env`` fixture that the other stage test modules
import: a real ``Vault`` in a temp dir, a real in-memory ``Ledger``, and the conftest ``FakeProviders``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pytest

from conftest import FakeProviders
from maf import handoff as hf
from maf.config import Settings
from maf.handoff import Handoff, HandoffInvalid, HandoffKind, HandoffMeta
from maf.ledger import BudgetExceeded, Ledger
from maf.providers import Attachment, CompletionRequest, ProviderError
from maf.stages import base
from maf.stages.base import (
    StageContext,
    as_wikilink,
    generate,
    generate_handoff,
    neutralize_headings,
    one_line,
    render_inputs,
    review_block,
    role_system,
    with_sections,
    write_workspace_file,
)
from maf.types import ExecutionMode, StageName
from maf.vault import RunIndex, RunPaths, Vault

RUN_ID = "2026-09-28-test-run"


@dataclass
class StageEnv:
    """A real run folder plus fakes; ``ctx()`` builds a ``StageContext`` for any stage and round."""

    settings: Settings
    vault: Vault
    paths: RunPaths
    ledger: Ledger
    fakes: FakeProviders
    index: RunIndex
    now: datetime
    written: dict[str, Handoff] = field(default_factory=dict)

    def ctx(
        self,
        stage: StageName,
        *,
        round: int = 1,
        mode: ExecutionMode | None = None,
        review_note: str | None = None,
        unresolved_critical: int = 0,
    ) -> StageContext:
        index = self.index.model_copy(
            update={"stage": stage, "round": round, "mode": mode, "unresolved_critical": unresolved_critical}
        )
        return StageContext(
            index=index,
            settings=self.settings,
            vault=self.vault,
            paths=self.paths,
            ledger=self.ledger,
            providers=self.fakes.as_providers(),
            stage=stage,
            now=self.now,
            review_note=review_note,
        )

    def put(self, name: str, kind: HandoffKind, body: str, *, round: int = 1, from_: str = "claude") -> Handoff:
        """Write a prior note as the pipeline would and record it in ``index.handoffs``."""
        meta = HandoffMeta(
            run_id=RUN_ID, stage=kind, from_=from_, to="next", created=self.now, model="m", cost_usd=0.0, round=round
        )
        handoff = hf.build_handoff(body, meta)
        self.vault.write_handoff(RUN_ID, name, handoff)
        self.index.handoffs.append(name)
        self.written[name] = handoff
        return handoff

    def workspace_file(self, rel: str, content: str | bytes = "x\n") -> Path:
        path = self.paths.workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path


@pytest.fixture
def stage_env(settings: Settings, fake_providers: FakeProviders, fixed_now: datetime, tmp_path: Path) -> StageEnv:
    vault = Vault(settings.vault_path, settings.workspaces_path)
    brief_file = tmp_path / "spec.pdf"
    brief_file.write_bytes(b"%PDF-1.4 fake")
    index = RunIndex(
        run_id=RUN_ID,
        budget_usd=25.0,
        created=fixed_now,
        updated=fixed_now,
        workspace=str(settings.workspaces_path / RUN_ID),
        brief="Design a portable O(1) allocator.",
    )
    paths = vault.create_run(index, [brief_file])
    return StageEnv(
        settings=settings,
        vault=vault,
        paths=paths,
        ledger=Ledger(RUN_ID, 25.0),
        fakes=fake_providers,
        index=vault.read_index(RUN_ID),
        now=fixed_now,
    )


def prompt_of(request: CompletionRequest) -> str:
    return request.messages[-1].content


# ---------------------------------------------------------------------- StageContext


def test_call_meters_through_the_ledger(stage_env: StageEnv) -> None:
    ctx = stage_env.ctx("strategy")
    stage_env.fakes.chatgpt.script("hello")
    result = ctx.call("chatgpt", CompletionRequest.simple("gpt-6-sol", "hi", max_output_tokens=10), purpose="probe")
    assert result.text == "hello"
    (entry,) = stage_env.ledger.entries
    assert (entry.stage, entry.agent, entry.provider, entry.purpose) == ("strategy", "chatgpt", "openai", "probe")
    assert entry.cost_usd == pytest.approx(0.01)


def test_call_refuses_when_budget_is_exhausted(stage_env: StageEnv) -> None:
    ctx = stage_env.ctx("strategy")
    ctx.ledger = Ledger(RUN_ID, 0.01)
    with pytest.raises(BudgetExceeded):
        ctx.call("chatgpt", CompletionRequest.simple("gpt-6-sol", "hi", max_output_tokens=10))
    assert stage_env.fakes.chatgpt.calls == []


def test_call_retries_transient_errors(stage_env: StageEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(base, "_sleep", sleeps.append)
    stage_env.fakes.gemini.script(ProviderError("503", provider="gemini", retryable=True), "ok")
    result = stage_env.ctx("ingestion").call("gemini", CompletionRequest.simple("g", "hi", max_output_tokens=10))
    assert result.text == "ok"
    assert sleeps == [base.RETRY_BACKOFF_S[0]]
    assert len(stage_env.fakes.gemini.calls) == 2


def test_call_gives_up_after_max_retries(stage_env: StageEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "_sleep", lambda _s: None)
    err = ProviderError("overloaded", provider="anthropic", retryable=True)
    stage_env.fakes.claude.script(err, err, err, "never")
    with pytest.raises(ProviderError):
        stage_env.ctx("final").call("claude", CompletionRequest.simple("c", "hi", max_output_tokens=10))
    assert len(stage_env.fakes.claude.calls) == 1 + base.MAX_TRANSIENT_RETRIES


def test_call_does_not_retry_permanent_or_claude_code_errors(stage_env: StageEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "_sleep", lambda _s: pytest.fail("must not sleep"))
    stage_env.fakes.chatgpt.script(ProviderError("bad request", provider="openai"))
    stage_env.fakes.claude_code.script(ProviderError("crashed", provider="claude_code", retryable=True))
    ctx = stage_env.ctx("execution")
    with pytest.raises(ProviderError):
        ctx.call("chatgpt", CompletionRequest.simple("g", "hi", max_output_tokens=10))
    with pytest.raises(ProviderError):
        ctx.call("claude_code", CompletionRequest.simple("c", "hi", max_output_tokens=10, max_budget_usd=1.0))
    assert len(stage_env.fakes.claude_code.calls) == 1


def test_prior_notes_and_latest_note_name(stage_env: StageEnv, sample_bodies: dict[str, str]) -> None:
    stage_env.put("01-ingestion", HandoffKind.INGESTION, sample_bodies["ingestion"])
    stage_env.put("03-execution", HandoffKind.EXECUTION, sample_bodies["execution"])
    stage_env.put("03-execution-r2", HandoffKind.EXECUTION, sample_bodies["execution"], round=2)
    ctx = stage_env.ctx("final", round=3)
    notes = ctx.prior_notes()
    assert list(notes) == ["01-ingestion", "03-execution", "03-execution-r2"]
    assert notes["01-ingestion"].sections == stage_env.written["01-ingestion"].sections
    assert ctx.latest_note_name(HandoffKind.EXECUTION) == "03-execution-r2"
    assert ctx.latest_note_name(HandoffKind.CROSSCHECK) == "04-crosscheck-r3"


# ---------------------------------------------------------------------- generate_handoff


def test_generate_handoff_fills_meta_and_injects_format_spec(stage_env: StageEnv, sample_bodies: dict[str, str]) -> None:
    stage_env.fakes.chatgpt.script(sample_bodies["strategy"])
    ctx = stage_env.ctx("strategy", round=2)
    handoff = generate_handoff(
        ctx, "chatgpt", HandoffKind.STRATEGY, system="SYS", prompt="Do it.", to="claude", inputs=["01-ingestion"]
    )
    (request,) = stage_env.fakes.chatgpt.calls
    assert request.system == "SYS"
    assert prompt_of(request).startswith("Do it.")
    assert hf.format_spec(HandoffKind.STRATEGY) in prompt_of(request)
    assert request.effort == "high"
    assert request.max_output_tokens == stage_env.settings.output_limits.chatgpt
    meta = handoff.meta
    assert (meta.from_, meta.to, meta.model, meta.round, meta.stage) == (
        "chatgpt", "claude", ctx.model("chatgpt"), 2, HandoffKind.STRATEGY,
    )
    assert meta.inputs == ["[[01-ingestion]]"]
    assert meta.cost_usd == pytest.approx(0.01)
    assert meta.created == stage_env.now
    assert "maf/strategy" in meta.tags
    assert handoff.section("Chosen Strategy").startswith("TLSF")


def test_format_spec_not_duplicated_when_already_in_prompt(stage_env: StageEnv, sample_bodies: dict[str, str]) -> None:
    spec = hf.format_spec(HandoffKind.INGESTION)
    stage_env.fakes.gemini.script(sample_bodies["ingestion"])
    generate_handoff(
        stage_env.ctx("ingestion"), "gemini", HandoffKind.INGESTION,
        system="", prompt=f"task\n\n{spec}", to="strategy", inputs=[],
    )
    assert prompt_of(stage_env.fakes.gemini.calls[0]).count(spec) == 1
    assert stage_env.fakes.gemini.calls[0].effort is None


def test_repair_once_then_success_sums_costs(stage_env: StageEnv, sample_bodies: dict[str, str]) -> None:
    bad = "## Summary\n\nno other sections"
    stage_env.fakes.gemini.script(bad, sample_bodies["ingestion"])
    attachment = Attachment(path=stage_env.paths.workspace / "inputs" / "spec.pdf")
    generated = generate(
        stage_env.ctx("ingestion"), "gemini", HandoffKind.INGESTION,
        system="SYS", prompt="p", to="strategy", inputs=[],
        web_search=True, attachments=(attachment,), max_search_queries=7,
    )
    first, repair = stage_env.fakes.gemini.calls
    assert first.web_search and first.attachments
    assert not repair.web_search and repair.attachments == ()
    assert repair.system == "SYS"
    assert "did not pass validation" in prompt_of(repair)
    assert bad in prompt_of(repair)
    assert generated.handoff.meta.cost_usd == pytest.approx(0.02)
    assert len(generated.results) == 2
    assert [e.purpose for e in stage_env.ledger.entries] == ["ingestion", "repair"]


def test_second_invalid_body_raises(stage_env: StageEnv) -> None:
    stage_env.fakes.claude.script("garbage", "still garbage", "unused")
    with pytest.raises(HandoffInvalid) as excinfo:
        generate_handoff(stage_env.ctx("final"), "claude", HandoffKind.FINAL, system="", prompt="p", to="user", inputs=[])
    assert excinfo.value.kind == HandoffKind.FINAL
    assert len(stage_env.fakes.claude.calls) == 2


def test_check_errors_go_through_the_repair(stage_env: StageEnv, sample_bodies: dict[str, str]) -> None:
    stage_env.fakes.chatgpt.script(sample_bodies["strategy"], sample_bodies["strategy"].replace("TLSF:", "Buddy:"))
    seen: list[str] = []

    def check(handoff: Handoff) -> list[str]:
        seen.append(handoff.section("Chosen Strategy"))
        return ["pick the buddy allocator"] if "TLSF" in handoff.section("Chosen Strategy") else []

    handoff = generate_handoff(
        stage_env.ctx("strategy"), "chatgpt", HandoffKind.STRATEGY,
        system="", prompt="p", to="claude", inputs=[], check=check,
    )
    assert handoff.section("Chosen Strategy").startswith("Buddy")
    assert "pick the buddy allocator" in prompt_of(stage_env.fakes.chatgpt.calls[1])
    assert len(seen) == 2


def test_claude_code_gets_a_clamped_budget_and_attribution(stage_env: StageEnv, sample_bodies: dict[str, str]) -> None:
    stage_env.fakes.claude_code.script(sample_bodies["execution"])
    ctx = stage_env.ctx("execution", mode="code")
    ctx.ledger = Ledger(RUN_ID, 3.0)
    handoff = generate_handoff(ctx, "claude_code", HandoffKind.EXECUTION, system="", prompt="p", to="crosscheck", inputs=[])
    (request,) = stage_env.fakes.claude_code.calls
    assert request.max_budget_usd == pytest.approx(3.0)  # 8.0 default clamped to the remaining budget
    assert handoff.meta.from_ == "claude"
    assert handoff.meta.cost_usd == pytest.approx(0.50)


# ---------------------------------------------------------------------- helpers


def test_render_inputs_selects_sections(stage_env: StageEnv, sample_bodies: dict[str, str]) -> None:
    strategy = stage_env.put("02-strategy", HandoffKind.STRATEGY, sample_bodies["strategy"])
    execution = stage_env.put("03-execution", HandoffKind.EXECUTION, sample_bodies["execution"])
    text = render_inputs(
        {"02-strategy": strategy, "03-execution": execution},
        {"02-strategy": ("Acceptance Criteria", "No Such Section")},
    )
    assert text.startswith('<note name="02-strategy">\n## Acceptance Criteria')
    assert "## Risks" not in text
    assert '<note name="03-execution">' in text and "## Known Limitations" in text
    assert text.count("</note>") == 2
    assert render_inputs({"02-strategy": strategy}, {"02-strategy": ("Nope",)}).count("(no sections)") == 1


def test_role_system_prompts(settings: Settings) -> None:
    assert str(settings.python_executable) in role_system("claude_code", settings)
    for role in ("chatgpt", "gemini", "claude"):
        assert role_system(role, settings)  # type: ignore[arg-type]


def test_with_sections_revalidates(stage_env: StageEnv, sample_bodies: dict[str, str]) -> None:
    execution = stage_env.put("03-execution", HandoffKind.EXECUTION, sample_bodies["execution"])
    updated = with_sections(execution, {"Summary": "  new  "})
    assert updated.section("Summary") == "new"
    assert list(updated.sections) == list(execution.sections)
    with pytest.raises(HandoffInvalid):
        with_sections(execution, {"Summary": ""})


def test_small_text_helpers(stage_env: StageEnv) -> None:
    assert neutralize_headings("# a\ntext\n  ## b\n#nospace") == "\\# a\ntext\n  \\## b\n#nospace"
    assert one_line("a\n  b\tc") == "a b c"
    assert one_line("x" * 20, 10) == "xxxxxxx..."
    assert as_wikilink("n") == "[[n]]" and as_wikilink("[[n|a]]") == "[[n|a]]"
    assert review_block(None) == "" and review_block("  ") == ""
    assert "Use TLSF" in review_block("Use TLSF")
    path = write_workspace_file(stage_env.ctx("execution"), ".maf/p.md", "hello")
    assert path.read_text() == "hello"
    assert not [p for p in path.parent.iterdir() if p.name.endswith(".tmp")]


# ---------------------------------------------------------------------- end to end


def test_default_backends_drive_a_full_run_with_one_loop(
    settings: Settings, fake_providers: FakeProviders, fixed_now: datetime, sample_bodies: dict[str, str]
) -> None:
    """Every stage through the real pipeline and vault: code mode, one LOOP, then PASS and final."""
    from maf.pipeline import Pipeline
    from maf.types import RunStatus

    execution = sample_bodies["execution"]
    critical = "## Summary\n\nBug.\n\n## Issues\n\n- [critical] GPT-1: stress test crashes\n"
    clean = "## Summary\n\nFine.\n\n## Issues\n\nNone.\n"

    def build(_request: CompletionRequest) -> str:
        workspace = next(settings.workspaces_path.iterdir())
        (workspace / "src").mkdir(exist_ok=True)
        (workspace / "src" / "alloc.c").write_text("int x;\n")
        (workspace / "test").mkdir(exist_ok=True)
        (workspace / "test" / "posix_test.log").write_text("ok\n")
        return execution

    fake_providers.chatgpt.script(
        {
            "summary": "Allocator.",
            "execution_mode": "code",
            "gemini_instructions": "Survey TLSF.",
            "search_queries": ["TLSF"],
            "deliverable": "C allocator.",
        },
        sample_bodies["strategy"],
        critical,  # round 1 critique
        clean,  # round 2 critique
    )
    fake_providers.gemini.script(sample_bodies["ingestion"], clean, clean)
    fake_providers.claude.script(
        clean,
        "## Summary\n\nOk.\n\n## Responses\n\n- GPT-1 [accept]: will fix\n",
        clean,
        sample_bodies["final"],
    )
    fake_providers.claude_code.script(
        build,
        {"fixed": [], "not_fixed": [{"id": "GPT-1", "reason": "needs a redesign"}], "summary": "tried"},
        build,
    )

    pipeline = Pipeline(settings, providers_factory=fake_providers.factory(), clock=lambda: fixed_now)
    index = pipeline.run(pipeline.create("Design a portable O(1) allocator.").run_id)

    assert index.status == RunStatus.COMPLETED, index.error
    assert index.mode == "code" and index.round == 2 and index.unresolved_critical == 0
    assert index.handoffs[:3] == ["01a-routing", "01-ingestion", "02-strategy"]
    assert "04-crosscheck" in index.handoffs and "04-crosscheck-r2" in index.handoffs
    assert index.handoffs[-1] == "05-final"
    vault = Vault(settings.vault_path, settings.workspaces_path)
    assert vault.read_handoff(index.run_id, "04-crosscheck").section("Verdict") == "LOOP"
    assert vault.read_handoff(index.run_id, "04-crosscheck-r2").section("Verdict") == "PASS"
    assert (vault.paths(index.run_id).deliverables / "src" / "alloc.c").is_file()
    assert all(not f.replies for f in fake_providers.all())
