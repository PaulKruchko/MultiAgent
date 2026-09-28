"""Tests for maf.ledger: persistence, budget checks, aggregation and metered_call."""

from __future__ import annotations

import json
import shutil
import threading
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from conftest import FakeProvider
from maf.ledger import (
    MIN_CLAUDE_CODE_BUDGET_USD,
    BudgetExceeded,
    Ledger,
    LedgerEntry,
    estimate_tokens,
    metered_call,
)
from maf.providers import CompletionRequest, CompletionResult, ProviderError, StructuredOutputError
from maf.types import AgentName, ProviderName, StageName, Usage

FIXTURES = Path(__file__).parent / "fixtures" / "ledger"
RUN = "2026-09-28-demo"


def _entry(
    cost: float,
    *,
    agent: AgentName = "chatgpt",
    provider: ProviderName = "openai",
    stage: StageName = "ingestion",
    run_id: str = RUN,
) -> LedgerEntry:
    return LedgerEntry(
        ts=datetime(2026, 9, 28, 12, 0, tzinfo=UTC),
        run_id=run_id,
        stage=stage,
        agent=agent,
        provider=provider,
        model="m",
        usage=Usage(input_tokens=10, output_tokens=5),
        cost_usd=cost,
        worst_case_usd=cost * 2,
        purpose="t",
    )


def _req(model: str = "gpt-6-sol", **kw: object) -> CompletionRequest:
    return CompletionRequest.simple(model, "hello", max_output_tokens=100, **kw)  # type: ignore[arg-type]


# -- basics ----------------------------------------------------------------------------------


def test_empty_ledger() -> None:
    ledger = Ledger(RUN, 10.0)
    assert ledger.entries == ()
    assert ledger.spent_usd == 0.0
    assert ledger.remaining_usd == 10.0
    assert ledger.by_agent() == {"chatgpt": 0.0, "gemini": 0.0, "claude": 0.0}
    assert ledger.by_provider() == {}
    assert ledger.by_stage() == {}


@pytest.mark.parametrize("cap", [-1.0, float("nan"), float("inf")])
def test_invalid_cap_rejected(cap: float) -> None:
    with pytest.raises(ValueError):
        Ledger(RUN, cap)


def test_record_and_aggregate() -> None:
    ledger = Ledger(RUN, 10.0)
    ledger.record(_entry(0.10, stage="strategy"))
    ledger.record(_entry(0.20, agent="gemini", provider="gemini"))
    ledger.record(_entry(1.00, agent="claude", provider="claude_code", stage="execution"))
    ledger.record(_entry(0.30, agent="claude", provider="anthropic", stage="final"))
    assert ledger.spent_usd == pytest.approx(1.6)
    assert ledger.remaining_usd == pytest.approx(8.4)
    assert ledger.by_agent() == pytest.approx({"chatgpt": 0.1, "gemini": 0.2, "claude": 1.3})
    assert ledger.by_provider() == pytest.approx({"openai": 0.1, "gemini": 0.2, "anthropic": 0.3, "claude_code": 1.0})
    assert list(ledger.by_stage()) == ["ingestion", "strategy", "execution", "final"]
    assert ledger.by_stage()["execution"] == pytest.approx(1.0)


def test_entries_is_snapshot() -> None:
    ledger = Ledger(RUN, 10.0)
    snap = ledger.entries
    ledger.record(_entry(0.1))
    assert snap == ()
    assert len(ledger.entries) == 1


def test_remaining_floors_at_zero_after_overrun() -> None:
    ledger = Ledger(RUN, 1.0)
    ledger.record(_entry(1.5))
    assert ledger.remaining_usd == 0.0
    assert ledger.spent_usd == 1.5


def test_record_rejects_other_run() -> None:
    with pytest.raises(ValueError, match="other"):
        Ledger(RUN, 1.0).record(_entry(0.1, run_id="other"))


# -- check -----------------------------------------------------------------------------------


def test_check_allows_equality_and_rejects_over() -> None:
    ledger = Ledger(RUN, 1.0)
    ledger.record(_entry(0.4))
    ledger.check(0.6, "exact")
    with pytest.raises(BudgetExceeded) as info:
        ledger.check(0.6001, "over")
    err = info.value
    assert (err.cap_usd, err.spent_usd, err.requested_usd, err.what) == (1.0, pytest.approx(0.4), 0.6001, "over")
    assert "over" in str(err)


def test_check_tolerates_float_rounding() -> None:
    ledger = Ledger(RUN, 0.3)
    ledger.record(_entry(0.1))
    ledger.record(_entry(0.2 - 0.1))
    ledger.check(ledger.remaining_usd, "rest")


def test_check_rejects_negative_worst_case() -> None:
    with pytest.raises(ValueError):
        Ledger(RUN, 1.0).check(-0.1, "x")


def test_zero_cap_allows_only_free_calls() -> None:
    ledger = Ledger(RUN, 0.0)
    ledger.check(0.0, "free")
    with pytest.raises(BudgetExceeded):
        ledger.check(0.01, "paid")


# -- persistence -----------------------------------------------------------------------------


def test_persist_and_reload_exact(tmp_path: Path) -> None:
    path = tmp_path / "runs" / RUN / "ledger.jsonl"
    ledger = Ledger(RUN, 5.0, path)
    entries = [_entry(0.1), _entry(0.2, agent="claude", provider="anthropic"), _entry(1e-7)]
    for e in entries:
        ledger.record(e)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["cost_usd"] == 0.1
    reloaded = Ledger.load(RUN, 5.0, path)
    assert reloaded.entries == tuple(entries)
    assert reloaded.spent_usd == ledger.spent_usd
    assert reloaded.path == path


def test_load_missing_file_binds_path(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger.load(RUN, 5.0, path)
    assert ledger.entries == ()
    ledger.record(_entry(0.5))
    assert Ledger.load(RUN, 5.0, path).spent_usd == 0.5


def test_load_recorded_fixture_with_raised_cap(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    shutil.copy(FIXTURES / "resume.jsonl", path)
    ledger = Ledger.load(RUN, 50.0, path)  # resume --budget 50
    assert ledger.cap_usd == 50.0
    assert ledger.spent_usd == pytest.approx(2.9)
    assert ledger.by_agent() == pytest.approx({"chatgpt": 0.038, "gemini": 0.362, "claude": 2.5})
    assert ledger.entries[2].error.startswith("ProviderError")
    assert ledger.entries[1].usage.search_queries == 6


def test_load_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text(_entry(0.1).model_dump_json() + "\n\n" + _entry(0.2).model_dump_json() + "\n", encoding="utf-8")
    assert Ledger.load(RUN, 5.0, path).spent_usd == pytest.approx(0.3)


def test_load_truncates_torn_final_line(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "ledger.jsonl"
    good = _entry(0.1).model_dump_json() + "\n"
    path.write_text(good + '{"ts": "2026-09-28T12:0', encoding="utf-8")
    ledger = Ledger.load(RUN, 5.0, path)
    assert ledger.spent_usd == 0.1
    assert path.read_text(encoding="utf-8") == good
    assert "torn" in caplog.text
    ledger.record(_entry(0.2))
    assert Ledger.load(RUN, 5.0, path).spent_usd == pytest.approx(0.3)


def test_load_keeps_complete_final_line_without_newline(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text(_entry(0.1).model_dump_json(), encoding="utf-8")
    ledger = Ledger.load(RUN, 5.0, path)
    assert ledger.spent_usd == 0.1
    ledger.record(_entry(0.2))
    assert Ledger.load(RUN, 5.0, path).spent_usd == pytest.approx(0.3)


def test_load_rejects_mid_file_corruption(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text("garbage\n" + _entry(0.1).model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        Ledger.load(RUN, 5.0, path)


def test_load_rejects_foreign_run(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_text(_entry(0.1, run_id="other").model_dump_json() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="other"):
        Ledger.load(RUN, 5.0, path)


def test_record_fsyncs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import maf.ledger as ledger_mod

    calls: list[int] = []
    real = ledger_mod.os.fsync
    monkeypatch.setattr(ledger_mod.os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    ledger = Ledger(RUN, 5.0, tmp_path / "ledger.jsonl")
    ledger.record(_entry(0.1))
    ledger.record(_entry(0.1))
    assert len(calls) >= 2


def test_concurrent_records_are_all_persisted(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(RUN, 1000.0, path)

    def worker() -> None:
        for _ in range(50):
            ledger.record(_entry(0.01))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(ledger.entries) == 400
    assert Ledger.load(RUN, 1000.0, path).spent_usd == pytest.approx(4.0)


# -- estimate_tokens -------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "tokens"), [("", 0), ("a", 1), ("abc", 1), ("abcd", 2), ("x" * 3000, 1000)])
def test_estimate_tokens(text: str, tokens: int) -> None:
    assert estimate_tokens(text) == tokens


# -- metered_call ----------------------------------------------------------------------------


def test_metered_call_records_success() -> None:
    ledger = Ledger(RUN, 1.0)
    provider = FakeProvider(name="openai", agent="chatgpt", cost_per_call=0.02, worst_case_usd=0.3).script("hi")
    result = metered_call(ledger, provider, _req(), stage="strategy", purpose="options")
    assert result.text == "hi"
    (entry,) = ledger.entries
    assert entry.cost_usd == 0.02
    assert entry.worst_case_usd == 0.3
    assert (entry.stage, entry.agent, entry.provider, entry.model, entry.purpose) == (
        "strategy",
        "chatgpt",
        "openai",
        "gpt-6-sol",
        "options",
    )
    assert entry.usage == provider.usage
    assert entry.run_id == RUN
    assert entry.ts.tzinfo is not None
    assert entry.error == ""


def test_metered_call_refuses_before_calling() -> None:
    ledger = Ledger(RUN, 0.10)
    provider = FakeProvider(name="openai", agent="chatgpt", worst_case_usd=0.2).script("never")
    with pytest.raises(BudgetExceeded) as info:
        metered_call(ledger, provider, _req(), stage="ingestion", purpose="triage")
    assert provider.calls == []
    assert ledger.entries == ()
    assert "triage" in info.value.what


def test_metered_call_records_overrun_and_next_check_sees_it() -> None:
    ledger = Ledger(RUN, 1.0)
    provider = FakeProvider(name="openai", agent="chatgpt", cost_per_call=0.9, worst_case_usd=0.5)
    provider.script("a", "b")
    metered_call(ledger, provider, _req(), stage="strategy")
    assert ledger.spent_usd == 0.9
    with pytest.raises(BudgetExceeded):
        metered_call(ledger, provider, _req(), stage="strategy")
    assert len(provider.calls) == 1


def test_metered_call_clamps_claude_code_budget() -> None:
    ledger = Ledger(RUN, 3.0)
    ledger.record(_entry(2.0))
    provider = FakeProvider(name="claude_code", agent="claude", cost_per_call=0.4, worst_case_usd=100.0).script("ok")
    metered_call(ledger, provider, _req("claude-opus-5-5", max_budget_usd=8.0), stage="execution")
    sent = provider.calls[0]
    assert sent.max_budget_usd == pytest.approx(1.0)
    assert ledger.entries[-1].worst_case_usd == pytest.approx(1.0)
    assert ledger.entries[-1].provider == "claude_code"
    assert ledger.by_agent()["claude"] == pytest.approx(0.4)


def test_metered_call_keeps_smaller_budget() -> None:
    ledger = Ledger(RUN, 25.0)
    provider = FakeProvider(name="claude_code", agent="claude", worst_case_usd=100.0).script("ok")
    request = _req("claude-opus-5-5", max_budget_usd=8.0)
    metered_call(ledger, provider, request, stage="execution")
    assert provider.calls[0] is request


def test_metered_call_defaults_claude_code_budget_to_remaining() -> None:
    ledger = Ledger(RUN, 5.0)
    provider = FakeProvider(name="claude_code", agent="claude", worst_case_usd=100.0).script("ok")
    metered_call(ledger, provider, _req("claude-opus-5-5"), stage="execution")
    assert provider.calls[0].max_budget_usd == 5.0


def test_metered_call_claude_code_with_nothing_left_refuses() -> None:
    ledger = Ledger(RUN, 1.0)
    ledger.record(_entry(1.0))
    provider = FakeProvider(name="claude_code", agent="claude").script("never")
    with pytest.raises(BudgetExceeded):
        metered_call(ledger, provider, _req("claude-opus-5-5", max_budget_usd=8.0), stage="execution")
    assert provider.calls == []


def test_metered_call_claude_code_below_cli_minimum_refuses() -> None:
    ledger = Ledger(RUN, 1.0)
    ledger.record(_entry(1.0 - MIN_CLAUDE_CODE_BUDGET_USD / 2))
    provider = FakeProvider(name="claude_code", agent="claude").script("never")
    with pytest.raises(BudgetExceeded):
        metered_call(ledger, provider, _req("claude-opus-5-5", max_budget_usd=8.0), stage="execution")
    assert provider.calls == []


def test_metered_call_records_partial_spend_on_provider_error() -> None:
    ledger = Ledger(RUN, 10.0)
    err = ProviderError("error_max_budget_usd", provider="claude_code", cost_usd=1.75)
    provider = FakeProvider(name="claude_code", agent="claude").script(err)
    with pytest.raises(ProviderError) as info:
        metered_call(ledger, provider, _req("claude-opus-5-5", max_budget_usd=2.0), stage="execution", purpose="run")
    assert info.value is err
    (entry,) = ledger.entries
    assert entry.cost_usd == 1.75
    assert entry.usage == Usage()
    assert entry.model == "claude-opus-5-5"
    assert "error_max_budget_usd" in entry.error


def test_metered_call_records_partial_spend_on_error_subclass() -> None:
    ledger = Ledger(RUN, 10.0)
    provider = FakeProvider(name="anthropic", agent="claude").script(
        StructuredOutputError("bad json", provider="anthropic", cost_usd=0.3)
    )
    with pytest.raises(StructuredOutputError):
        metered_call(ledger, provider, _req("claude-opus-5-5"), stage="final")
    assert ledger.spent_usd == 0.3


def test_metered_call_zero_cost_error_not_recorded() -> None:
    ledger = Ledger(RUN, 10.0)
    provider = FakeProvider(name="openai", agent="chatgpt").script(ProviderError("503", provider="openai"))
    with pytest.raises(ProviderError):
        metered_call(ledger, provider, _req(), stage="strategy")
    assert ledger.entries == ()


def test_metered_call_releases_reservation_after_errors() -> None:
    ledger = Ledger(RUN, 2.0)
    provider = FakeProvider(name="openai", agent="chatgpt", worst_case_usd=1.0)
    provider.script(ProviderError("x", provider="openai"), RuntimeError("bug"), "ok")
    with pytest.raises(ProviderError):
        metered_call(ledger, provider, _req(), stage="strategy")
    with pytest.raises(RuntimeError):
        metered_call(ledger, provider, _req(), stage="strategy")
    assert ledger.spent_usd == pytest.approx(1.0)  # the unexpected error is charged its worst case
    assert metered_call(ledger, provider, _req(), stage="strategy").text == "ok"


@pytest.mark.parametrize("provider_name", ["openai", "claude_code"])
def test_metered_call_charges_worst_case_on_interrupt(tmp_path: Path, provider_name: ProviderName) -> None:
    """Ctrl+C mid-call: the spend is unknown, so the reserved worst case is persisted before re-raising."""
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(RUN, 25.0, path)
    def interrupt(_request: CompletionRequest) -> str:
        raise KeyboardInterrupt

    provider = FakeProvider(name=provider_name, agent="chatgpt", worst_case_usd=4.0).script(interrupt)
    request = _req().model_copy(update={"max_budget_usd": 8.0}) if provider_name == "claude_code" else _req()
    with pytest.raises(KeyboardInterrupt):
        metered_call(ledger, provider, request, stage="execution")
    (entry,) = Ledger.load(RUN, 25.0, path).entries
    assert entry.cost_usd == entry.worst_case_usd == 4.0
    assert entry.error.startswith("KeyboardInterrupt")
    assert ledger._available() == pytest.approx(21.0)


def test_metered_call_passes_persisted_entries(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = Ledger(RUN, 5.0, path)
    provider = FakeProvider(name="gemini", agent="gemini", cost_per_call=0.25).script("ok")
    metered_call(ledger, provider, _req("gemini-3.8-flash"), stage="ingestion", purpose="ingest")
    assert Ledger.load(RUN, 5.0, path).by_agent()["gemini"] == 0.25


class _SlowProvider:
    """Provider whose complete() blocks until released, to hold calls in flight."""

    name: ProviderName = "openai"
    agent: AgentName = "chatgpt"

    def __init__(self, worst: float, cost: float) -> None:
        self.worst = worst
        self.cost = cost
        self.started = threading.Semaphore(0)
        self.release = threading.Event()
        self.calls = 0

    def worst_case_cost(self, request: CompletionRequest, on: date | None = None) -> float:
        return self.worst

    def complete(self, request: CompletionRequest) -> CompletionResult:
        self.calls += 1
        self.started.release()
        assert self.release.wait(5)
        return CompletionResult(
            text="ok", usage=Usage(), cost_usd=self.cost, model=request.model, provider=self.name
        )


def test_concurrent_calls_cannot_jointly_exceed_cap() -> None:
    ledger = Ledger(RUN, 1.0)
    provider = _SlowProvider(worst=0.6, cost=0.5)
    outcomes: list[object] = []

    def call() -> None:
        try:
            outcomes.append(metered_call(ledger, provider, _req(), stage="crosscheck", purpose="critique"))
        except BudgetExceeded as exc:
            outcomes.append(exc)

    first = threading.Thread(target=call)
    first.start()
    assert provider.started.acquire(timeout=5)  # first call in flight, 0.6 reserved
    second = threading.Thread(target=call)
    second.start()
    second.join(5)
    provider.release.set()
    first.join(5)
    assert provider.calls == 1
    assert sum(isinstance(o, BudgetExceeded) for o in outcomes) == 1
    assert ledger.spent_usd == 0.5


def test_concurrent_calls_within_budget_all_run() -> None:
    ledger = Ledger(RUN, 1.0)
    provider = _SlowProvider(worst=0.3, cost=0.1)
    threads = [
        threading.Thread(target=metered_call, args=(ledger, provider, _req()), kwargs={"stage": "crosscheck"})
        for _ in range(3)
    ]
    for t in threads:
        t.start()
    for _ in threads:
        assert provider.started.acquire(timeout=5)
    provider.release.set()
    for t in threads:
        t.join(5)
    assert len(ledger.entries) == 3
    assert ledger.spent_usd == pytest.approx(0.3)
