"""MCP server tests: in-process ``mcp.Client(server)``, a real temp vault, scripted stage backends. No network."""

from __future__ import annotations

import os
import socket
import stat
import threading
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import anyio
import mcp
import pytest

from maf import mcp_server
from maf.config import Settings
from maf.handoff import HandoffKind, HandoffMeta, build_handoff
from maf.mcp_server import RunManager, build_server, check_loopback, serve
from maf.pipeline import Pipeline
from maf.stages.base import NoteOut, StageContext, StageOutput
from maf.types import STAGE_ORDER, RunStatus, StageName
from maf.vault import note_name


class Backend:
    """Stage backend producing no notes; ``final`` writes 05-final and one deliverable; can block or raise.
    ``unresolved`` > 0 makes ``crosscheck`` report that many unresolved critical issues and loop back; ``unmet``
    lines are the acceptance criteria ``final`` reports as not met."""

    def __init__(self, name: StageName, final_body: str) -> None:
        self.name = name
        self.final_body = final_body
        self.gate: threading.Event | None = None
        self.entered = threading.Event()
        self.error: Exception | None = None
        self.unresolved = 0
        self.unmet: list[str] = []

    def run_stage(self, ctx: StageContext) -> StageOutput:
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10)
        if self.error is not None:
            raise self.error
        if self.name == "crosscheck":
            return StageOutput(notes=[], index_updates={"unresolved_critical": self.unresolved}, loop_back=self.unresolved > 0)
        if self.name != "final":
            return StageOutput(notes=[])
        artifact = ctx.paths.workspace / "alloc.c"
        artifact.write_text("int x;\n", encoding="utf-8")
        ctx.vault.copy_deliverable(ctx.run_id, artifact)
        meta = HandoffMeta.model_validate(
            {"run_id": ctx.run_id, "stage": HandoffKind.FINAL, "from": "claude", "to": "user",
             "created": ctx.now, "model": "fake", "cost_usd": 0.0}
        )
        return StageOutput(
            notes=[NoteOut(note_name(HandoffKind.FINAL), build_handoff(self.final_body, meta))],
            index_updates={"criteria_unmet": len(self.unmet), "unmet_criteria": list(self.unmet)},
        )


class Env:
    def __init__(self, settings: Settings, fake_providers: Any, final_body: str) -> None:
        self.backends = {s: Backend(s, final_body) for s in STAGE_ORDER}
        self.pipeline = Pipeline(settings, providers_factory=fake_providers.factory(), backends=self.backends)  # type: ignore[arg-type]
        self.manager = RunManager(self.pipeline)
        self.server = build_server(self.manager)

    def call(self, fn: Callable[[mcp.Client], Awaitable[Any]]) -> Any:
        async def main() -> Any:
            async with mcp.Client(self.server) as client:
                return await fn(client)

        return anyio.run(main)

    def tool(self, name: str, **arguments: Any) -> Any:
        return self.call(lambda c: c.call_tool(name, arguments))


@pytest.fixture
def env(settings: Settings, fake_providers: Any, sample_bodies: dict[str, str]):  # type: ignore[no-untyped-def]
    e = Env(settings, fake_providers, sample_bodies["final"])
    yield e
    for backend in e.backends.values():
        if backend.gate is not None:
            backend.gate.set()
    e.manager.shutdown(wait=True)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ok(result: Any) -> Any:
    assert not result.is_error, result.content
    return result.structured_content


def _err(result: Any) -> str:
    assert result.is_error
    return result.content[0].text


# --------------------------------------------------------------------------- tool surface


def test_tools_and_annotations(env: Env) -> None:
    tools = {t.name: t for t in env.call(lambda c: c.list_tools()).tools}
    assert set(tools) == {"start_run", "get_run_status", "get_run_result", "list_runs"}
    assert tools["start_run"].annotations.read_only_hint is False
    for name in ("get_run_status", "get_run_result", "list_runs"):
        assert tools[name].annotations.read_only_hint is True
    assert set(tools["start_run"].input_schema["properties"]) == {"brief", "files", "budget_usd", "tier"}
    assert tools["start_run"].input_schema["required"] == ["brief"]


def test_server_info_names_maf_and_its_version(env: Env) -> None:
    from maf import __version__

    info = env.call(lambda c: _async_value(c.server_info))
    assert (info.name, info.version) == ("maf", __version__)  # was "" (MCPServer's default)


async def _async_value(value: Any) -> Any:
    return value


# --------------------------------------------------------------------------- lifecycle


def test_start_run_returns_immediately_then_completes(env: Env, tmp_path: Path) -> None:
    gate = threading.Event()
    env.backends["ingestion"].gate = gate
    src = tmp_path / "inbox" / "paper.pdf"
    src.parent.mkdir()
    src.write_bytes(b"%PDF-1.7")

    started = _ok(env.tool("start_run", brief="Burn control thesis", files=[str(src)], budget_usd=4.0, tier="max"))
    run_id = started["run_id"]
    assert started["status"] == "pending"
    assert env.backends["ingestion"].entered.wait(5)
    assert env.manager.is_active(run_id)

    status = _ok(env.tool("get_run_status", run_id=run_id))
    assert status["status"] == "running" and status["stage"] == "ingestion"
    assert (status["budget_usd"], status["round"], status["error"]) == (4.0, 1, None)
    assert set(status) == {
        "run_id", "status", "stage", "round", "spent_usd", "budget_usd", "spend_by_agent", "unresolved_critical",
        "criteria_unmet", "unmet_criteria", "error", "handoffs",
    }
    pending = _ok(env.tool("get_run_result", run_id=run_id))
    assert pending["final_markdown"] is None and pending["deliverables"] == []

    gate.set()
    final_index = env.manager.wait(run_id, timeout=10)
    assert final_index is not None and final_index.status == RunStatus.COMPLETED
    assert not env.manager.is_active(run_id)

    index = env.pipeline.status(run_id)
    assert (index.tier, index.review, index.origin) == ("max", False, "mcp")
    assert index.owner is not None and index.owner.endswith(f":{os.getpid()}")
    assert (Path(index.workspace) / "inputs" / "paper.pdf").is_file()

    status = _ok(env.tool("get_run_status", run_id=run_id))
    assert status["status"] == "completed" and status["handoffs"] == ["05-final"]
    result = _ok(env.tool("get_run_result", run_id=run_id))
    assert (result["status"], result["unresolved_critical"], result["criteria_unmet"]) == ("completed", 0, 0)
    assert (result["unmet_criteria"], result["deliverables_total"]) == ([], 1)
    assert result["final_markdown"].startswith("## Summary")
    assert "## Limitations" in result["final_markdown"]
    assert result["vault_path"] == str(env.pipeline.vault.paths(run_id).root)
    assert [Path(p).name for p in result["deliverables"]] == ["alloc.c"]


def test_runs_execute_sequentially_on_one_worker(env: Env) -> None:
    gate = threading.Event()
    env.backends["strategy"].gate = gate
    first = _ok(env.tool("start_run", brief="first"))["run_id"]
    second = _ok(env.tool("start_run", brief="second"))["run_id"]
    assert env.backends["strategy"].entered.wait(5)
    assert env.pipeline.status(second).status == RunStatus.PENDING  # queued behind the first
    assert env.manager.is_active(first) and env.manager.is_active(second)
    gate.set()
    assert env.manager.wait(second, timeout=10).status == RunStatus.COMPLETED  # type: ignore[union-attr]
    assert env.pipeline.status(first).status == RunStatus.COMPLETED


def test_completed_with_issues_is_reported_by_every_read_tool(env: Env) -> None:
    env.backends["crosscheck"].unresolved = 4
    run_id = _ok(env.tool("start_run", brief="x"))["run_id"]
    assert env.manager.wait(run_id, timeout=10).status == RunStatus.COMPLETED_WITH_ISSUES  # type: ignore[union-attr]

    status = _ok(env.tool("get_run_status", run_id=run_id))
    assert (status["status"], status["unresolved_critical"], status["round"], status["error"]) == (
        "completed_with_issues", 4, 3, None
    )
    result = _ok(env.tool("get_run_result", run_id=run_id))
    assert (result["status"], result["unresolved_critical"]) == ("completed_with_issues", 4)
    assert result["final_markdown"].startswith("## Summary")  # the report is still delivered
    (row,) = _ok(env.tool("list_runs"))["result"]
    assert row["status"] == "completed_with_issues"



def test_legacy_completed_run_with_open_criticals_is_reported_with_issues(env: Env) -> None:
    """run.md from before completed_with_issues existed: ``status: completed`` with open criticals."""
    run_id = _ok(env.tool("start_run", brief="x"))["run_id"]
    assert env.manager.wait(run_id, timeout=10).status == RunStatus.COMPLETED  # type: ignore[union-attr]
    run_md = env.pipeline.vault.paths(run_id).run_md
    text = run_md.read_text(encoding="utf-8")
    run_md.write_text(text.replace("\nunresolved_critical: 0\n", "\nunresolved_critical: 20\n"), encoding="utf-8")

    status = _ok(env.tool("get_run_status", run_id=run_id))
    assert (status["status"], status["unresolved_critical"]) == ("completed_with_issues", 20)
    result = _ok(env.tool("get_run_result", run_id=run_id))
    assert (result["status"], result["unresolved_critical"]) == ("completed_with_issues", 20)
    (row,) = _ok(env.tool("list_runs"))["result"]
    assert row["status"] == "completed_with_issues"

def test_server_instructions_explain_completed_with_issues() -> None:
    assert "completed_with_issues" in mcp_server.SERVER_INSTRUCTIONS
    assert "criteria_unmet" in mcp_server.SERVER_INSTRUCTIONS and "clean-room" in mcp_server.SERVER_INSTRUCTIONS


def test_unmet_acceptance_criteria_are_reported(env: Env) -> None:
    env.backends["final"].unmet = ["AC-2 [unmet]: every ITER claim is sourced", "clean-room [unmet]: rebuild"]
    run_id = _ok(env.tool("start_run", brief="x"))["run_id"]
    assert env.manager.wait(run_id, timeout=10).status == RunStatus.COMPLETED_WITH_ISSUES  # type: ignore[union-attr]

    for tool in ("get_run_status", "get_run_result"):
        payload = _ok(env.tool(tool, run_id=run_id))
        assert (payload["status"], payload["unresolved_critical"], payload["criteria_unmet"]) == (
            "completed_with_issues", 0, 2
        )
        assert payload["unmet_criteria"] == env.backends["final"].unmet
    assert _ok(env.tool("get_run_result", run_id=run_id))["final_markdown"].startswith("## Summary")


def test_deliverables_list_is_capped(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_server, "DELIVERABLES_MAX_LISTED", 2)
    run_id = _ok(env.tool("start_run", brief="x"))["run_id"]
    env.manager.wait(run_id, timeout=10)
    deliverables = env.pipeline.vault.paths(run_id).deliverables
    for name in ("b.c", "c.c", "d.c"):
        (deliverables / name).write_text("x")
    result = _ok(env.tool("get_run_result", run_id=run_id))
    assert [Path(p).name for p in result["deliverables"]] == ["alloc.c", "b.c"]
    assert result["deliverables_total"] == 4


def test_failed_run_reports_error(env: Env) -> None:
    env.backends["execution"].error = RuntimeError("compiler missing")
    run_id = _ok(env.tool("start_run", brief="x"))["run_id"]
    env.manager.wait(run_id, timeout=10)
    status = _ok(env.tool("get_run_status", run_id=run_id))
    assert status["status"] == "failed"
    assert status["error"] == "execution: RuntimeError: compiler missing"
    assert _ok(env.tool("get_run_result", run_id=run_id))["final_markdown"] is None


def test_final_markdown_is_truncated(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_server, "RESULT_MAX_CHARS", 40)
    run_id = _ok(env.tool("start_run", brief="x"))["run_id"]
    env.manager.wait(run_id, timeout=10)
    text = _ok(env.tool("get_run_result", run_id=run_id))["final_markdown"]
    head, marker = text.split("\n\n[... truncated ", 1)
    assert len(head) == 40
    assert "05-final.md" in marker


def test_list_runs_newest_first_with_limit(env: Env) -> None:
    ids = []
    for brief in ("alpha", "beta", "gamma"):
        run_id = _ok(env.tool("start_run", brief=brief))["run_id"]
        env.manager.wait(run_id, timeout=10)
        ids.append(run_id)
    rows = _ok(env.tool("list_runs", limit=2))["result"]
    assert len(rows) == 2
    assert set(rows[0]) == {"run_id", "status", "stage", "spent_usd", "created"}
    assert {r["run_id"] for r in _ok(env.tool("list_runs"))["result"]} == set(ids)
    assert len(_ok(env.tool("list_runs", limit=0))["result"]) == 1  # clamped to >= 1


# --------------------------------------------------------------------------- input validation


def test_start_run_rejects_relative_and_missing_files(env: Env, tmp_path: Path) -> None:
    assert "absolute" in _err(env.tool("start_run", brief="x", files=["rel/path.pdf"]))
    missing = str(tmp_path / "inbox" / "nope.pdf")
    assert _err(env.tool("start_run", brief="x", files=[missing])).endswith(f": not an allowed inbox file: {missing}")
    assert env.pipeline.list_runs() == []


def test_file_refusals_do_not_reveal_the_filesystem(env: Env, tmp_path: Path) -> None:
    """Existing and missing files outside the inbox, and a symlink out of it, get the same refusal, which echoes only
    the caller's string: no probing of which files exist, no symlink targets."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    secret = tmp_path / "secret.key"
    secret.write_text("x")
    (inbox / "link.txt").symlink_to(secret)
    for given in (str(secret), str(tmp_path / "absent.key"), str(inbox / "link.txt"), "/etc/hostname", "/proc/self/environ",
                  str(inbox / ".." / "secret.key")):
        message = _err(env.tool("start_run", brief="x", files=[given]))
        assert message.endswith(f": not an allowed inbox file: {given}"), message
        assert str(secret) not in message.replace(given, "")
    assert env.pipeline.list_runs() == []


def test_mcp_budget_ceiling_also_caps_the_default_budget(env: Env) -> None:
    env.pipeline.settings = env.pipeline.settings.model_copy(update={"budget_usd": 25.0, "mcp_max_budget_usd": 5.0})
    assert "exceeds the server ceiling $5.00" in _err(env.tool("start_run", brief="x", budget_usd=5.01))
    omitted = _ok(env.tool("start_run", brief="x"))["run_id"]
    explicit = _ok(env.tool("start_run", brief="y", budget_usd=2.5))["run_id"]
    for run_id in (omitted, explicit):
        env.manager.wait(run_id, timeout=10)
    assert env.pipeline.status(omitted).budget_usd == 5.0  # not the $25 CLI default
    assert env.pipeline.status(explicit).budget_usd == 2.5
    env.pipeline.settings = env.pipeline.settings.model_copy(update={"mcp_max_budget_usd": None})
    default = _ok(env.tool("start_run", brief="z"))["run_id"]
    env.manager.wait(default, timeout=10)
    assert env.pipeline.status(default).budget_usd == 25.0


def test_default_mcp_ceiling_is_low_not_the_cli_budget(env: Env) -> None:
    """With no mcp_max_budget_usd in the config, an MCP run gets at most $5, not budget_usd ($25)."""
    assert env.pipeline.settings.mcp_budget_ceiling_usd == 5.0
    assert "exceeds the server ceiling $5.00" in _err(env.tool("start_run", brief="x", budget_usd=6.0))
    run_id = _ok(env.tool("start_run", brief="x"))["run_id"]
    env.manager.wait(run_id, timeout=10)
    assert env.pipeline.status(run_id).budget_usd == 5.0


def test_start_run_refuses_beyond_the_pending_limit(env: Env) -> None:
    gate = threading.Event()
    env.backends["ingestion"].gate = gate
    env.pipeline.settings = env.pipeline.settings.model_copy(update={"mcp_max_pending_runs": 2})
    first = _ok(env.tool("start_run", brief="a", budget_usd=1.0))["run_id"]
    second = _ok(env.tool("start_run", brief="b", budget_usd=1.0))["run_id"]
    message = _err(env.tool("start_run", brief="c", budget_usd=1.0))
    assert "2 MCP run(s) are already queued or running (mcp_max_pending_runs 2)" in message
    assert len(env.pipeline.list_runs()) == 2  # the refused one created nothing
    gate.set()
    for run_id in (first, second):
        env.manager.wait(run_id, timeout=10)
    assert env.manager.active_count() == 0
    env.manager.wait(_ok(env.tool("start_run", brief="d", budget_usd=1.0))["run_id"], timeout=10)


def test_start_run_refuses_beyond_the_daily_mcp_budget(env: Env) -> None:
    gate = threading.Event()
    env.backends["ingestion"].gate = gate
    env.pipeline.settings = env.pipeline.settings.model_copy(update={"mcp_daily_budget_usd": 6.0})
    first = _ok(env.tool("start_run", brief="a", budget_usd=4.0))["run_id"]
    message = _err(env.tool("start_run", brief="b", budget_usd=5.0))  # a running run counts with its whole budget
    assert "take the MCP spend of the last 24 h to $9.00, above mcp_daily_budget_usd $6.00 ($2.00 left)" in message
    gate.set()
    env.manager.wait(first, timeout=10)
    assert env.pipeline.status(first).spent_usd == 0.0  # a finished run counts with what it spent
    env.manager.wait(_ok(env.tool("start_run", brief="c", budget_usd=5.0))["run_id"], timeout=10)


def test_start_run_limits_hold_under_concurrent_calls(env: Env) -> None:
    gate = threading.Event()
    env.backends["ingestion"].gate = gate
    env.pipeline.settings = env.pipeline.settings.model_copy(update={"mcp_max_pending_runs": 1})

    async def burst(client: mcp.Client) -> list[Any]:
        results: list[Any] = []

        async def one(i: int) -> None:
            results.append(await client.call_tool("start_run", {"brief": f"r{i}", "budget_usd": 1.0}))

        async with anyio.create_task_group() as group:
            for i in range(4):
                group.start_soon(one, i)
        return results

    results = env.call(burst)
    assert sum(not r.is_error for r in results) == 1
    assert len(env.pipeline.list_runs()) == 1
    gate.set()


def test_mcp_committed_usd_counts_recent_mcp_runs_only(settings: Settings) -> None:
    from maf.vault import RunIndex

    now = datetime(2026, 9, 29, 12, 0)

    def run(age_h: float, status: RunStatus, *, origin: str | None = "mcp", budget: float = 5.0,
            spent: float = 1.0) -> RunIndex:
        created = now - timedelta(hours=age_h)
        return RunIndex(run_id=f"r{age_h}{status.value}{origin}", status=status, budget_usd=budget, spent_usd=spent,
                        created=created, updated=created, workspace="/w", brief="b", origin=origin)  # type: ignore[arg-type]

    runs = [
        run(1, RunStatus.PENDING),                 # queued: its whole budget
        run(2, RunStatus.RUNNING, spent=2.0),      # running: its whole budget
        run(3, RunStatus.COMPLETED, spent=0.75),   # finished: what it spent
        run(4, RunStatus.FAILED, spent=0.5),       # stopped: what it spent
        run(5, RunStatus.COMPLETED, origin=None),  # CLI run: not counted
        run(25, RunStatus.COMPLETED, spent=4.0),   # older than 24 h: not counted
    ]
    assert mcp_server.mcp_committed_usd(runs, now) == pytest.approx(5.0 + 5.0 + 0.75 + 0.5)


def test_results_mask_api_keys(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """A key a sandboxed command read into the report must not reach the ChatGPT conversation."""
    key = "plain-looking-secret-0123456789"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    shaped = "sk-ant-api03-" + "x" * 30
    env.backends["final"].final_body = env.backends["final"].final_body.replace(
        "## Summary\n", f"## Summary\nLeaked {key} and {shaped} here.\n", 1)
    env.backends["final"].unmet = [f"AC-1 unmet: output contains {key}"]
    run_id = _ok(env.tool("start_run", brief="x"))["run_id"]
    env.manager.wait(run_id, timeout=10)
    result = _ok(env.tool("get_run_result", run_id=run_id))
    status = _ok(env.tool("get_run_status", run_id=run_id))
    assert "Leaked [redacted] and [redacted] here." in result["final_markdown"]
    for payload in (result, status):
        text = repr(payload)
        assert key not in text and shaped not in text
    assert status["unmet_criteria"] == ["AC-1 unmet: output contains [redacted]"]
    env.backends["ingestion"].error = RuntimeError(f"boom with {key}")
    failed = _ok(env.tool("start_run", brief="y"))["run_id"]
    env.manager.wait(failed, timeout=10)
    error = _ok(env.tool("get_run_status", run_id=failed))["error"]
    assert "boom with [redacted]" in error and key not in error


def test_start_run_rejects_empty_brief_and_bad_budget(env: Env) -> None:
    assert "brief" in _err(env.tool("start_run", brief="  "))
    assert "budget" in _err(env.tool("start_run", brief="x", budget_usd=-1.0))


@pytest.mark.parametrize("budget", ["NaN", "nan", "Infinity", "-Infinity"])  # a float NaN arrives as null
def test_start_run_refuses_non_finite_budget(env: Env, budget: Any) -> None:
    """pydantic turns "NaN"/"Infinity" (plain JSON strings) into floats; NaN compares False with the ceiling, and
    the run it created sat pending forever (the ledger refuses a NaN cap, which crashed the worker job)."""
    env.pipeline.settings = env.pipeline.settings.model_copy(update={"mcp_max_budget_usd": 1.0})
    assert "finite" in _err(env.tool("start_run", brief="x", budget_usd=budget))
    assert env.pipeline.list_runs() == []


@pytest.mark.parametrize("run_id", ["2026-01-01-nope", "../../etc", "a/b", ""])
def test_unknown_or_malicious_run_ids(env: Env, run_id: str) -> None:
    for tool in ("get_run_status", "get_run_result"):
        message = _err(env.tool(tool, run_id=run_id))
        assert "unknown run_id" in message or "invalid run_id" in message


# --------------------------------------------------------------------------- RunManager and serve


def test_run_manager_survives_pipeline_crash(settings: Settings) -> None:
    class Exploding:
        def create(self, brief: str, files: list[Path], **options: Any) -> Any:
            from types import SimpleNamespace

            return SimpleNamespace(run_id="r1")

        def run(self, run_id: str) -> Any:
            raise RuntimeError("already running")

    manager = RunManager(Exploding())  # type: ignore[arg-type]
    assert manager.start("x", []) == "r1"
    assert manager.wait("r1", timeout=5) is None
    assert not manager.is_active("r1")
    assert not manager.is_active("never-started")
    with pytest.raises(KeyError):
        manager.wait("never-started")
    manager.shutdown(wait=True)


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "localhost"])
def test_check_loopback_accepts(host: str) -> None:
    check_loopback(host)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "example.com", ""])
def test_serve_refuses_non_loopback(env: Env, host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        serve(env.pipeline, host=host)


def test_serve_binds_streamable_http_on_mcp_path(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    async def fake_uvicorn(app: Any, listeners: list[socket.socket], uds: Path | None) -> None:
        seen.update(app=app, addresses=[s.getsockname()[:2] for s in listeners], uds=uds)

    monkeypatch.setattr(mcp_server, "_run_uvicorn", fake_uvicorn)
    port = _free_port()
    serve(env.pipeline, host="127.0.0.1", port=port)
    assert (seen["addresses"], seen["uds"]) == ([("127.0.0.1", port)], None)
    assert [route.path for route in seen["app"].routes] == ["/mcp"]


def test_serve_on_a_unix_socket(env: Env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "run" / "maf" / "mcp.sock"
    seen: dict[str, Any] = {}

    async def fake_uvicorn(app: Any, listeners: list[socket.socket], uds: Path | None) -> None:
        info = path.lstat()
        seen.update(family=listeners[0].family, uds=uds, mode=stat.S_IMODE(info.st_mode), sock=stat.S_ISSOCK(info.st_mode),
                    parent=stat.S_IMODE(path.parent.stat().st_mode))

    monkeypatch.setattr(mcp_server, "_run_uvicorn", fake_uvicorn)
    serve(env.pipeline, host="127.0.0.1", port=8765, uds=path)
    assert seen == {"family": socket.AF_UNIX, "uds": path, "mode": 0o600, "sock": True, "parent": 0o700}
    assert not path.exists()  # removed at shutdown


def test_bind_unix_replaces_a_stale_socket_and_refuses_a_live_one(tmp_path: Path) -> None:
    path = tmp_path / "s.sock"
    stale = mcp_server.bind_unix(path)
    stale.close()  # the file stays, nothing accepts on it
    live = mcp_server.bind_unix(path)
    try:
        with pytest.raises(OSError, match="another server is listening"):
            mcp_server.bind_unix(path)
    finally:
        live.close()
    (tmp_path / "file").write_text("x")
    with pytest.raises(ValueError, match="not a socket"):
        mcp_server.bind_unix(tmp_path / "file")
    with pytest.raises(ValueError, match="absolute"):
        mcp_server.bind_unix(Path("rel.sock"))
    with pytest.raises(ValueError, match="longer than"):
        mcp_server.bind_unix(tmp_path / ("x" * 120))


def test_second_server_fails_on_the_address_before_touching_runs(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second ``maf serve`` on the same vault must not mark the first server's queued runs failed."""
    recovered: list[str] = []
    monkeypatch.setattr(RunManager, "recover", lambda self: recovered.append("recover") or [])
    busy = mcp_server.bind_tcp("127.0.0.1", 0)
    port = busy[0].getsockname()[1]
    try:
        with pytest.raises(OSError):
            serve(env.pipeline, host="127.0.0.1", port=port)
    finally:
        for sock in busy:
            sock.close()
    assert recovered == []


def test_serve_stdio_runs_the_stdio_transport_and_skips_the_loopback_check(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class FakeServer:
        async def run_stdio_async(self) -> None:
            calls.append("stdio")

    async def no_http(app: Any, listeners: list[socket.socket], uds: Path | None) -> None:
        raise AssertionError("HTTP must not start in stdio mode")

    monkeypatch.setattr(mcp_server, "build_server", lambda manager: FakeServer())
    monkeypatch.setattr(mcp_server, "_run_uvicorn", no_http)
    serve(env.pipeline, host="0.0.0.0", port=1, transport="stdio")  # host/port are irrelevant for stdio
    assert calls == ["stdio"]
    with pytest.raises(ValueError, match="transport"):
        serve(env.pipeline, transport="sse")  # type: ignore[arg-type]


def test_serve_marks_orphans_failed_after_binding_and_before_serving(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    real_bind = mcp_server.bind_tcp
    monkeypatch.setattr(mcp_server, "bind_tcp", lambda host, port: order.append("bind") or real_bind(host, port))
    monkeypatch.setattr(RunManager, "recover", lambda self: order.append("recover") or [])

    async def fake_uvicorn(app: Any, listeners: list[socket.socket], uds: Path | None) -> None:
        order.append("serve")

    monkeypatch.setattr(mcp_server, "_run_uvicorn", fake_uvicorn)
    serve(env.pipeline, port=_free_port())
    assert order == ["bind", "recover", "serve"]


# --------------------------------------------------------------------------- Host / Origin (DNS-rebinding protection)


@pytest.mark.parametrize(
    ("host", "port", "expected"),
    [
        ("127.0.0.1", 8765, ["127.0.0.1:8765", "localhost:8765"]),
        ("127.0.0.2", 9000, ["127.0.0.2:9000", "localhost:9000"]),
        ("::1", 8765, ["[::1]:8765", "localhost:8765"]),
        ("localhost", 8765, ["127.0.0.1:8765", "[::1]:8765", "localhost:8765"]),
        ("127.0.0.1", 80, ["127.0.0.1:80", "localhost:80", "127.0.0.1", "localhost"]),
    ],
)
def test_allowed_hosts_name_only_this_listener(host: str, port: int, expected: list[str]) -> None:
    assert mcp_server.allowed_hosts(host, port) == expected
    assert not any("*" in h for h in expected)


def test_transport_security_is_exact() -> None:
    security = mcp_server.transport_security("127.0.0.2", 8765, ("https://chatgpt.com",))
    assert security.enable_dns_rebinding_protection is True
    assert security.allowed_hosts == ["127.0.0.2:8765", "localhost:8765"]
    assert security.allowed_origins == ["https://chatgpt.com"]
    assert mcp_server.transport_security("127.0.0.1", 8765).allowed_origins == []


_INITIALIZE = {
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}},
}


def _post_initialize(env: Env, headers: dict[str, str], origins: tuple[str, ...] = ()) -> int:
    """POST initialize to the exact Starlette app ``serve`` would run on 127.0.0.1:8765 and return the status."""
    from starlette.testclient import TestClient

    app = mcp_server.build_http_app(env.server, "127.0.0.1", 8765, origins)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        response = client.post(
            "/mcp",
            json=_INITIALIZE,
            headers={"Accept": "application/json, text/event-stream", **headers},
        )
        return response.status_code


@pytest.mark.parametrize(
    ("headers", "status"),
    [
        ({}, 200),  # what tunnel-client sends: Host 127.0.0.1:<port>, no Origin
        ({"Host": "localhost:8765"}, 200),
        ({"Host": "127.0.0.1:9999"}, 421),  # another port
        ({"Host": "evil.example:8765"}, 421),  # DNS rebinding
        ({"Host": "127.0.0.1"}, 421),
        ({"Origin": "https://chatgpt.com"}, 403),  # not allowed unless configured
        ({"Origin": "http://localhost:6274"}, 403),  # a local web page (mcp's own default would allow it)
        ({"Origin": "null"}, 403),
    ],
)
def test_http_app_host_and_origin_policy(env: Env, headers: dict[str, str], status: int) -> None:
    assert _post_initialize(env, headers) == status


def test_configured_origin_is_accepted_and_only_that_one(env: Env) -> None:
    origins = ("https://chatgpt.com",)
    assert _post_initialize(env, {"Origin": "https://chatgpt.com"}, origins) == 200
    assert _post_initialize(env, {"Origin": "https://chatgpt.com.evil.example"}, origins) == 403
    assert _post_initialize(env, {"Origin": "http://chatgpt.com"}, origins) == 403


def test_serve_passes_the_configured_origins(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, ...]] = []
    real = mcp_server.build_http_app

    def spy(server: Any, host: str, port: int, origins: tuple[str, ...] = ()) -> Any:
        seen.append(tuple(origins))
        return real(server, host, port, origins)

    async def fake_uvicorn(app: Any, listeners: list[socket.socket], uds: Path | None) -> None:
        return None

    env.pipeline.settings = env.pipeline.settings.model_copy(update={"mcp_allowed_origins": ("https://chatgpt.com",)})
    monkeypatch.setattr(mcp_server, "build_http_app", spy)
    monkeypatch.setattr(mcp_server, "_run_uvicorn", fake_uvicorn)
    serve(env.pipeline, port=_free_port())
    assert seen == [("https://chatgpt.com",)]


# --------------------------------------------------------------------------- stdio, for real


def test_stdio_subprocess_serves_the_tools(tmp_path: Path) -> None:
    """``maf serve --stdio`` as a child process, driven by mcp's stdio client (what tunnel-client --mcp.command does)."""
    import sys

    from mcp import StdioServerParameters

    config = tmp_path / "config.yaml"
    config.write_text(
        f"vault_path: {tmp_path / 'vault'}\nworkspaces_path: {tmp_path / 'ws'}\nmcp_inbox: {tmp_path / 'inbox'}\n",
        encoding="utf-8",
    )
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "maf.cli", "--config", str(config), "serve", "--stdio"],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PYTHONPATH": str(Path(mcp_server.__file__).parents[1])},
    )

    async def main() -> tuple[list[str], Any]:
        with anyio.fail_after(60):
            async with mcp.Client(params, mode="legacy") as client:
                tools = sorted(t.name for t in (await client.list_tools()).tools)
                return tools, await client.call_tool("list_runs", {})

    tools, runs = anyio.run(main)
    assert tools == ["get_run_result", "get_run_status", "list_runs", "start_run"]
    assert _ok(runs) == {"result": []}
