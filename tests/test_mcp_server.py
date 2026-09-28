"""MCP server tests: in-process ``mcp.Client(server)``, a real temp vault, scripted stage backends. No network."""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
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
    """Stage backend producing no notes; ``final`` writes 05-final and one deliverable; can block or raise."""

    def __init__(self, name: StageName, final_body: str) -> None:
        self.name = name
        self.final_body = final_body
        self.gate: threading.Event | None = None
        self.entered = threading.Event()
        self.error: Exception | None = None

    def run_stage(self, ctx: StageContext) -> StageOutput:
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(10)
        if self.error is not None:
            raise self.error
        if self.name != "final":
            return StageOutput(notes=[])
        artifact = ctx.paths.workspace / "alloc.c"
        artifact.write_text("int x;\n", encoding="utf-8")
        ctx.vault.copy_deliverable(ctx.run_id, artifact)
        meta = HandoffMeta.model_validate(
            {"run_id": ctx.run_id, "stage": HandoffKind.FINAL, "from": "claude", "to": "user",
             "created": ctx.now, "model": "fake", "cost_usd": 0.0}
        )
        return StageOutput(notes=[NoteOut(note_name(HandoffKind.FINAL), build_handoff(self.final_body, meta))])


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


# --------------------------------------------------------------------------- lifecycle


def test_start_run_returns_immediately_then_completes(env: Env, tmp_path: Path) -> None:
    gate = threading.Event()
    env.backends["ingestion"].gate = gate
    src = tmp_path / "inbox" / "paper.pdf"
    src.parent.mkdir()
    src.write_bytes(b"%PDF-1.7")

    started = _ok(env.tool("start_run", brief="Burn control thesis", files=[str(src)], budget_usd=9.0, tier="max"))
    run_id = started["run_id"]
    assert started["status"] == "pending"
    assert env.backends["ingestion"].entered.wait(5)
    assert env.manager.is_active(run_id)

    status = _ok(env.tool("get_run_status", run_id=run_id))
    assert status["status"] == "running" and status["stage"] == "ingestion"
    assert (status["budget_usd"], status["round"], status["error"]) == (9.0, 1, None)
    assert set(status) == {"run_id", "status", "stage", "round", "spent_usd", "budget_usd", "spend_by_agent", "error", "handoffs"}
    pending = _ok(env.tool("get_run_result", run_id=run_id))
    assert pending["final_markdown"] is None and pending["deliverables"] == []

    gate.set()
    final_index = env.manager.wait(run_id, timeout=10)
    assert final_index is not None and final_index.status == RunStatus.COMPLETED
    assert not env.manager.is_active(run_id)

    index = env.pipeline.status(run_id)
    assert (index.tier, index.review) == ("max", False)
    assert (Path(index.workspace) / "inputs" / "paper.pdf").is_file()

    status = _ok(env.tool("get_run_status", run_id=run_id))
    assert status["status"] == "completed" and status["handoffs"] == ["05-final"]
    result = _ok(env.tool("get_run_result", run_id=run_id))
    assert result["status"] == "completed"
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
    assert "not found" in _err(env.tool("start_run", brief="x", files=[str(tmp_path / "nope.pdf")]))
    assert env.pipeline.list_runs() == []


def test_start_run_rejects_empty_brief_and_bad_budget(env: Env) -> None:
    assert "brief" in _err(env.tool("start_run", brief="  "))
    assert "budget" in _err(env.tool("start_run", brief="x", budget_usd=-1.0))


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

    class FakeServer:
        async def run_streamable_http_async(self, **kw: Any) -> None:
            seen.update(kw)

    monkeypatch.setattr(mcp_server, "build_server", lambda manager: FakeServer())
    serve(env.pipeline, host="127.0.0.1", port=8765)
    assert seen == {"host": "127.0.0.1", "port": 8765, "streamable_http_path": "/mcp"}
