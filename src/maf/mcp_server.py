"""MCP server for the ChatGPT app: start runs and inspect them. Streamable HTTP on 127.0.0.1.

Owner: orchestration.

mcp 2.x note: ``FastMCP`` was renamed to ``MCPServer`` (``from mcp.server.mcpserver import MCPServer``).
Verified against mcp 2.2.0: ``MCPServer(name, instructions=...)``, ``@server.tool(annotations=ToolAnnotations(...))``,
``server.run_streamable_http_async(host=, port=, streamable_http_path="/mcp")`` or
``server.streamable_http_app(host=...)`` for uvicorn. In-process tests: ``mcp.Client(server)``.
Sync tool functions run in a worker thread (``anyio.to_thread``), so file I/O here never blocks the loop.
Only ``ToolError`` messages reach the client; any other exception is reported as a generic failure.

Tools (every tool returns in well under ChatGPT's 1-minute limit):
- ``start_run(brief: str, files: list[str] | None = None, budget_usd: float | None = None,
  tier: "default" | "max" | None = None) -> {"run_id", "status"}``: write tool
  (``read_only_hint=False``). Creates the run synchronously (fast, no model calls) and queues it.
  ``files`` are absolute paths that must resolve inside ``settings.mcp_inbox``; ``budget_usd`` may not exceed
  ``settings.mcp_budget_ceiling_usd``. ``review`` is always False from MCP (no gate UI).
- ``get_run_status(run_id) -> {"run_id", "status", "stage", "round", "spent_usd", "budget_usd",
  "spend_by_agent", "unresolved_critical", "error", "handoffs"}``: read-only.
- ``get_run_result(run_id) -> {"run_id", "status", "unresolved_critical", "final_markdown" | None,
  "deliverables": [paths], "vault_path"}``. ``final_markdown`` is the 05-final.md body (for ``completed`` and
  ``completed_with_issues`` runs), truncated to ``RESULT_MAX_CHARS`` with a marker.
- ``list_runs(limit: int = 20) -> [{"run_id", "status", "stage", "spent_usd", "created"}]``: read-only.

``status`` is a ``RunStatus`` value. ``completed_with_issues`` means final ran only because the cross-check loop
cap was hit, with ``unresolved_critical`` critical issues still open (the final report lists them).
"""

from __future__ import annotations

import functools
import ipaddress
import logging
import re
import signal
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from maf.handoff import HandoffKind, render_body
from maf.pipeline import Pipeline
from maf.vault import RunIndex, note_name

log = logging.getLogger(__name__)

RESULT_MAX_CHARS = 60_000
LIST_MAX_LIMIT = 200
MCP_PATH = "/mcp"

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

SERVER_INSTRUCTIONS = (
    "MultiAgent (maf) runs a five-stage pipeline (ingestion, strategy, execution, cross-check, final) "
    "across ChatGPT, Gemini and Claude, writing every handoff to an Obsidian vault. start_run returns a "
    "run_id immediately and the run continues in the background, often for many minutes; poll "
    "get_run_status, then call get_run_result once the status is 'completed' or 'completed_with_issues'. "
    "'completed_with_issues' means the cross-check loop cap was reached with unresolved critical issues "
    "(count in unresolved_critical): tell the user the result is not verified and point to its Limitations. "
    "'failed' and 'budget_exceeded' carry the reason in error."
)


class RunManager:
    """Background execution of pipeline runs. One worker by default: runs execute sequentially so
    budgets and the local toolchain are not contended."""

    def __init__(self, pipeline: Pipeline, *, max_workers: int = 1) -> None:
        self.pipeline = pipeline
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="maf-run")
        self._futures: dict[str, Future[Any]] = {}
        self._lock = threading.Lock()

    def start(self, brief: str, files: list[Path], **options: Any) -> str:
        """``pipeline.create`` synchronously, then submit ``pipeline.run`` and return the run_id immediately."""
        index = self.pipeline.create(brief, files, **options)
        with self._lock:
            future = self._executor.submit(self._execute, index.run_id)
            self._futures[index.run_id] = future
        return index.run_id

    def _execute(self, run_id: str) -> RunIndex | None:
        try:
            return self.pipeline.run(run_id)
        except Exception:  # noqa: BLE001 - a worker must never die silently
            log.exception("background run %s crashed", run_id)
            return None

    def is_active(self, run_id: str) -> bool:
        """True while ``run_id`` is queued or executing in this manager."""
        with self._lock:
            future = self._futures.get(run_id)
        return future is not None and not future.done()

    def wait(self, run_id: str, timeout: float | None = None) -> RunIndex | None:
        """Block until a submitted run finishes (tests and shutdown paths)."""
        with self._lock:
            future = self._futures.get(run_id)
        if future is None:
            raise KeyError(run_id)
        return future.result(timeout=timeout)

    def recover(self) -> list[str]:
        """Mark runs orphaned by a previous server (or CLI) process FAILED, so clients do not poll them forever."""
        failed = self.pipeline.fail_orphans()
        for run_id in failed:
            log.warning("run %s was left pending/running by an earlier process; marked failed", run_id)
        return failed

    def shutdown(self, wait: bool = False) -> None:
        """Stop accepting work. The in-flight run stops at its next stage boundary (marked FAILED, interrupted);
        queued runs that never started stay PENDING. ``maf resume`` continues either."""
        if not wait:
            self.pipeline.request_stop()
        self._executor.shutdown(wait=wait, cancel_futures=not wait)


def _tool_error(message: str) -> Exception:
    from mcp.server.mcpserver.exceptions import ToolError

    return ToolError(message)


def _check_run_id(run_id: str) -> str:
    """Remote input becomes a path component, so only plain run-id characters are accepted."""
    if not _RUN_ID_RE.fullmatch(run_id) or ".." in run_id:
        raise _tool_error(f"invalid run_id: {run_id!r}")
    return run_id


def _read_index(manager: RunManager, run_id: str) -> RunIndex:
    try:
        return manager.pipeline.status(_check_run_id(run_id))
    except FileNotFoundError:
        raise _tool_error(f"unknown run_id: {run_id}") from None


def _status_payload(index: RunIndex) -> dict[str, Any]:
    return {
        "run_id": index.run_id,
        "status": index.status.value,
        "stage": index.stage,
        "round": index.round,
        "spent_usd": round(index.spent_usd, 4),
        "budget_usd": index.budget_usd,
        "spend_by_agent": {agent: round(usd, 4) for agent, usd in index.spend_by_agent.items()},
        "unresolved_critical": index.unresolved_critical,
        "error": index.error,
        "handoffs": list(index.handoffs),
    }


def _truncate(text: str, limit: int, where: Path) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n\n[... truncated {omitted} characters; full note at {where}]"


def _result_payload(manager: RunManager, index: RunIndex) -> dict[str, Any]:
    vault = manager.pipeline.vault
    paths = vault.paths(index.run_id)
    final_name = note_name(HandoffKind.FINAL)
    final_markdown: str | None = None
    if index.status.finished and vault.has_note(index.run_id, final_name):
        handoff = vault.read_handoff(index.run_id, final_name)
        body = render_body(handoff.sections)
        if handoff.title:
            body = f"# {handoff.title}\n\n{body}"
        final_markdown = _truncate(body, RESULT_MAX_CHARS, paths.note(final_name))
    deliverables = (
        sorted(str(p) for p in paths.deliverables.rglob("*") if p.is_file()) if paths.deliverables.is_dir() else []
    )
    return {
        "run_id": index.run_id,
        "status": index.status.value,
        "unresolved_critical": index.unresolved_critical,
        "final_markdown": final_markdown,
        "deliverables": deliverables,
        "vault_path": str(paths.root),
    }


def build_server(manager: RunManager) -> Any:
    """Build the ``mcp.server.mcpserver.MCPServer`` with the four tools above bound to ``manager``.
    Return type is ``MCPServer`` (typed ``Any`` here to keep the import lazy)."""
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    server = MCPServer("maf", instructions=SERVER_INSTRUCTIONS)
    read_only = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)

    @server.tool(
        annotations=ToolAnnotations(
            title="Start a MultiAgent run",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=False,
            open_world_hint=True,
        )
    )
    def start_run(
        brief: str,
        files: list[str] | None = None,
        budget_usd: float | None = None,
        tier: Literal["default", "max"] | None = None,
    ) -> dict[str, Any]:
        """Start a pipeline run for `brief` and return its run_id immediately; the run continues in the
        background. `files` are absolute paths of files inside the server's inbox directory, `budget_usd` caps
        spend for this run (default and ceiling from config), and `tier` picks the model tier."""
        settings = manager.pipeline.settings
        paths: list[Path] = []
        for name in files or []:
            path = Path(name)
            if not path.is_absolute():
                raise _tool_error(f"file paths must be absolute: {name}")
            paths.append(path)
        if paths and settings.mcp_inbox is None:
            raise _tool_error("input files are disabled for MCP runs (no mcp_inbox configured)")
        options: dict[str, Any] = {"review": False, "input_root": settings.mcp_inbox}
        if budget_usd is not None:
            ceiling = settings.mcp_budget_ceiling_usd
            if budget_usd > ceiling:
                raise _tool_error(f"budget_usd {budget_usd} exceeds the server ceiling ${ceiling:.2f}")
            options["budget_usd"] = budget_usd
        if tier is not None:
            options["tier"] = tier
        try:
            run_id = manager.start(brief, paths, **options)
        except (FileNotFoundError, ValueError) as exc:
            raise _tool_error(str(exc)) from None
        return {"run_id": run_id, "status": manager.pipeline.status(run_id).status.value}

    @server.tool(annotations=read_only)
    def get_run_status(run_id: str) -> dict[str, Any]:
        """Current status, stage, round, spend and handoff notes of a run."""
        return _status_payload(_read_index(manager, run_id))

    @server.tool(annotations=read_only)
    def get_run_result(run_id: str) -> dict[str, Any]:
        """The final report (05-final body, once completed or completed_with_issues), the unresolved critical
        count, deliverable paths and the vault folder of a run."""
        return _result_payload(manager, _read_index(manager, run_id))

    @server.tool(annotations=read_only)
    def list_runs(limit: int = 20) -> list[dict[str, Any]]:
        """Recent runs, newest first."""
        limit = max(1, min(limit, LIST_MAX_LIMIT))
        return [
            {
                "run_id": r.run_id,
                "status": r.status.value,
                "stage": r.stage,
                "spent_usd": round(r.spent_usd, 4),
                "created": r.created.isoformat(),
            }
            for r in manager.pipeline.list_runs()[:limit]
        ]

    return server


def check_loopback(host: str) -> None:
    """``ValueError`` unless ``host`` is a loopback address or ``localhost``."""
    if host == "localhost":
        return
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise ValueError(f"refusing to bind {host!r}: only loopback hosts are allowed (use the Secure MCP Tunnel)")


def serve(pipeline: Pipeline, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Blocking: run the streamable-HTTP server at ``http://{host}:{port}/mcp``. Refuses non-loopback
    hosts (``ValueError``); the Secure MCP Tunnel is the only external path."""
    import anyio

    check_loopback(host)
    manager = RunManager(pipeline)
    manager.recover()
    server = build_server(manager)
    log.info("maf MCP server on http://%s:%d%s", host, port, MCP_PATH)
    previous = signal.signal(signal.SIGTERM, _raise_interrupt)  # systemd stop: same clean path as Ctrl+C
    try:
        anyio.run(functools.partial(server.run_streamable_http_async, host=host, port=port, streamable_http_path=MCP_PATH))
    finally:
        signal.signal(signal.SIGTERM, previous)
        log.info("shutting down: the in-flight run (if any) stops after its current stage")
        manager.shutdown(wait=False)


def _raise_interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt(f"signal {signum}")
