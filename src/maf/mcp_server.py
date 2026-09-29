"""MCP server for the ChatGPT app: start runs and inspect them. Streamable HTTP on 127.0.0.1, or stdio.

Owner: orchestration.

Transports (``serve``):
- ``http`` (default): ``http://127.0.0.1:<port>/mcp``, loopback binds only, with DNS-rebinding protection pinned by
  ``transport_security``: the Host header must name this listener (``<bind address>:<port>`` or ``localhost:<port>``),
  which is exactly what tunnel-client sends (the host:port of its configured MCP URL), and an Origin header is refused
  unless it is listed in ``settings.mcp_allowed_origins`` (empty by default: tunnel-client and non-browser clients send
  none, while browsers always send one on POST).
- ``http`` on a Unix socket (``uds``; the systemd unit, ``--uds %t/maf/mcp.sock``): the same HTTP app, but the
  listener is a 0600 socket in the owner's 0700 runtime directory, so only the owner's uid can connect. A loopback TCP
  port is open to every local user and to containers on the host network, and maf has no login of its own. The
  host/port then only name the Host header (tunnel-client: ``--mcp.server-url url=http://127.0.0.1:<port>/mcp,
  unix-socket=<path>`` sends ``Host: 127.0.0.1:<port>``).
- ``stdio``: the same tools over stdin/stdout, for an MCP host that spawns maf itself (``tunnel-client --mcp.command``).
  mcp 2.2's ``stdio_server`` points fd 1 at stderr while serving, so stray output cannot corrupt the protocol.

The listening socket is bound before orphaned runs are recovered, so a second server on the same vault fails on the
taken address and touches no run.

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
  ``files`` are absolute paths that must resolve inside ``settings.mcp_inbox`` (any problem gives one generic refusal
  naming only the caller's string); ``budget_usd`` may not exceed ``settings.mcp_budget_ceiling_usd``, and an omitted
  one is ``min(settings.budget_usd, ceiling)``. Refused while ``settings.mcp_max_pending_runs`` MCP runs are queued or
  running in this server, or when the run's budget would take the MCP spend of the last 24 hours
  (``mcp_committed_usd``) above ``settings.mcp_daily_budget_usd``: a remembered ChatGPT approval then still cannot
  queue unbounded spend. ``review`` is always False from MCP (no gate UI).
- ``get_run_status(run_id) -> {"run_id", "status", "stage", "round", "spent_usd", "budget_usd",
  "spend_by_agent", "unresolved_critical", "criteria_unmet", "unmet_criteria", "error", "handoffs"}``: read-only.
- ``get_run_result(run_id) -> {"run_id", "status", "unresolved_critical", "criteria_unmet", "unmet_criteria",
  "final_markdown" | None, "deliverables": [paths], "deliverables_total", "vault_path"}``. ``final_markdown`` is the
  05-final.md body (for ``completed`` and ``completed_with_issues`` runs), truncated to ``RESULT_MAX_CHARS`` with a
  marker; ``deliverables`` lists at most ``DELIVERABLES_MAX_LISTED`` of the ``deliverables_total`` files.
- ``list_runs(limit: int = 20) -> [{"run_id", "status", "stage", "spent_usd", "created"}]``: read-only.

Text fields that carry model or tool output (``final_markdown``, ``error``, ``unmet_criteria``) pass through
``maf.redact.redact`` with the key values of maf's environment, so a key a run leaked into its report (a sandboxed
command can read more than the workspace) does not reach the ChatGPT conversation.

``status`` is a ``RunStatus`` value. ``completed_with_issues`` means final ran but the result is not verified: the
cross-check loop cap was hit with ``unresolved_critical`` critical issues still open, and/or ``criteria_unmet``
acceptance criteria (maf's own checks included: ``clean-room``, ``source-audit``, ``lint``) are not met, one line
each in ``unmet_criteria``. The final report lists both.
"""

from __future__ import annotations

import errno
import ipaddress
import logging
import math
import os
import re
import signal
import socket
import stat
import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from maf.handoff import HandoffKind, render_body
from maf.pipeline import Pipeline, process_owner
from maf.redact import known_secrets, redact
from maf.types import RunStatus
from maf.vault import RunIndex, note_name

log = logging.getLogger(__name__)

RESULT_MAX_CHARS = 60_000
DELIVERABLES_MAX_LISTED = 200
"""A code run exports its whole workspace tree; ``get_run_result`` lists this many paths and the total count."""
LIST_MAX_LIMIT = 200
MCP_PATH = "/mcp"
HTTP_SHUTDOWN_GRACE_S = 5
"""On SIGTERM, uvicorn waits at most this long for in-flight HTTP requests before cancelling them (its default waits
forever). mcp closes its own SSE streams at shutdown (with an open session maf exited within a second either way),
so this only bounds a stuck request. The in-flight run then stops at its next stage boundary (``RunManager.shutdown``)."""

LISTEN_BACKLOG = 128
MCP_SPEND_WINDOW_S = 24 * 3600
"""``mcp_daily_budget_usd`` counts the MCP runs created in this many seconds before a ``start_run``."""
UNIX_PATH_MAX = 107
"""Longest Unix socket path Linux accepts (``sun_path`` is 108 bytes with the terminating NUL)."""

Transport = Literal["http", "stdio"]

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

SERVER_INSTRUCTIONS = (
    "MultiAgent (maf) runs a five-stage pipeline (ingestion, strategy, execution, cross-check, final) "
    "across ChatGPT, Gemini and Claude, writing every handoff to an Obsidian vault. start_run returns a "
    "run_id immediately and the run continues in the background, often for many minutes; poll "
    "get_run_status, then call get_run_result once the status is 'completed' or 'completed_with_issues'. "
    "'completed_with_issues' means the result is not verified: the cross-check loop cap was reached with "
    "unresolved critical issues (count in unresolved_critical), and/or acceptance criteria were not met "
    "(count in criteria_unmet, one line each in unmet_criteria; 'clean-room' means the exported deliverables did "
    "not rebuild from scratch, 'source-audit' that a reference was not verified on the web, 'lint' that the "
    "deliverables link pipeline notes). Tell the user so and point to the final report's Acceptance and Limitations. "
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
        self.admission = threading.Lock()
        """Held by ``start_run`` from its limit checks until the run is queued, so concurrent calls cannot both
        pass a limit that only one of them fits."""

    def active_count(self) -> int:
        """Runs submitted to this manager that have not finished (queued or executing)."""
        with self._lock:
            return sum(1 for future in self._futures.values() if not future.done())

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


def mcp_committed_usd(runs: Iterable[RunIndex], now: datetime) -> float:
    """What the MCP-created runs (``origin == "mcp"``) of the ``MCP_SPEND_WINDOW_S`` before ``now`` committed: the
    budget of a queued, running or paused run (it may still spend all of it), the actual spend of any other."""
    cutoff = now.timestamp() - MCP_SPEND_WINDOW_S
    total = 0.0
    for run in runs:
        if run.origin != "mcp" or run.created.timestamp() < cutoff:
            continue
        open_ = run.status in (RunStatus.PENDING, RunStatus.RUNNING, RunStatus.AWAITING_REVIEW)
        total += max(run.budget_usd, run.spent_usd) if open_ else run.spent_usd
    return total


def _scrub(text: str | None) -> str | None:
    """``text`` with maf's API keys (and key-shaped strings) masked: tool results go into a ChatGPT conversation."""
    return None if text is None else redact(text, known_secrets())


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
        "criteria_unmet": index.criteria_unmet,
        "unmet_criteria": [_scrub(line) for line in index.unmet_criteria],
        "error": _scrub(index.error),
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
        final_markdown = _scrub(_truncate(body, RESULT_MAX_CHARS, paths.note(final_name)))
    deliverables = (
        sorted(str(p) for p in paths.deliverables.rglob("*") if p.is_file()) if paths.deliverables.is_dir() else []
    )
    return {
        "run_id": index.run_id,
        "status": index.status.value,
        "unresolved_critical": index.unresolved_critical,
        "criteria_unmet": index.criteria_unmet,
        "unmet_criteria": [_scrub(line) for line in index.unmet_criteria],
        "final_markdown": final_markdown,
        "deliverables": deliverables[:DELIVERABLES_MAX_LISTED],
        "deliverables_total": len(deliverables),
        "vault_path": str(paths.root),
    }


def build_server(manager: RunManager) -> Any:
    """Build the ``mcp.server.mcpserver.MCPServer`` with the four tools above bound to ``manager``.
    Return type is ``MCPServer`` (typed ``Any`` here to keep the import lazy)."""
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    from maf import __version__

    server = MCPServer("maf", instructions=SERVER_INSTRUCTIONS, version=__version__)
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
        spend for this run in USD (optional; default and ceiling come from the server config), and `tier` picks the
        model tier."""
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
        ceiling = settings.mcp_budget_ceiling_usd
        # pydantic parses "NaN"/"Infinity" (strings or JSON literals) into floats, and NaN > ceiling is False.
        if budget_usd is not None and not (math.isfinite(budget_usd) and budget_usd > 0):
            raise _tool_error(f"budget_usd must be a positive finite number, got {budget_usd}")
        if budget_usd is not None and budget_usd > ceiling:
            raise _tool_error(f"budget_usd {budget_usd} exceeds the server ceiling ${ceiling:.2f}")
        # Omitted: the configured default, but never above the MCP ceiling.
        budget = min(settings.budget_usd, ceiling) if budget_usd is None else budget_usd
        options["budget_usd"] = budget
        if tier is not None:
            options["tier"] = tier
        with manager.admission:
            _check_mcp_limits(manager, budget)
            try:
                run_id = manager.start(brief, paths, origin="mcp", owner=process_owner(), **options)
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
        count, the unmet acceptance criteria, deliverable paths and the vault folder of a run."""
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


def _check_mcp_limits(manager: RunManager, budget: float) -> None:
    """``ToolError`` if this server already holds ``mcp_max_pending_runs`` MCP runs, or if ``budget`` would take the
    MCP spend of the last 24 hours above ``mcp_daily_budget_usd``. Call with ``manager.admission`` held."""
    settings = manager.pipeline.settings
    active = manager.active_count()
    if active >= settings.mcp_max_pending_runs:
        raise _tool_error(
            f"refused: {active} MCP run(s) are already queued or running (mcp_max_pending_runs "
            f"{settings.mcp_max_pending_runs}); start this one after one of them has finished"
        )
    cap = settings.mcp_daily_budget_usd
    committed = mcp_committed_usd(manager.pipeline.list_runs(), manager.pipeline.clock())
    if committed + budget > cap + 1e-9:
        raise _tool_error(
            f"refused: a ${budget:.2f} run would take the MCP spend of the last 24 h to ${committed + budget:.2f}, "
            f"above mcp_daily_budget_usd ${cap:.2f} (${max(0.0, cap - committed):.2f} left); a smaller budget_usd, "
            "or the CLI, can still run"
        )


def allowed_hosts(host: str, port: int) -> list[str]:
    """Host header values the listener at ``host:port`` accepts: the bind address and ``localhost``, with this port
    (``localhost`` binds also accept ``127.0.0.1`` and ``[::1]``; on port 80 the bare names too, since HTTP clients
    omit the default port). tunnel-client sends the host:port of its configured MCP URL (Go's net/http ignores any
    forwarded ``Host``), so ``http://127.0.0.1:<port>/mcp`` and ``http://localhost:<port>/mcp`` pass, while a
    DNS-rebinding page's Host (its own domain) and other ports do not. ``host`` must pass ``check_loopback``."""
    names = {"localhost"}
    if host == "localhost":
        names |= {"127.0.0.1", "[::1]"}
    else:
        address = ipaddress.ip_address(host)
        names.add(f"[{address.compressed}]" if address.version == 6 else address.compressed)
    hosts = [f"{name}:{port}" for name in sorted(names)]
    if port == 80:
        hosts += sorted(names)
    return hosts


def transport_security(host: str, port: int, allowed_origins: Sequence[str] = ()) -> Any:
    """``mcp.server.transport_security.TransportSecuritySettings`` for the loopback listener: DNS-rebinding
    protection on, Host limited to ``allowed_hosts(host, port)``, Origin absent or exactly one of ``allowed_origins``
    (``settings.mcp_allowed_origins``; no wildcards). Replaces mcp's loopback default, which accepts any port and
    every ``http://localhost:*`` page, and which it skips entirely for other loopback addresses such as 127.0.0.2."""
    from mcp.server.transport_security import TransportSecuritySettings

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts(host, port),
        allowed_origins=list(allowed_origins),
    )


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


def build_http_app(server: Any, host: str, port: int, allowed_origins: Sequence[str] = ()) -> Any:
    """The Starlette app ``serve`` runs: streamable HTTP at ``MCP_PATH`` with ``transport_security``."""
    return server.streamable_http_app(
        streamable_http_path=MCP_PATH,
        transport_security=transport_security(host, port, allowed_origins),
        host=host,
    )


def bind_tcp(host: str, port: int) -> list[socket.socket]:
    """Listening sockets on every address ``host`` resolves to (``localhost``: 127.0.0.1 and ::1), port ``port``.
    ``OSError`` (such as EADDRINUSE) if one cannot be bound; nothing stays open then."""
    sockets: list[socket.socket] = []
    seen: set[tuple[int, str]] = set()
    try:
        for family, kind, proto, _name, address in socket.getaddrinfo(
            host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
        ):
            if (family, address[0]) in seen:
                continue
            seen.add((family, address[0]))
            sock = socket.socket(family, kind, proto)
            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind(address)
            sock.listen(LISTEN_BACKLOG)
    except BaseException:
        for sock in sockets:
            sock.close()
        raise
    return sockets


def bind_unix(path: Path) -> socket.socket:
    """A listening Unix socket at ``path``, mode 0600 from the start (created under umask 0177). The parent is created
    0700 if missing. A stale socket file (nothing accepts on it) is replaced; a live one raises ``OSError``
    (EADDRINUSE), anything else at ``path`` ``ValueError``, as does a relative or too long path."""
    if not path.is_absolute():
        raise ValueError(f"the Unix socket path must be absolute, got {str(path)!r}")
    if len(os.fsencode(str(path))) > UNIX_PATH_MAX:
        raise ValueError(f"the Unix socket path {path} is longer than {UNIX_PATH_MAX} bytes")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        mode = None
    if mode is not None:
        if not stat.S_ISSOCK(mode):
            raise ValueError(f"{path} exists and is not a socket; refusing to replace it")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            try:
                probe.connect(str(path))
            except (ConnectionRefusedError, FileNotFoundError):
                path.unlink(missing_ok=True)  # stale: left behind by a server that died
            else:
                raise OSError(errno.EADDRINUSE, f"another server is listening on {path}")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    previous = os.umask(0o177)
    try:
        sock.bind(str(path))
        os.chmod(path, 0o600)
        sock.listen(LISTEN_BACKLOG)
    except BaseException:
        sock.close()
        raise
    finally:
        os.umask(previous)
    return sock


def _remove_socket(path: Path, inode: int) -> None:
    """Unlink the socket file ``serve`` created, unless something else has replaced it since."""
    try:
        if path.lstat().st_ino == inode:
            path.unlink()
    except OSError:
        pass


async def _run_uvicorn(app: Any, listeners: list[socket.socket], uds: Path | None = None) -> None:
    """uvicorn on the already-bound ``listeners`` (``uds`` only names the socket in its startup log line)."""
    import uvicorn

    host, port = ("127.0.0.1", 0) if uds is not None else listeners[0].getsockname()[:2]
    config = uvicorn.Config(app, host=host, port=port, uds=None if uds is None else str(uds), log_level="info",
                            timeout_graceful_shutdown=HTTP_SHUTDOWN_GRACE_S)
    await uvicorn.Server(config).serve(sockets=listeners)


def serve(
    pipeline: Pipeline,
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    transport: Transport = "http",
    uds: Path | None = None,
) -> None:
    """Blocking: serve the tools over ``transport``. ``http`` listens at ``http://{host}:{port}/mcp`` and refuses
    non-loopback hosts (``ValueError``), since the Secure MCP Tunnel is the only external path; with ``uds`` it
    listens on that Unix socket (``bind_unix``) instead, and ``host``/``port`` only set the accepted Host header.
    ``stdio`` ignores ``host``/``port``/``uds``. The listener is bound first (``OSError`` if the address is taken),
    then runs orphaned by an earlier process are marked failed. SIGTERM (systemd stop) takes the same path as Ctrl+C:
    stop serving, then let the in-flight run stop at its next stage boundary."""
    import anyio

    if transport not in ("http", "stdio"):
        raise ValueError(f"unknown transport {transport!r}")
    listeners: list[socket.socket] = []
    socket_inode: int | None = None
    if transport == "http":
        check_loopback(host)
        if uds is not None:
            listeners = [bind_unix(uds)]
            socket_inode = uds.lstat().st_ino
        else:
            listeners = bind_tcp(host, port)
    try:
        manager = RunManager(pipeline)
        manager.recover()
        server = build_server(manager)
        previous = signal.signal(signal.SIGTERM, _raise_interrupt)
        try:
            if transport == "stdio":
                log.info("maf MCP server on stdio")
                anyio.run(server.run_stdio_async)
            else:
                app = build_http_app(server, host, port, pipeline.settings.mcp_allowed_origins)
                if uds is not None:
                    log.info("maf MCP server on unix socket %s, path %s, Host %s:%d", uds, MCP_PATH, host, port)
                else:
                    log.info("maf MCP server on http://%s:%d%s", host, port, MCP_PATH)
                anyio.run(_run_uvicorn, app, listeners, uds)
        finally:
            signal.signal(signal.SIGTERM, previous)
            log.info("shutting down: the in-flight run (if any) stops after its current stage")
            manager.shutdown(wait=False)
    finally:
        for sock in listeners:
            sock.close()
        if uds is not None and socket_inode is not None:
            _remove_socket(uds, socket_inode)


def _raise_interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt(f"signal {signum}")
