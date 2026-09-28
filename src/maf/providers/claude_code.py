"""Claude Code headless (``claude -p``) as a provider, for coding and execution in a sandboxed workspace.

Owner: providers.

CLI facts, verified from ``claude --help`` (v2.1.284) and strings in the binary:

- The prompt goes on **stdin** (avoids the 128 KiB per-argument limit). ``-p/--print``, ``--output-format json``.
- ``--model <id>``, ``--effort <level>``, ``--max-budget-usd <amount>`` (print mode only),
  ``--allowedTools <tools...>``, ``--disallowedTools <tools...>``, ``--tools <list>`` (built-in set),
  ``--permission-mode dontAsk`` (anything not allowed is denied, no prompts),
  ``--permission-prompts none``, ``--no-session-persistence``, ``--json-schema <schema>``,
  ``--append-system-prompt <text>``, ``--settings <json>``, ``--add-dir`` (NOT used: no dirs outside cwd),
  ``--bare`` (skip hooks/CLAUDE.md/plugins; auth strictly ``ANTHROPIC_API_KEY``).
- Result JSON (single object): ``type="result"``, ``subtype`` in {``success``, ``error_max_turns``,
  ``error_during_execution``, ``error_max_budget_usd``}, ``is_error``, ``result`` (final text),
  ``session_id``, ``total_cost_usd``, ``duration_ms``, ``duration_api_ms``, ``num_turns``,
  ``usage``, ``modelUsage``, ``permission_denials``, ``structured_output`` (with ``--json-schema``).

Sandbox: the subprocess cwd is ``workspaces/<run_id>/``. Writes outside it are prevented by
(a) no ``--add-dir``, (b) ``--permission-mode dontAsk`` with an explicit allowlist, and
(c) Claude Code's OS sandbox for Bash via ``--settings``:
``{"sandbox": {"enabled": true, "failIfUnavailable": true, "allowUnsandboxedCommands": false,
"autoAllowBashIfSandboxed": false, "filesystem": {"allowWrite": ["<workspace>"]},
"network": {"allowedDomains": []}}}`` (bubblewrap is installed at /usr/bin/bwrap). Web tools are disallowed.

File tools are allowed only with a workspace scope (``Read(./**)``, ``Edit(./**)``, ``Write(./**)``;
``check_scoped_tools`` rejects bare rules), and ``SENSITIVE_READ_PATHS`` are denied to both the file tools
(``permissions.deny``) and sandboxed Bash (``sandbox.filesystem.denyRead``).

The subprocess env is built from an allowlist (``PATH``, ``HOME``, locale, ``ANTHROPIC_API_KEY``...), so
other credentials and ``CLAUDECODE`` (nested-session detection) never reach it, and
``CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1`` asks the CLI to strip its API key from Bash children.
``TMPDIR`` and ``MPLCONFIGDIR`` point into ``<workspace>/.maf/`` because the sandbox only permits
writes inside the workspace.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from maf.providers.base import (
    CompletionRequest,
    CompletionResult,
    ProviderError,
    StructuredOutputError,
    parse_json_output,
    token_worst_case,
)
from maf.providers.claude_provider import DEFAULT_EFFORT, usage_from_message
from maf.types import AgentName, ProviderName

PROVIDER: ProviderName = "claude_code"
DISALLOWED_TOOLS: tuple[str, ...] = ("WebFetch", "WebSearch")
STDERR_TAIL_CHARS = 2_000

PATH_SCOPED_TOOLS = frozenset({"Read", "Edit", "Write", "MultiEdit", "NotebookEdit"})
"""File tools whose allow rules must carry a path scope: a bare rule matches every path on the machine."""

ENV_ALLOWLIST: tuple[str, ...] = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LANGUAGE", "TERM", "TZ")
"""Inherited variables passed to the CLI (plus ``LC_*`` and ``ANTHROPIC_API_KEY``). Everything else, such as
cloud or GitHub tokens, is dropped, so Bash children cannot copy it into workspace artifacts."""

SENSITIVE_READ_PATHS: tuple[str, ...] = (
    "~/.ssh",
    "~/.gnupg",
    "~/.aws",
    "~/.azure",
    "~/.config",
    "~/.docker",
    "~/.kube",
    "~/.netrc",
    "~/.git-credentials",
    "~/.claude",
    "~/.local/share/keyrings",
)
"""Denied to both the file tools (``permissions.deny``) and sandboxed Bash (``sandbox.filesystem.denyRead``)."""

DEFAULT_TURN_CONTEXT_TOKENS = 200_000
DEFAULT_TURN_OUTPUT_TOKENS = 64_000


class ClaudeCodeOutput(BaseModel):
    """Parsed ``--output-format json`` result. Unknown keys are kept (``extra="allow"``)."""

    model_config = ConfigDict(extra="allow")

    type: str = "result"
    subtype: str = ""
    is_error: bool = False
    result: str = ""
    session_id: str = ""
    total_cost_usd: float = 0.0
    duration_ms: int = 0
    num_turns: int = 0
    usage: dict[str, Any] = Field(default_factory=dict)
    structured_output: Any = None
    permission_denials: list[Any] = Field(default_factory=list)


class CompletedProcess(BaseModel):
    """What the injectable runner returns. Tests substitute a fake runner (no subprocess)."""

    returncode: int
    stdout: str
    stderr: str = ""


Runner = Callable[[Sequence[str], str, Path, dict[str, str], float], CompletedProcess]
"""``runner(argv, stdin_text, cwd, env, timeout_s) -> CompletedProcess``."""


class ClaudeCodeTimeout(ProviderError):
    """The CLI exceeded ``timeout_s`` and was killed. Spend is unknown, so ``complete`` charges the full budget."""


def subprocess_runner(
    argv: Sequence[str], stdin_text: str, cwd: Path, env: dict[str, str], timeout_s: float
) -> CompletedProcess:
    """Default runner: ``subprocess.run`` with text I/O and timeout. Timeout raises ``ProviderError``.

    The CLI runs in its own process group so a timeout also kills the build/test commands it spawned."""
    try:
        proc = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
    except OSError as exc:
        raise ProviderError(f"cannot start Claude Code ({argv[0]}): {exc}", provider=PROVIDER) from exc
    try:
        stdout, stderr = proc.communicate(input=stdin_text, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        proc.communicate()
        raise ClaudeCodeTimeout(f"Claude Code timed out after {timeout_s:.0f}s", provider=PROVIDER) from None
    except BaseException:
        _kill_group(proc)
        proc.communicate()
        raise
    return CompletedProcess(returncode=proc.returncode, stdout=stdout, stderr=stderr)


def _kill_group(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def sandbox_settings(workspace: Path, deny_read: Sequence[str] = SENSITIVE_READ_PATHS) -> dict[str, Any]:
    """The ``--settings`` JSON object shown in the module docstring, bound to ``workspace``.

    ``deny_read`` entries are ``~``-relative or absolute paths. They become ``Read``/``Edit`` deny rules
    (``~/x/**`` or ``//abs/x/**``) and ``sandbox.filesystem.denyRead`` entries."""
    rules = [_rule_path(p) for p in deny_read]
    return {
        "permissions": {"deny": [f"{tool}({rule})" for rule in rules for tool in ("Read", "Edit")]},
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "allowUnsandboxedCommands": False,
            "autoAllowBashIfSandboxed": False,
            "filesystem": {"allowWrite": [str(workspace.resolve())], "denyRead": list(deny_read)},
            "network": {"allowedDomains": []},
        },
    }


def _rule_path(path: str) -> str:
    """Permission-rule path form: ``~/x`` stays home-relative, ``/abs`` becomes ``//abs``; ``/**`` is appended."""
    base = path.rstrip("/")
    if base.startswith("/"):
        base = "/" + base
    return f"{base}/**"


def check_scoped_tools(tools: Sequence[str]) -> None:
    """``ValueError`` for a bare ``Read``/``Edit``/``Write``-style allow rule (it would grant every path)."""
    bare = [t for t in tools if t.strip() in PATH_SCOPED_TOOLS]
    if bare:
        raise ValueError(f"file tool rules need a path scope such as 'Edit(./**)': {', '.join(bare)}")


def format_budget(usd: float) -> str:
    """Round *down* to 1/100 cent so the CLI cap never exceeds the ledger's clamp."""
    return f"{math.floor(usd * 10_000) / 10_000:.4f}"


class ClaudeCodeProvider:
    name: ProviderName = PROVIDER
    agent: AgentName = "claude"

    def __init__(
        self,
        workspace: Path,
        *,
        executable: Path,
        allowed_tools: Sequence[str],
        timeout_s: float = 3600.0,
        runner: Runner | None = None,
        extra_env: dict[str, str] | None = None,
        path_prepend: Sequence[Path] = (),
        deny_read: Sequence[str] = SENSITIVE_READ_PATHS,
        turn_context_tokens: int = DEFAULT_TURN_CONTEXT_TOKENS,
        turn_output_tokens: int = DEFAULT_TURN_OUTPUT_TOKENS,
    ) -> None:
        """``path_prepend`` puts directories (e.g. the project venv's ``bin``) first on the CLI's ``PATH``.
        ``deny_read`` lists paths no tool may read. ``turn_*_tokens`` size one model turn, the amount the CLI
        can overshoot ``--max-budget-usd`` by (see ``turn_headroom_usd``)."""
        check_scoped_tools(allowed_tools)
        self.workspace = workspace
        self.executable = executable
        self.allowed_tools = tuple(allowed_tools)
        self.timeout_s = timeout_s
        self._runner = runner or subprocess_runner
        self._extra_env = extra_env or {}
        self._path_prepend = tuple(path_prepend)
        self._deny_read = tuple(deny_read)
        self._turn_context_tokens = turn_context_tokens
        self._turn_output_tokens = turn_output_tokens

    def build_argv(self, request: CompletionRequest) -> list[str]:
        """Pure argv construction (unit-tested). ``request.system`` goes via ``--append-system-prompt``;
        ``request.json_schema`` via ``--json-schema json.dumps(schema)``; ``max_budget_usd`` is required.

        The variadic tool lists come last: nothing positional follows them (the prompt is on stdin)."""
        budget = request.max_budget_usd
        if budget is None:
            raise ValueError("Claude Code requests need max_budget_usd (metered_call sets it)")
        if budget < 0.0001:
            raise ValueError(f"Claude Code budget too small: {budget!r}")
        argv = [
            str(self.executable),
            "-p",
            "--output-format", "json",
            "--model", request.model,
            "--effort", request.effort or DEFAULT_EFFORT,
            "--max-budget-usd", format_budget(budget),
            "--permission-mode", "dontAsk",
            "--permission-prompts", "none",
            "--no-session-persistence",
            "--bare",
            "--settings", json.dumps(sandbox_settings(self.workspace, self._deny_read), separators=(",", ":")),
        ]  # fmt: skip
        if request.system:
            argv += ["--append-system-prompt", request.system]
        if request.json_schema is not None:
            argv += ["--json-schema", json.dumps(request.json_schema, separators=(",", ":"))]
        if self.allowed_tools:
            argv += ["--allowedTools", *self.allowed_tools]
        argv += ["--disallowedTools", *DISALLOWED_TOOLS]
        return argv

    def build_stdin(self, request: CompletionRequest) -> str:
        """Concatenate ``request.messages`` into one prompt (Claude Code takes a single user turn).

        A lone user message passes through verbatim; a multi-turn history is rendered as labelled blocks."""
        if len(request.messages) == 1 and request.messages[0].role == "user":
            return request.messages[0].content
        labels = {"user": "User", "assistant": "Assistant (earlier turn)"}
        return "\n\n".join(f"## {labels[m.role]}\n\n{m.content}" for m in request.messages)

    def build_env(self) -> dict[str, str]:
        """Subprocess environment from an allowlist (``ENV_ALLOWLIST``, ``LC_*``, ``ANTHROPIC_API_KEY``), with
        ``CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1`` so the CLI strips its credentials from Bash children, plus
        workspace-local temp dirs."""
        env = {
            k: v
            for k, v in os.environ.items()
            if k in ENV_ALLOWLIST or k.startswith("LC_") or k == "ANTHROPIC_API_KEY"
        }
        env["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"] = "1"
        if self._path_prepend:
            env["PATH"] = os.pathsep.join([*(str(p) for p in self._path_prepend), env.get("PATH", "")])
        scratch = self.workspace.resolve() / ".maf"
        env["TMPDIR"] = str(scratch / "tmp")
        env["MPLCONFIGDIR"] = str(scratch / "mpl")
        env.update(self._extra_env)
        return env

    def complete(self, request: CompletionRequest) -> CompletionResult:
        """Run the CLI in ``self.workspace`` and parse the JSON.

        - ``request.max_budget_usd`` must be set (``metered_call`` does it); else ``ValueError``.
        - ``is_error`` or a non-zero exit raises ``ProviderError(cost_usd=total_cost_usd)``.
          ``subtype == "error_max_budget_usd"`` is ``retryable=False``.
        - ``cost_usd`` = ``total_cost_usd`` (authoritative; not re-priced from tokens).
        - ``text`` = ``result``; ``parsed`` = ``structured_output`` when a schema was given.
        - A timeout, a crash without a JSON result, or a runner failure charges ``worst_case_cost``
          (budget plus one turn), since the real spend is unknown.
        """
        argv = self.build_argv(request)
        env = self.build_env()
        if not env.get("ANTHROPIC_API_KEY"):
            raise ProviderError("ANTHROPIC_API_KEY is not set (required by claude --bare)", provider=PROVIDER)
        for directory in (self.workspace, Path(env["TMPDIR"]), Path(env["MPLCONFIGDIR"])):
            directory.mkdir(parents=True, exist_ok=True)

        unknown_spend = self.worst_case_cost(request)  # charged whenever the CLI's own cost report is missing
        try:
            proc = self._runner(argv, self.build_stdin(request), self.workspace, env, self.timeout_s)
        except ClaudeCodeTimeout as exc:
            raise ProviderError(str(exc), provider=PROVIDER, cost_usd=unknown_spend) from exc
        except ProviderError:
            raise  # the CLI could not be started: nothing was spent
        except Exception as exc:
            raise ProviderError(
                f"Claude Code runner failed: {type(exc).__name__}: {exc}", provider=PROVIDER, cost_usd=unknown_spend
            ) from exc

        try:
            out = self.parse_output(proc.stdout)
        except ProviderError as exc:
            detail = _tail(proc.stderr) or str(exc)
            raise ProviderError(
                f"Claude Code exited {proc.returncode} without a result: {detail}",
                provider=PROVIDER,
                cost_usd=unknown_spend,
            ) from exc

        cost = max(0.0, out.total_cost_usd)
        if out.is_error or proc.returncode != 0 or out.subtype not in ("", "success"):
            detail = out.result or _tail(proc.stderr) or "no detail"
            raise ProviderError(
                f"Claude Code failed (exit {proc.returncode}, subtype={out.subtype or 'unknown'}): {_tail(detail)}",
                provider=PROVIDER,
                cost_usd=cost,
                retryable=False,
            )

        parsed = None
        if request.json_schema is not None:
            if isinstance(out.structured_output, dict):
                raw_json = json.dumps(out.structured_output)
            elif out.structured_output is None:
                raw_json = out.result  # older CLIs put the JSON in ``result``
            else:
                raise StructuredOutputError(
                    f"structured_output is {type(out.structured_output).__name__}, expected an object",
                    provider=PROVIDER,
                    cost_usd=cost,
                )
            parsed = parse_json_output(raw_json, request.json_schema, provider=PROVIDER, cost_usd=cost)

        return CompletionResult(
            text=out.result,
            parsed=parsed,
            usage=usage_from_message(out.usage),
            cost_usd=cost,
            model=request.model,
            provider=PROVIDER,
            stop_reason=out.subtype,
            session_id=out.session_id,
            raw=out.model_dump(mode="json"),
        )

    def worst_case_cost(self, request: CompletionRequest, on: date | None = None) -> float:
        """``request.max_budget_usd`` plus ``turn_headroom_usd``: the CLI checks its cap only between turns."""
        if request.max_budget_usd is None:
            raise ValueError("Claude Code requests need max_budget_usd (metered_call sets it)")
        return request.max_budget_usd + self.turn_headroom_usd(request, on)

    def turn_headroom_usd(self, request: CompletionRequest, on: date | None = None) -> float:
        """Worst-case price of one model turn (full context at input/cache-write rate plus maximum output):
        how far ``total_cost_usd`` can overshoot ``--max-budget-usd``. ``metered_call`` holds it back."""
        return token_worst_case(request.model, self._turn_context_tokens, self._turn_output_tokens, on=on)

    @staticmethod
    def parse_output(stdout: str) -> ClaudeCodeOutput:
        """Parse the last JSON object on stdout (tolerate leading log noise).

        Raises ``ProviderError`` when stdout holds no result object."""
        payload = _last_json_value(stdout)
        if isinstance(payload, list):  # verbose mode emits the whole event list
            results = [e for e in payload if isinstance(e, dict) and e.get("type") == "result"]
            payload = results[-1] if results else None
        if not isinstance(payload, dict):
            raise ProviderError("no JSON result object on Claude Code stdout", provider=PROVIDER)
        return ClaudeCodeOutput.model_validate(payload)


def _last_json_value(stdout: str) -> Any:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    for i in range(len(lines) - 1, -1, -1):
        if not lines[i].lstrip().startswith(("{", "[")):
            continue
        start = offsets[i] + len(lines[i]) - len(lines[i].lstrip())
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        return value
    return None


def _tail(text: str) -> str:
    text = text.strip()
    return text if len(text) <= STDERR_TAIL_CHARS else "..." + text[-STDERR_TAIL_CHARS:]


__all__ = [
    "DISALLOWED_TOOLS",
    "SENSITIVE_READ_PATHS",
    "check_scoped_tools",
    "ClaudeCodeOutput",
    "ClaudeCodeProvider",
    "ClaudeCodeTimeout",
    "CompletedProcess",
    "Runner",
    "sandbox_settings",
    "subprocess_runner",
]
