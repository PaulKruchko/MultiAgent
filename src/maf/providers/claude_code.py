"""Claude Code headless (``claude -p``) as a provider, for coding and execution in a sandboxed workspace.

Owner: providers.

CLI facts, verified from ``claude --help`` (v2.1.284) and strings in the binary:

- The prompt goes on **stdin** (avoids the 128 KiB per-argument limit). ``-p/--print``, ``--output-format json``.
- ``--model <id>``, ``--effort <level>``, ``--max-budget-usd <amount>`` (print mode only),
  ``--allowedTools <tools...>``, ``--disallowedTools <tools...>``, ``--tools <list>`` (built-in set),
  ``--permission-mode dontAsk`` (anything not allowed is denied, no prompts),
  ``--permission-prompts none``, ``--no-session-persistence``, ``--json-schema <schema>``,
  ``--append-system-prompt <text>``, ``--settings <json>``, ``--add-dir`` (NOT used: no dirs outside cwd),
  ``--tools <list>`` (the built-in tool set), ``--setting-sources ""`` (no user/project/local settings, so no
  hooks or plugins), ``--strict-mcp-config`` (no MCP servers). ``--bare`` is NOT used: it forces a simple mode
  that exposes only Bash/Read/Edit, which ``--tools`` cannot widen (verified live 2026-09-28).
- Result JSON (single object): ``type="result"``, ``subtype`` in {``success``, ``error_max_turns``,
  ``error_during_execution``, ``error_max_budget_usd``}, ``is_error``, ``result`` (final text),
  ``session_id``, ``total_cost_usd``, ``duration_ms``, ``duration_api_ms``, ``num_turns``,
  ``usage``, ``modelUsage``, ``permission_denials``, ``structured_output`` (with ``--json-schema``).
  It is printed once, at the end, so a session killed at its timeout leaves no cost report on stdout.
- ``--output-format stream-json`` (print mode needs ``--verbose`` with it) is NOT used. The 2.1.284 binary writes the
  same result object as the last event of the stream (``L.write(ze)`` of the message ``json`` mode prints), and its
  schema has ``total_cost_usd``, ``is_error`` and ``structured_output``, which would let a timed-out session's
  streamed usage be estimated. But no stream from a real run is recorded as a fixture (recording one is a paid call),
  so the switch waits for one (checked 2026-09-29).

Sandbox: the subprocess cwd is ``workspaces/<run_id>/``. Writes outside it are prevented by
(a) no ``--add-dir``, (b) ``--permission-mode dontAsk`` with an explicit allowlist, and
(c) Claude Code's OS sandbox for Bash via ``--settings``:
``{"sandbox": {"enabled": true, "failIfUnavailable": true, "allowUnsandboxedCommands": false,
"autoAllowBashIfSandboxed": true, "filesystem": {"allowWrite": ["<workspace>"]},
"network": {"allowedDomains": []}}}`` (bubblewrap is installed at /usr/bin/bwrap). Web tools are disallowed.

File tools are allowed only with a workspace scope (``Read(./**)``, ``Edit(./**)``, ``Write(./**)``;
``check_scoped_tools`` rejects bare rules), and ``SENSITIVE_READ_PATHS`` are denied to both the file tools
(``permissions.deny``) and sandboxed Bash (``sandbox.filesystem.denyRead``). The tool and ``TMPDIR`` length rules live
in ``maf.sandbox``, so settings apply the same ones when they load (``Settings.claude_code_tools`` and
``claude_code_tmp_base``), before a run spends anything.

The subprocess env is built from an allowlist (``PATH``, ``HOME``, locale...), so credentials and ``CLAUDECODE``
(nested-session detection) never reach it. ``ANTHROPIC_API_KEY`` is NOT in the env either (sandboxed Bash
inherits the CLI's env): the key goes to the CLI through ``apiKeyHelper`` (``cat`` of a per-call 0600 file
under ``~/.config/maf/secrets``, which the sandbox cannot read), deleted when the call ends.
``CLAUDE_CODE_SUBPROCESS_ENV_SCRUB`` is NOT used: in CLI 2.1.284 it forces the default permission mode and
the OS sandbox stops confining Bash writes (verified live 2026-09-28).
``BASH_MAX_TIMEOUT_MS`` raises the longest timeout the Bash tool accepts (2.1.284: 10 minutes unless set, ``l=600000``
overridden only by that variable; the default per command stays 2 minutes, ``BASH_DEFAULT_TIMEOUT_MS``), so a long
simulation or QEMU battery is not killed mid-run; a ``bound_to`` copy for a long reproduction raises the default too.
``MPLCONFIGDIR`` points into ``<workspace>/.maf/``. ``TMPDIR`` is ``<tmp_base>/maf-<12 random hex>`` (``/tmp`` by
default), picked once per provider instance: a private 0700 directory that is the one write location outside the
workspace (``sandbox.filesystem.allowWrite`` lists both), removed when the provider is collected or the process
exits. The name is random because the ``--settings`` argv shows it to every local user, and a predictable name could
be created first by someone else. It must be short: the sandbox runtime creates its Unix sockets under ``TMPDIR``,
and a ``TMPDIR`` inside a long workspace path pushed them past the 108-byte ``sun_path`` limit, so every Bash call
failed with "Failed to create bridge sockets" (verified live 2026-09-28). ``complete`` refuses a ``TMPDIR`` longer
than ``max_tmpdir_bytes()``, the CLI's own budget for the ``TMPDIR`` it gives sandboxed commands.

Sandbox failures are fatal: output matching ``SANDBOX_FAILURE_PATTERNS`` raises ``SandboxUnavailable`` (with the
real spend), and ``preflight`` proves sandboxed Bash can read the workspace and write both ``$TMPDIR`` and the
workspace (``PREFLIGHT_COMMAND`` on a random file) before a run pays for real work.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import weakref
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from maf.providers.base import (
    CompletionRequest,
    CompletionResult,
    ProviderError,
    SandboxUnavailable,
    StructuredOutputError,
    parse_json_output,
    token_worst_case,
)
from maf.providers.claude_provider import DEFAULT_EFFORT, usage_from_message
from maf.sandbox import (
    CLI_CHILD_TMPDIR_MAX_BYTES,
    DEFAULT_TMP_BASE,
    check_scoped_tools,
    max_tmpdir_bytes,
    scratch_tmpdir,
)
from maf.types import AgentName, ProviderName

PROVIDER: ProviderName = "claude_code"
DISALLOWED_TOOLS: tuple[str, ...] = ("WebFetch", "WebSearch")
STDERR_TAIL_CHARS = 2_000

ENV_ALLOWLIST: tuple[str, ...] = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LANGUAGE", "TERM", "TZ")
"""Inherited variables passed to the CLI (plus ``LC_*``). Everything else, such as
cloud or GitHub tokens, is dropped, so Bash children cannot copy it into workspace artifacts."""

SENSITIVE_READ_PATHS: tuple[str, ...] = (
    # credentials and key stores
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
    "~/.password-store",
    "~/.pki",
    "~/.pypirc",
    "~/.npmrc",
    "~/.pgpass",
    "~/.vault-token",
    "~/.cargo/credentials",
    "~/.cargo/credentials.toml",
    # shell startup files, which often export API keys, and histories, which hold keys typed or pasted
    "~/.bashrc",
    "~/.bash_profile",
    "~/.bash_login",
    "~/.profile",
    "~/.bash_aliases",
    "~/.zshrc",
    "~/.zshenv",
    "~/.zprofile",
    "~/.bash_history",
    "~/.zsh_history",
    "~/.python_history",
    "~/.node_repl_history",
    "~/.psql_history",
    "~/.mysql_history",
    "~/.sqlite_history",
    "~/.viminfo",
    # browser and mail profiles (cookies, saved passwords)
    "~/.mozilla",
    "~/.thunderbird",
    "~/snap/firefox",
    "~/snap/chromium",
    "~/snap/thunderbird",
    "~/.var/app",
)
"""Denied to both the file tools (``permissions.deny``) and sandboxed Bash (``sandbox.filesystem.denyRead``). Sandboxed
Bash can read the rest of the filesystem, and a brief can come from a ChatGPT conversation that read untrusted pages,
so everything here is what a prompt injection would look for first: key files, shell files that export keys (the docs
move provider keys to the ``~/.config/maf/maf.env`` this list already covers), histories and browser profiles. Entries
may be files or directories and may be missing (the list always held missing ones, such as ``~/.kube`` on the dev
host, and runs passed the sandbox preflight)."""

DEFAULT_TURN_CONTEXT_TOKENS = 200_000
DEFAULT_TURN_OUTPUT_TOKENS = 64_000

SANDBOX_FAILURE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsandbox (?:is required but |has |had )?failed to initiali[sz]e", re.IGNORECASE),
    re.compile(r"\bfailed to initiali[sz]e (?:the )?(?:\w+ )?sandbox\b", re.IGNORECASE),
    re.compile(r"failed to create bridge sockets?", re.IGNORECASE),
    re.compile(r"\bbwrap: \S"),
)
"""Sandbox start-up failures in Claude Code's output or stderr: the CLI's "Sandbox is required but failed to
initialize: Failed to create bridge sockets after 5 attempts", a model's paraphrase of it ("the Bash sandbox failed
to initialize"), or bubblewrap's own ``bwrap: ...`` errors. Deliberately narrow: a false match stops a healthy run."""

PREFLIGHT_FILE = ".maf/preflight.bin"
"""Workspace-relative random file the preflight hashes."""
PREFLIGHT_OUTPUT = ".maf/preflight.out"
"""Workspace-relative file the preflight command writes the digest to; Python reads it back."""
PREFLIGHT_COMMAND = (
    f'cp {PREFLIGHT_FILE} "$TMPDIR/maf-preflight" && sha256sum "$TMPDIR/maf-preflight" | tee {PREFLIGHT_OUTPUT}'
)
"""Reads the workspace, writes ``$TMPDIR`` (only a successful copy is hashed) and writes the workspace (``tee``): the
two write locations every compiler, here-doc and Python ``tempfile`` call depends on."""
PREFLIGHT_BYTES = 4096
PREFLIGHT_MAX_OUTPUT_TOKENS = 1_024
PREFLIGHT_MIN_BUDGET_USD = 0.15
PREFLIGHT_TURNS = 3
PREFLIGHT_CONTEXT_TOKENS = 25_000
"""The default preflight cap (``preflight_budget_usd``) prices ``PREFLIGHT_TURNS`` turns of this much context (a
first turn carries about 13-21k tokens of system prompt and tools) at the worst-case rate."""
PREFLIGHT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"digest": {"type": "string"}},
    "required": ["digest"],
    "additionalProperties": False,
}
PREFLIGHT_PROMPT = (
    "Sandbox health check. Use the Bash tool exactly once to run this command, and nothing else:\n\n"
    f"{PREFLIGHT_COMMAND}\n\n"
    "Answer with the 64-character hexadecimal digest it printed as `digest`. If the command failed, answer with "
    "its error message as `digest` instead. Do not compute the digest any other way and do not use other tools."
)

_HEX_DIGEST = re.compile(r"\b[0-9a-fA-F]{64}\b")


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
    """The CLI exceeded ``timeout_s`` and was killed. Spend is unknown, so ``complete`` raises it with ``cost_usd`` at
    the worst case (budget plus one turn): the cap stays hard. Its work stays in the workspace, so an execution or fix
    session gets one continuation (``maf.stages.base.work_session``)."""


class ClaudeCodeBudgetExhausted(ProviderError):
    """The session stopped at its ``--max-budget-usd`` cap (``subtype == "error_max_budget_usd"``). Not retryable."""


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


DEFAULT_SECRETS_DIR = Path.home() / ".config" / "maf" / "secrets"
"""Per-call API key files for ``apiKeyHelper``. Must sit under a ``SENSITIVE_READ_PATHS`` entry (``~/.config``)."""


def sandbox_settings(
    workspace: Path,
    deny_read: Sequence[str] = SENSITIVE_READ_PATHS,
    api_key_helper: str | None = None,
    tmpdir: Path | None = None,
) -> dict[str, Any]:
    """The ``--settings`` JSON object shown in the module docstring, bound to ``workspace``.

    ``deny_read`` entries are ``~``-relative or absolute paths. They become ``Read``/``Edit`` deny rules
    (``~/x/**`` or ``//abs/x/**``) and ``sandbox.filesystem.denyRead`` entries. ``tmpdir`` (the CLI's
    ``TMPDIR``) is writable next to the workspace: the sandbox runtime puts its sockets there."""
    rules = [_rule_path(p) for p in deny_read]
    allow_write = [str(workspace.resolve())]
    if tmpdir is not None:
        allow_write.append(str(tmpdir))
    settings: dict[str, Any] = {
        "permissions": {"deny": [f"{tool}({rule})" for rule in rules for tool in ("Read", "Edit")]},
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "allowUnsandboxedCommands": False,
            "autoAllowBashIfSandboxed": True,
            "filesystem": {"allowWrite": allow_write, "denyRead": list(deny_read)},
            "network": {"allowedDomains": []},
        },
    }
    if api_key_helper is not None:
        settings["apiKeyHelper"] = api_key_helper
    return settings


def _rule_path(path: str) -> str:
    """Permission-rule path form: ``~/x`` stays home-relative, ``/abs`` becomes ``//abs``; ``/**`` is appended."""
    base = path.rstrip("/")
    if base.startswith("/"):
        base = "/" + base
    return f"{base}/**"


def builtin_tools(allowed: Sequence[str]) -> list[str]:
    """Tool names for ``--tools``: the distinct names in ``allowed`` rules (``Bash(make *)`` -> ``Bash``).
    ``Edit`` implies ``Write``: Edit rules govern both, and Write must still be listed to be available."""
    names = [rule.split("(", 1)[0].strip() for rule in allowed]
    if "Edit" in names:
        names.append("Write")
    return list(dict.fromkeys(names))


def format_budget(usd: float) -> str:
    """Round *down* to 1/100 cent so the CLI cap never exceeds the ledger's clamp."""
    return f"{math.floor(usd * 10_000) / 10_000:.4f}"


def check_tmpdir(tmpdir: Path) -> None:
    """``ProviderError`` (cost 0, not retryable) when ``tmpdir`` is relative, or longer than ``max_tmpdir_bytes()``
    (counted in bytes, as ``sun_path`` is), so sandboxed commands would get a ``TMPDIR`` too long for their sockets."""
    if not tmpdir.is_absolute():
        raise ProviderError(f"Claude Code TMPDIR must be absolute, got {str(tmpdir)!r}", provider=PROVIDER)
    length, limit = len(os.fsencode(tmpdir)), max_tmpdir_bytes()
    if length > limit:
        raise ProviderError(
            f"Claude Code TMPDIR {tmpdir} is {length} bytes long (at most {limit}): sandboxed commands get "
            f"TMPDIR=<TMPDIR>/claude-{os.getuid()}, which must stay within {CLI_CHILD_TMPDIR_MAX_BYTES} bytes for "
            "Unix socket paths under it to fit the 108-byte limit; set claude_code_tmp_base to a shorter directory",
            provider=PROVIDER,
        )


def ensure_private_dir(path: Path) -> bool:
    """Create ``path`` with mode 0700, or check an existing one: it must be a real directory (not a symlink)
    owned by this user, and its mode is reset to 0700. ``ProviderError`` (cost 0) when it cannot be used.
    The parent must exist. Returns True when this call created the directory."""
    try:
        path.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        created = False
    except OSError as exc:
        raise ProviderError(f"cannot create Claude Code TMPDIR {path}: {exc}", provider=PROVIDER) from exc
    if path.is_symlink():
        raise ProviderError(f"refusing Claude Code TMPDIR {path}: it is a symlink", provider=PROVIDER)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)  # no-follow closes the symlink race
    except OSError as exc:
        raise ProviderError(f"refusing Claude Code TMPDIR {path}: {exc.strerror or exc}", provider=PROVIDER) from exc
    try:
        info = os.fstat(fd)
        if info.st_uid != os.getuid():
            raise ProviderError(
                f"refusing Claude Code TMPDIR {path}: owned by uid {info.st_uid}, not {os.getuid()}",
                provider=PROVIDER,
            )
        if stat.S_IMODE(info.st_mode) != 0o700:
            os.fchmod(fd, 0o700)
    finally:
        os.close(fd)
    return created


def sandbox_failure(*texts: str) -> str | None:
    """The first line of ``texts`` matching ``SANDBOX_FAILURE_PATTERNS`` (stripped, capped at 300 characters),
    or None."""
    for text in texts:
        for line in text.splitlines():
            if any(pattern.search(line) for pattern in SANDBOX_FAILURE_PATTERNS):
                return line.strip()[:300]
    return None


def reported_digest(parsed: dict[str, Any] | None) -> str | None:
    """The lower-case SHA-256 hex digest in a preflight answer's ``digest`` field, or None."""
    value = (parsed or {}).get("digest")
    match = _HEX_DIGEST.search(value) if isinstance(value, str) else None
    return match.group(0).lower() if match else None


def written_digest(path: Path) -> str | None:
    """The lower-case SHA-256 hex digest in the first ``PREFLIGHT_BYTES`` of the regular file ``path`` (not followed
    if it is a symlink), or None."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as fh:
        if not stat.S_ISREG(os.fstat(fh.fileno()).st_mode):
            return None
        return reported_digest({"digest": fh.read(PREFLIGHT_BYTES).decode("ascii", "replace")})


def preflight_budget_usd(model: str, on: date | None = None) -> float:
    """Default ``--max-budget-usd`` of the preflight: ``PREFLIGHT_TURNS`` turns of ``PREFLIGHT_CONTEXT_TOKENS`` context
    and ``PREFLIGHT_MAX_OUTPUT_TOKENS`` output at the worst-case rate (``token_worst_case``), at least
    ``PREFLIGHT_MIN_BUDGET_USD``. About $0.44 on claude-opus-5-5 and $1.09 on claude-fable-5-1, whose first turn alone
    can cost more than a flat $0.15. Only the actual spend is billed."""
    turn = token_worst_case(model, PREFLIGHT_CONTEXT_TOKENS, PREFLIGHT_MAX_OUTPUT_TOKENS, on=on)
    return max(PREFLIGHT_MIN_BUDGET_USD, PREFLIGHT_TURNS * turn)


def preflight_request(model: str, budget_usd: float) -> CompletionRequest:
    """The cheap sandbox check: low effort, a small budget, and ``PREFLIGHT_SCHEMA`` structured output."""
    return CompletionRequest.simple(
        model,
        PREFLIGHT_PROMPT,
        max_output_tokens=PREFLIGHT_MAX_OUTPUT_TOKENS,
        json_schema=PREFLIGHT_SCHEMA,
        schema_name="preflight",
        effort="low",
        max_budget_usd=budget_usd,
    )


class ClaudeCodeProvider:
    name: ProviderName = PROVIDER
    agent: AgentName = "claude"

    def __init__(
        self,
        workspace: Path,
        *,
        executable: Path,
        allowed_tools: Sequence[str],
        timeout_s: float = 5400.0,
        runner: Runner | None = None,
        extra_env: dict[str, str] | None = None,
        path_prepend: Sequence[Path] = (),
        deny_read: Sequence[str] = SENSITIVE_READ_PATHS,
        turn_context_tokens: int = DEFAULT_TURN_CONTEXT_TOKENS,
        turn_output_tokens: int = DEFAULT_TURN_OUTPUT_TOKENS,
        secrets_dir: Path = DEFAULT_SECRETS_DIR,
        tmp_base: Path = DEFAULT_TMP_BASE,
        bash_timeout_s: float | None = None,
    ) -> None:
        """``path_prepend`` puts directories (e.g. the project venv's ``bin``) first on the CLI's ``PATH``.
        ``deny_read`` lists paths no tool may read. ``turn_*_tokens`` size one model turn, the amount the CLI
        can overshoot ``--max-budget-usd`` by (see ``turn_headroom_usd``). ``tmp_base`` holds this instance's
        ``TMPDIR`` (``Settings.claude_code_tmp_base``). ``bash_timeout_s`` is the longest Bash command timeout
        (``BASH_MAX_TIMEOUT_MS``; None keeps the CLI's 10 minutes); keep it below ``timeout_s``."""
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
        self.secrets_dir = secrets_dir
        self.tmp_base = tmp_base
        self.bash_timeout_s = bash_timeout_s
        self._bash_default_is_max = False
        self._tmpdir: Path | None = None
        self.sandbox_verified = False
        """Set by a passing ``preflight``; cleared when a call reports a sandbox failure."""

    def bound_to(
        self, workspace: Path, *, deny_read: Sequence[str] = (), bash_default_is_max: bool = False
    ) -> ClaudeCodeProvider:
        """A copy of this provider whose cwd and only writable directory (besides ``TMPDIR``) is ``workspace``, with
        ``deny_read`` added to the paths no tool may read. With ``bash_default_is_max`` every Bash command may run
        for ``bash_timeout_s`` without asking for it (``BASH_DEFAULT_TIMEOUT_MS``). The copy shares this instance's
        ``TMPDIR`` and its preflight verdict: same CLI, same sandbox, another directory. The final stage's clean room
        uses it, so the reproduction can neither read nor write the workspace it came from."""
        self.tmpdir  # picked before copying, so both instances agree
        clone = copy.copy(self)
        clone.workspace = workspace
        clone._deny_read = (*self._deny_read, *deny_read)
        clone._bash_default_is_max = bash_default_is_max
        return clone

    @property
    def tmpdir(self) -> Path:
        """The CLI's ``TMPDIR``: a random ``scratch_tmpdir(tmp_base)``, picked on first use and kept, so ``build_env``
        and ``build_argv`` agree. No I/O: ``complete`` creates it."""
        if self._tmpdir is None:
            self._tmpdir = scratch_tmpdir(self.tmp_base)
        return self._tmpdir

    def _prepare_tmpdir(self) -> None:
        """Create ``tmpdir`` (0700) or re-check it before a call. If the path is unusable (another user's directory,
        a symlink or a file: someone took the name after seeing it in ``ps``), switch once to a fresh random name
        instead of blocking the run. A directory created here is removed when the provider is collected or the
        process exits; nothing is read back from it."""
        try:
            created = ensure_private_dir(self.tmpdir)
        except ProviderError:
            self._tmpdir = None
            created = ensure_private_dir(self.tmpdir)
        if created:
            weakref.finalize(self, shutil.rmtree, self.tmpdir, ignore_errors=True)

    def build_argv(self, request: CompletionRequest, api_key_helper: str | None = None) -> list[str]:
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
            "--setting-sources", "",
            "--strict-mcp-config",
            "--tools", ",".join(builtin_tools(self.allowed_tools)),
            "--settings", json.dumps(
                sandbox_settings(self.workspace, self._deny_read, api_key_helper, tmpdir=self.tmpdir),
                separators=(",", ":"),
            ),
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
        """Subprocess environment from an allowlist (``ENV_ALLOWLIST``, ``LC_*``) plus ``TMPDIR`` (the short
        ``self.tmpdir``), a workspace-local ``MPLCONFIGDIR`` and ``BASH_MAX_TIMEOUT_MS`` (with ``bash_timeout_s``). No
        credentials: the API key reaches the CLI via ``apiKeyHelper``. Pure: ``complete`` creates and checks the
        directories."""
        env = {k: v for k, v in os.environ.items() if k in ENV_ALLOWLIST or k.startswith("LC_")}
        if self._path_prepend:
            env["PATH"] = os.pathsep.join([*(str(p) for p in self._path_prepend), env.get("PATH", "")])
        env["TMPDIR"] = str(self.tmpdir)
        env["MPLCONFIGDIR"] = str(self.workspace.resolve() / ".maf" / "mpl")
        if self.bash_timeout_s is not None:
            env["BASH_MAX_TIMEOUT_MS"] = str(int(self.bash_timeout_s * 1000))
            if self._bash_default_is_max:
                env["BASH_DEFAULT_TIMEOUT_MS"] = env["BASH_MAX_TIMEOUT_MS"]
        env.update(self._extra_env)
        return env

    def complete(self, request: CompletionRequest) -> CompletionResult:
        """Run the CLI in ``self.workspace`` and parse the JSON.

        - ``request.max_budget_usd`` must be set (``metered_call`` does it); else ``ValueError``.
        - ``is_error`` or a non-zero exit raises ``ProviderError(cost_usd=total_cost_usd, retryable=False)``;
          ``subtype == "error_max_budget_usd"`` raises its subclass ``ClaudeCodeBudgetExhausted``.
        - ``cost_usd`` = ``total_cost_usd`` (authoritative; not re-priced from tokens).
        - ``text`` = ``result``; ``parsed`` = ``structured_output`` when a schema was given.
        - A timeout (``ClaudeCodeTimeout``), a crash without a JSON result, or a runner failure charges
          ``worst_case_cost`` (budget plus one turn), since the real spend is unknown.
        - A sandbox start-up failure (``sandbox_failure`` on the result text, stderr and the strings in
          ``structured_output``) raises ``SandboxUnavailable`` carrying ``total_cost_usd`` (the worst case when
          there is no JSON result). It is checked first, so it never surfaces as a plain failure or as a
          ``StructuredOutputError`` (which the fix pass tolerates).
        - Before spawning: ``TMPDIR`` longer than ``max_tmpdir_bytes()`` raises ``ProviderError`` at cost 0, and so
          does an unusable one (a symlink, another user's) when a fresh random name is unusable too.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set (required for Claude Code API billing)", provider=PROVIDER)
        check_tmpdir(self.tmpdir)
        unknown_spend = self.worst_case_cost(request)  # charged whenever the CLI's own cost report is missing
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._prepare_tmpdir()  # may switch to a fresh name, so the env and argv are built after it
        env = self.build_env()
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

        key_file = self._write_key_file(api_key)
        try:
            argv = self.build_argv(request, api_key_helper=f"cat {shlex.quote(str(key_file))}")
            proc = self._runner(argv, self.build_stdin(request), self.workspace, env, self.timeout_s)
        except ClaudeCodeTimeout as exc:
            raise ClaudeCodeTimeout(str(exc), provider=PROVIDER, cost_usd=unknown_spend) from exc
        except ProviderError:
            raise  # the CLI could not be started: nothing was spent
        except Exception as exc:
            raise ProviderError(
                f"Claude Code runner failed: {type(exc).__name__}: {exc}", provider=PROVIDER, cost_usd=unknown_spend
            ) from exc
        finally:
            key_file.unlink(missing_ok=True)

        try:
            out = self.parse_output(proc.stdout)
        except ProviderError as exc:
            self._check_sandbox(unknown_spend, proc.stderr, proc.stdout)
            detail = _tail(proc.stderr) or str(exc)
            raise ProviderError(
                f"Claude Code exited {proc.returncode} without a result: {detail}",
                provider=PROVIDER,
                cost_usd=unknown_spend,
            ) from exc

        cost = max(0.0, out.total_cost_usd)
        self._check_sandbox(cost, out.result, proc.stderr, *_strings(out.structured_output))
        if out.is_error or proc.returncode != 0 or out.subtype not in ("", "success"):
            detail = out.result or _tail(proc.stderr) or "no detail"
            error = ClaudeCodeBudgetExhausted if out.subtype == "error_max_budget_usd" else ProviderError
            raise error(
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

    def _check_sandbox(self, cost_usd: float, *texts: str) -> None:
        """``SandboxUnavailable(cost_usd)`` when ``texts`` show a sandbox start-up failure."""
        line = sandbox_failure(*texts)
        if line is None:
            return
        self.sandbox_verified = False
        raise SandboxUnavailable(
            f"Claude Code's Bash sandbox is unavailable ({line}); stopping instead of paying for sessions whose "
            f"commands all fail. Check bubblewrap and TMPDIR {self.tmpdir}",
            provider=PROVIDER,
            cost_usd=cost_usd,
        )

    def preflight(
        self,
        model: str,
        budget_usd: float | None = None,
        *,
        call: Callable[[CompletionRequest], CompletionResult] | None = None,
    ) -> CompletionResult:
        """Prove sandboxed Bash works before a run pays for real work.

        Writes ``PREFLIGHT_BYTES`` random bytes to ``<workspace>/PREFLIGHT_FILE`` and asks Claude Code (effort low,
        ``budget_usd``, by default ``preflight_budget_usd(model)``) to run exactly ``PREFLIGHT_COMMAND``: copy the file
        into ``$TMPDIR``, hash the copy, and ``tee`` the digest into ``<workspace>/PREFLIGHT_OUTPUT``. The digest must
        match ``hashlib`` both in the answer (``PREFLIGHT_SCHEMA``) and in that file (checked by Python), so a sandbox
        that starts but cannot write ``TMPDIR`` or the workspace fails too. A wrong or missing digest raises
        ``SandboxUnavailable``; its ``cost_usd`` is the preflight's spend, already recorded by a metered ``call``.
        Both files are always deleted.

        ``call`` performs the request: stages pass a metered ``StageContext.call``; the default is
        ``self.complete`` (unmetered). Running out of the preflight's own cap (``ClaudeCodeBudgetExhausted``) raises a
        plain ``ProviderError`` that says so; other ``ProviderError``s (budget, missing key) propagate unchanged.
        Sets ``sandbox_verified`` on success."""
        self.sandbox_verified = False
        budget = preflight_budget_usd(model) if budget_usd is None else budget_usd
        probe, output = self.workspace / PREFLIGHT_FILE, self.workspace / PREFLIGHT_OUTPUT
        probe.parent.mkdir(parents=True, exist_ok=True)
        for path in (probe, output):
            path.unlink(missing_ok=True)  # sessions can write here: never use a symlink one left behind
        data = os.urandom(PREFLIGHT_BYTES)
        fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        expected = hashlib.sha256(data).hexdigest()
        try:
            result = (call or self.complete)(preflight_request(model, budget))
            written = written_digest(output)
        except StructuredOutputError as exc:
            raise SandboxUnavailable(
                f"Claude Code sandbox preflight returned no digest: {exc}", provider=PROVIDER, cost_usd=exc.cost_usd
            ) from exc
        except ClaudeCodeBudgetExhausted as exc:
            raise ProviderError(
                f"Claude Code sandbox preflight ran out of its budget on {model} before answering; this is not a "
                "sandbox failure: raise claude_code_preflight_budget_usd (unset, it scales with the model's price) "
                f"or the run budget ({exc})",
                provider=PROVIDER,
                cost_usd=exc.cost_usd,
            ) from exc
        finally:
            for path in (probe, output):
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
        digest = reported_digest(result.parsed)
        if digest != expected:
            answer = " ".join(str((result.parsed or {}).get("digest", result.text)).split())[:300]
            raise SandboxUnavailable(
                f"Claude Code sandbox preflight failed: `{PREFLIGHT_COMMAND}` should print {expected}, "
                f"the session reported {answer or 'nothing'!r}",
                provider=PROVIDER,
                cost_usd=result.cost_usd,
            )
        if written != expected:
            raise SandboxUnavailable(
                f"Claude Code sandbox preflight failed: the session reported the right digest but {PREFLIGHT_OUTPUT} "
                "does not hold it, so sandboxed commands cannot write the workspace (or the command was not run as "
                "given); check sandbox.filesystem.allowWrite",
                provider=PROVIDER,
                cost_usd=result.cost_usd,
            )
        self.sandbox_verified = True
        return result

    def _write_key_file(self, api_key: str) -> Path:
        """A fresh 0600 file holding ``api_key`` in ``secrets_dir`` (0700), for the CLI's ``apiKeyHelper``."""
        self.secrets_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="anthropic-", dir=self.secrets_dir)
        with os.fdopen(fd, "w") as fh:
            fh.write(api_key)
        return Path(name)

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


def _strings(value: Any) -> list[str]:
    """Every string inside a JSON value (structured output), depth first."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _tail(text: str) -> str:
    text = text.strip()
    return text if len(text) <= STDERR_TAIL_CHARS else "..." + text[-STDERR_TAIL_CHARS:]


__all__ = [
    "CLI_CHILD_TMPDIR_MAX_BYTES",
    "DISALLOWED_TOOLS",
    "PREFLIGHT_COMMAND",
    "PREFLIGHT_FILE",
    "PREFLIGHT_OUTPUT",
    "SANDBOX_FAILURE_PATTERNS",
    "SENSITIVE_READ_PATHS",
    "check_scoped_tools",
    "check_tmpdir",
    "ClaudeCodeBudgetExhausted",
    "ClaudeCodeOutput",
    "ClaudeCodeProvider",
    "ClaudeCodeTimeout",
    "CompletedProcess",
    "ensure_private_dir",
    "max_tmpdir_bytes",
    "preflight_budget_usd",
    "preflight_request",
    "reported_digest",
    "Runner",
    "sandbox_failure",
    "sandbox_settings",
    "scratch_tmpdir",
    "subprocess_runner",
    "written_digest",
]
