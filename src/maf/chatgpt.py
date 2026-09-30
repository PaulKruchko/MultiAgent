"""``maf chatgpt setup|status``: run ``maf serve`` for the ChatGPT app behind the OpenAI Secure MCP Tunnel.

Owner: orchestration.

Two systemd user units (docs/CHATGPT.md explains the whole setup):
- ``maf-mcp.service``: ``maf serve --host 127.0.0.1 --port <mcp_port> --uds %t/maf/mcp.sock`` from the project venv,
  plus ``--config``/``--vault``/``--workspaces`` and ``MAF_BUDGET_USD`` when setup was given them. It serves HTTP on a
  0600 Unix socket in ``RuntimeDirectory=maf`` (``$XDG_RUNTIME_DIR/maf``, 0700), not on a TCP port: a loopback port is
  open to every local user and to containers on the host network, a socket in the owner's runtime directory only to
  the owner. Provider keys come from ``~/.config/maf/maf.env`` (services do not read ``~/.bashrc``). SIGTERM goes to
  maf only (``KillMode=mixed``): it stops serving at once and the in-flight run stops at its next stage boundary, for
  up to ``TimeoutStopSec`` (``mcp_stop_timeout``: the longest stage, derived from ``claude_code_timeout_s``). Exit 2
  (usage or configuration error) is not restarted.
- ``maf-tunnel.service``: ``tunnel-client run`` with the upstream ``url=http://127.0.0.1:<mcp_port>/mcp,
  unix-socket=%t/maf/mcp.sock`` (tunnel-client dials the socket and sends ``Host: 127.0.0.1:<mcp_port>``), the tunnel
  id and runtime key from ``~/.config/maf/tunnel.env``. ``Requires=``/``After=`` maf-mcp, so tunnel restarts never touch
  maf and in-flight runs survive them.

``setup`` writes both env files as 0600 templates if they are missing (an existing file is never overwritten, only
tightened to 0600), creates ``mcp_inbox`` (0700) if missing, renders the units into ``$XDG_CONFIG_HOME/systemd/user``
and runs ``systemctl --user daemon-reload``. A running unit keeps its old definition and environment until restarted,
so setup says so when it changed an active unit. ``status`` shows unit states, the last journal lines (known secret
values and key-shaped strings masked), which keys are set (never their values), the MCP health (a local initialize +
tools/list round trip over the unit's socket) and the tunnel-client readiness. ``contrib/systemd/`` holds
``render_units`` for the default layout; a test keeps it in sync.

Hardening (verified on Ubuntu 24.04, systemd 255, with ``kernel.apparmor_restrict_unprivileged_userns=1``, by running
bwrap in transient user units): Claude Code's Bash sandbox is bubblewrap, which needs unprivileged user namespaces.
``RestrictNamespaces=`` breaks it, and so does every option that implies ``PrivateUsers=yes`` in a user unit
(``ProtectSystem=``, ``ProtectHome=``, ``PrivateTmp=``, ``PrivateDevices=``, ``ProtectKernel*=``...): bwrap can then not
create its own namespaces. ``NoNewPrivileges=yes`` and the seccomp-based options did not break it (bwrap here is not
setuid, and it sets no_new_privs on sandboxed commands itself). maf-mcp therefore gets only ``NoNewPrivileges``,
``RestrictSUIDSGID``, ``RestrictRealtime``, ``UMask=0077`` and ``LimitCORE=0``; tunnel-client, a static Go binary that
needs nothing but outbound HTTPS and loopback, gets everything a user unit can apply (options that drop capabilities,
such as ``PrivateDevices=`` or ``ProtectClock=``, fail in a user unit with 218/CAPABILITIES).
"""

from __future__ import annotations

import ipaddress
import math
import os
import re
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from maf.config import Settings
from maf.redact import KEY_VARIABLES as KEY_VARIABLES  # re-exported: removed from tunnel-client's environment
from maf.redact import LOG_PATTERNS, known_secrets
from maf.redact import redact as _redact

MCP_UNIT = "maf-mcp.service"
TUNNEL_UNIT = "maf-tunnel.service"
UNITS = (MCP_UNIT, TUNNEL_UNIT)
TUNNEL_ENV = "tunnel.env"
MAF_ENV = "maf.env"
TUNNEL_ENV_KEYS = ("CONTROL_PLANE_TUNNEL_ID", "CONTROL_PLANE_API_KEY")
MAF_ENV_KEYS = ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY")
DEFAULT_HEALTH_PORT = 8766
"""tunnel-client's loopback health/admin listener (/healthz, /readyz, /ui); its own default 8080 is often taken."""
MCP_STOP_SESSIONS = 3
"""Claude Code sessions one stage can run back to back, each up to ``claude_code_timeout_s``: an execution session, its
continuation after a timeout and the repair of its handoff (a cross-check runs a fix session and its continuation, final
the clean room)."""
MCP_STOP_MARGIN_S = 1800.0
"""The rest of the longest stage besides its sessions: the sandbox preflight, the critiques, rebuttal and adjudication,
the source audits and the final report."""


def mcp_stop_timeout(claude_code_timeout_s: float) -> str:
    """maf-mcp ``TimeoutStopSec``: stop waits for the in-flight stage to end, and after this systemd SIGKILLs the whole
    group. A Claude Code session killed that way is never recorded (SIGKILL cannot be caught, so ``metered_call`` does
    not charge its worst case), and ``maf resume`` would continue against a ledger missing its spend. So it covers the
    longest stage: ``MCP_STOP_SESSIONS`` sessions of ``claude_code_timeout_s`` plus ``MCP_STOP_MARGIN_S``, rounded up to
    the minute: ``5h`` at the 90-minute default. A stop normally takes seconds to minutes; ``maf chatgpt setup`` must
    run again after ``claude_code_timeout_s`` changes."""
    minutes = math.ceil((MCP_STOP_SESSIONS * claude_code_timeout_s + MCP_STOP_MARGIN_S) / 60)
    return f"{minutes // 60}h" if minutes % 60 == 0 else f"{minutes}min"


MCP_STOP_TIMEOUT = mcp_stop_timeout(Settings.model_fields["claude_code_timeout_s"].default)
"""``mcp_stop_timeout`` at the default ``claude_code_timeout_s`` (the ``contrib/systemd/`` rendering)."""
TUNNEL_ID_RE = re.compile(r"tunnel_[0-9a-f]{32}")
CONTRIB_HOME = Path("/home/USER")
"""Home used to render ``contrib/systemd/`` (every path becomes ``%h/...``)."""
RUNTIME_SUBDIR = "maf"
"""maf-mcp's ``RuntimeDirectory=``: ``$XDG_RUNTIME_DIR/maf``, created 0700 by systemd and removed when maf stops."""
MCP_SOCKET_NAME = "mcp.sock"
MCP_SOCKET_UNIT_PATH = f"%t/{RUNTIME_SUBDIR}/{MCP_SOCKET_NAME}"
"""The MCP socket as both units name it (``%t`` is ``$XDG_RUNTIME_DIR`` in a user unit)."""
MAF_ENV_VARIABLES = ("MAF_BUDGET_USD",)
"""Settings environment variables without a ``maf serve`` flag; setup copies them into the unit as ``Environment=``."""

_UNIT_WORD_RE = re.compile(r"[A-Za-z0-9_@%+=:,./\[\]-]+")
_SPECIFIER_PREFIXES = ("%h", "%t")

Runner = Callable[[Sequence[str], float], "subprocess.CompletedProcess[str]"]
"""``(argv, timeout_s) -> CompletedProcess`` with text output; a missing binary or timeout is a failed process."""


def run_command(argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Default ``Runner``: never raises for a missing program (rc 127) or a timeout (rc 124)."""
    try:
        return subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout, check=False)  # noqa: S603
    except FileNotFoundError:
        return subprocess.CompletedProcess(list(argv), 127, "", f"{argv[0]}: command not found")
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(list(argv), 124, "", f"{argv[0]}: timed out after {timeout:g} s")


# --------------------------------------------------------------------------- layout and units


@dataclass(frozen=True)
class Layout:
    """Where setup writes: env files in ``maf_config_dir`` (``~/.config/maf``, like maf's config.yaml and secrets),
    units in ``unit_dir`` (``$XDG_CONFIG_HOME/systemd/user``, where the user manager looks)."""

    home: Path
    maf_config_dir: Path
    unit_dir: Path
    runtime_dir: Path = field(default_factory=lambda: Path(f"/run/user/{os.getuid()}"))
    """``$XDG_RUNTIME_DIR``, which ``%t`` stands for in the units."""

    @classmethod
    def default(cls, environ: Mapping[str, str] | None = None, home: Path | None = None) -> Layout:
        environ = os.environ if environ is None else environ
        home = Path.home() if home is None else home
        xdg = environ.get("XDG_CONFIG_HOME", "").strip()
        config_root = Path(xdg) if xdg and Path(xdg).is_absolute() else home / ".config"
        runtime = environ.get("XDG_RUNTIME_DIR", "").strip()
        runtime_dir = Path(runtime) if runtime and Path(runtime).is_absolute() else Path(f"/run/user/{os.getuid()}")
        return cls(home=home, maf_config_dir=home / ".config" / "maf", unit_dir=config_root / "systemd" / "user",
                   runtime_dir=runtime_dir)

    def expand(self, word: str) -> Path:
        """A unit word with ``%h``/``%t`` replaced by this layout's home and runtime directory."""
        if word.startswith("%h"):
            return Path(str(self.home) + word[2:])
        if word.startswith("%t"):
            return Path(str(self.runtime_dir) + word[2:])
        return Path(word)

    @property
    def tunnel_env(self) -> Path:
        return self.maf_config_dir / TUNNEL_ENV

    @property
    def maf_env(self) -> Path:
        return self.maf_config_dir / MAF_ENV


@dataclass(frozen=True)
class UnitParams:
    """Everything the two units depend on."""

    maf_command: tuple[str, ...]
    """argv prefix that runs the maf CLI, e.g. ``("/home/u/MultiAgent/.venv/bin/maf",)``."""
    mcp_host: str
    mcp_port: int
    tunnel_client: Path
    maf_config_dir: Path
    home: Path
    health_port: int = DEFAULT_HEALTH_PORT
    serve_args: tuple[str, ...] = ()
    """Extra ``maf serve`` arguments naming where settings come from (``settings_sources``), e.g. ``--config <path>``."""
    environment: tuple[tuple[str, str], ...] = ()
    """``Environment=`` pairs for maf-mcp (``MAF_BUDGET_USD``; ``settings_sources``)."""
    stop_timeout: str = MCP_STOP_TIMEOUT
    """maf-mcp ``TimeoutStopSec`` (``mcp_stop_timeout`` of the settings' ``claude_code_timeout_s``)."""

    @property
    def mcp_url(self) -> str:
        return mcp_url(self.mcp_host, self.mcp_port)

    @property
    def tunnel_upstream(self) -> str:
        """tunnel-client's ``--mcp.server-url``: the URL (whose host:port becomes the Host header) and the socket."""
        return f"url={self.mcp_url},unix-socket={MCP_SOCKET_UNIT_PATH}"

    @property
    def health_addr(self) -> str:
        return f"127.0.0.1:{self.health_port}"


def mcp_url(host: str, port: int) -> str:
    """``http://<host>:<port>/mcp`` with IPv6 literals bracketed."""
    try:
        bracket = ipaddress.ip_address(host).version == 6
    except ValueError:
        bracket = False
    return f"http://{f'[{host}]' if bracket else host}:{port}/mcp"


def default_maf_command() -> tuple[str, ...]:
    """The ``maf`` script next to the running interpreter (the project venv), else ``python -m maf.cli``."""
    script = Path(sys.executable).with_name("maf")
    if script.is_file():
        return (str(script),)
    return (sys.executable, "-m", "maf.cli")


def default_params(
    settings: Settings,
    layout: Layout,
    *,
    tunnel_client: Path | None = None,
    health_port: int = DEFAULT_HEALTH_PORT,
    maf_command: tuple[str, ...] | None = None,
    serve_args: tuple[str, ...] = (),
    environment: tuple[tuple[str, str], ...] = (),
) -> UnitParams:
    return UnitParams(
        maf_command=maf_command or default_maf_command(),
        mcp_host=settings.mcp_host,
        mcp_port=settings.mcp_port,
        tunnel_client=tunnel_client or default_tunnel_client(layout),
        maf_config_dir=layout.maf_config_dir,
        home=layout.home,
        health_port=health_port,
        serve_args=serve_args,
        environment=environment,
        stop_timeout=mcp_stop_timeout(settings.claude_code_timeout_s),
    )


def default_tunnel_client(layout: Layout) -> Path:
    return layout.home / ".local" / "bin" / "tunnel-client"


def settings_sources(
    *,
    config: Path | None,
    vault: Path | None,
    workspaces: Path | None,
    environ: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """``(serve_args, environment)`` that make the unit's ``maf serve`` read the settings this setup read: the
    ``--config``/``--vault``/``--workspaces`` options, else ``MAF_CONFIG``/``MAF_VAULT``/``MAF_WORKSPACES`` (as absolute
    paths), and ``MAF_BUDGET_USD``. Nothing when none is set, so the default unit names no config at all."""
    args: list[str] = []
    for option, given, variable in (("--config", config, "MAF_CONFIG"), ("--vault", vault, "MAF_VAULT"),
                                    ("--workspaces", workspaces, "MAF_WORKSPACES")):
        value = given if given is not None else (environ.get(variable, "").strip() or None)
        if value is not None:
            args += [option, str(Path(value).expanduser().resolve())]
    environment: list[tuple[str, str]] = []
    for variable in MAF_ENV_VARIABLES:
        value = environ.get(variable, "").strip()
        if value:
            try:
                environment.append((variable, f"{float(value):g}"))
            except ValueError:
                raise ValueError(f"{variable} must be a number, got {value!r}") from None
    return tuple(args), tuple(environment)


def contrib_params() -> UnitParams:
    """The default layout ``contrib/systemd/`` is rendered for: repo at ``~/MultiAgent``, port 8765."""
    home = CONTRIB_HOME
    return UnitParams(
        maf_command=(str(home / "MultiAgent" / ".venv" / "bin" / "maf"),),
        mcp_host="127.0.0.1",
        mcp_port=8765,
        tunnel_client=home / ".local" / "bin" / "tunnel-client",
        maf_config_dir=home / ".config" / "maf",
        home=home,
    )


def _word(value: str | Path, home: Path) -> str:
    """One unit-file word: paths under ``home`` become ``%h/...``; a leading ``%t`` (runtime directory) is kept.
    Anything that would need quoting or escaping (whitespace, ``$``, other ``%``, quotes, backslashes) is refused
    rather than escaped."""
    text = str(value)
    home_text = str(home).rstrip("/")
    if text == home_text or text.startswith(home_text + "/"):
        text = "%h" + text[len(home_text):]
    body = text[2:] if text.startswith(_SPECIFIER_PREFIXES) else text
    if not _UNIT_WORD_RE.fullmatch(text) or "%" in body:
        raise ValueError(f"cannot use {str(value)!r} in a systemd unit (no spaces, '$', '%' or quotes); "
                         "move it or write the unit by hand")
    return text


def render_mcp_unit(p: UnitParams) -> str:
    exec_start = " ".join(_word(w, p.home) for w in (*p.maf_command, "serve", *p.serve_args, "--host", p.mcp_host,
                                                    "--port", str(p.mcp_port), "--uds", MCP_SOCKET_UNIT_PATH))
    maf_env = _word(p.maf_config_dir / MAF_ENV, p.home)
    extra_env = "".join(f"Environment={_word(f'{key}={value}', p.home)}\n" for key, value in p.environment)
    return f"""\
# Generated by `maf chatgpt setup` (docs/CHATGPT.md). Customize with `systemctl --user edit {MCP_UNIT}`:
# a drop-in survives re-running setup, edits to this file do not.
[Unit]
Description=MultiAgent (maf) MCP server for the ChatGPT app, on a private Unix socket
Documentation=https://github.com/openai/tunnel-client

[Service]
Type=simple
ExecStart={exec_start}
WorkingDirectory=%h
# The MCP endpoint is a 0600 socket in this 0700 directory ($XDG_RUNTIME_DIR/{RUNTIME_SUBDIR}), not a TCP port: only
# your uid can connect, not other local users or containers on the host network. Removed when maf stops.
RuntimeDirectory={RUNTIME_SUBDIR}
RuntimeDirectoryMode=0700
# Provider keys (OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY), mode 0600: services do not read ~/.bashrc.
# A running maf keeps the keys it started with: after editing the file, `systemctl --user restart {MCP_UNIT}`.
EnvironmentFile={maf_env}
Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYTHONUNBUFFERED=1
{extra_env}# A crash restarts maf (runs it left behind are marked failed; `maf resume` continues them). A stop does not, and
# neither does exit 2 (bad configuration or arguments): the unit then stays failed with the reason in the journal.
Restart=on-failure
RestartSec=5
RestartPreventExitStatus=2
# SIGTERM reaches maf only, not its Claude Code children: maf stops serving at once and the in-flight run stops at
# its next stage boundary (marked failed, `maf resume` continues it). A stage can run three Claude Code sessions back
# to back (a session, its continuation after a timeout, a repair), each up to claude_code_timeout_s, so TimeoutStopSec
# covers that: after it systemd kills the whole group, and a session killed that way never reaches the cost ledger.
# Re-run `maf chatgpt setup` after changing claude_code_timeout_s. `systemctl --user stop --no-block` does not wait.
KillMode=mixed
TimeoutStopSec={p.stop_timeout}
# Hardening compatible with Claude Code's bubblewrap sandbox, which needs unprivileged user namespaces. Not set on
# purpose: RestrictNamespaces= (bwrap cannot create namespaces), and ProtectSystem=/ProtectHome=/PrivateTmp=/
# PrivateDevices=/ProtectKernel*= (in a user unit they imply PrivateUsers=, inside which bwrap fails too),
# MemoryDenyWriteExecute= (JIT runtimes), SystemCallArchitectures= (32-bit test binaries), PrivateNetwork=.
# NoNewPrivileges= is fine while bwrap is not setuid (`stat -c %A /usr/bin/bwrap` shows no "s"); drop it otherwise.
NoNewPrivileges=yes
RestrictSUIDSGID=yes
RestrictRealtime=yes
UMask=0077
LimitCORE=0

[Install]
WantedBy=default.target
"""


def render_tunnel_unit(p: UnitParams) -> str:
    _word(p.mcp_url, p.home)  # validates the URL part; the socket part is the fixed MCP_SOCKET_UNIT_PATH
    exec_start = " ".join(
        [_word(p.tunnel_client, p.home), "run", "--mcp.server-url", p.tunnel_upstream]
        + [_word(w, p.home) for w in ("--mcp.startup-wait-timeout", "60s", "--health.listen-addr", p.health_addr,
                                       "--log.format", "json", "--log.level", "info")]
    )
    tunnel_env = _word(p.maf_config_dir / TUNNEL_ENV, p.home)
    return f"""\
# Generated by `maf chatgpt setup` (docs/CHATGPT.md). Customize with `systemctl --user edit {TUNNEL_UNIT}`:
# a drop-in survives re-running setup, edits to this file do not.
[Unit]
Description=OpenAI Secure MCP Tunnel client for maf (outbound HTTPS to api.openai.com only)
Documentation=https://github.com/openai/tunnel-client
# Starting the tunnel starts maf serve; stopping or restarting maf serve stops or restarts the tunnel. Not BindsTo=:
# an auto-restarted maf crash should not take the tunnel down. Tunnel restarts never touch maf or its runs.
Requires={MCP_UNIT}
After={MCP_UNIT}

[Service]
Type=simple
# CONTROL_PLANE_TUNNEL_ID and CONTROL_PLANE_API_KEY (a Restricted key with only Tunnels: Read + Use), mode 0600.
# A running tunnel-client keeps the key it started with (it retries a refused one forever): after editing the file,
# `systemctl --user restart {TUNNEL_UNIT}` (runs continue).
EnvironmentFile={tunnel_env}
# tunnel-client falls back to OPENAI_API_KEY when CONTROL_PLANE_API_KEY is empty: never let it see a provider key.
UnsetEnvironment=OPENAI_API_KEY OPENAI_ADMIN_KEY GEMINI_API_KEY GOOGLE_API_KEY ANTHROPIC_API_KEY
ExecStart={exec_start}
Restart=always
RestartSec=5
TimeoutStopSec=30s
# tunnel-client is a static Go binary that needs only outbound HTTPS and loopback, so it gets everything a user unit
# can apply. PrivateDevices=, ProtectKernelModules=, ProtectKernelLogs= and ProtectClock= are left out because they
# drop capabilities, which a user manager cannot do (the unit fails with 218/CAPABILITIES); ProtectHostname= is
# ignored in a user unit.
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=read-only
PrivateTmp=yes
ProtectKernelTunables=yes
ProtectControlGroups=yes
RestrictNamespaces=yes
RestrictRealtime=yes
RestrictSUIDSGID=yes
LockPersonality=yes
MemoryDenyWriteExecute=yes
SystemCallArchitectures=native
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
UMask=0077
LimitCORE=0

[Install]
WantedBy=default.target
"""


def render_units(p: UnitParams) -> dict[str, str]:
    return {MCP_UNIT: render_mcp_unit(p), TUNNEL_UNIT: render_tunnel_unit(p)}


TUNNEL_ENV_TEMPLATE = """\
# tunnel-client settings for maf-tunnel.service (a systemd EnvironmentFile). Keep this file mode 0600.
# Fill it in with scripts/chatgpt-setup-wizard.sh, or by hand (KEY=value, no quotes, no spaces):
#   CONTROL_PLANE_TUNNEL_ID  from https://platform.openai.com/settings/organization/tunnels (tunnel_ + 32 hex)
#   CONTROL_PLANE_API_KEY    a Restricted key with only Tunnels: Read + Use; never an "All" key or an admin key
CONTROL_PLANE_TUNNEL_ID=
CONTROL_PLANE_API_KEY=
"""

MAF_ENV_TEMPLATE = """\
# Provider API keys for maf-mcp.service (a systemd EnvironmentFile). Keep this file mode 0600.
# systemd services do not read ~/.bashrc, so the keys your shell exports must be repeated here (KEY=value).
OPENAI_API_KEY=
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
"""


# --------------------------------------------------------------------------- env files and redaction


def read_env_file(path: Path) -> dict[str, str]:
    """``KEY=value`` lines of a systemd EnvironmentFile (comments and blank lines skipped, one level of matching
    quotes removed). Missing file: ``{}``."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def secret_values(layout: Layout, environ: Mapping[str, str] | None = None) -> set[str]:
    """Values ``status`` must never print: every key in both env files plus the API keys of this shell."""
    found = {v for k, v in read_env_file(layout.tunnel_env).items() if k.endswith("_KEY")}
    found |= {v for k, v in read_env_file(layout.maf_env).items() if k.endswith("_KEY")}
    return {v for v in found if len(v) >= 8} | known_secrets(environ)


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Mask the exact ``secrets`` and anything shaped like an OpenAI/Anthropic/Google key or a bearer token
    (``maf.redact.redact`` with ``LOG_PATTERNS``: this masks log lines)."""
    return _redact(text, secrets, patterns=LOG_PATTERNS)


def without_keys(argv: Sequence[str]) -> list[str]:
    """``argv`` run through ``env -u`` for every ``KEY_VARIABLES`` entry."""
    unset = [arg for key in KEY_VARIABLES for arg in ("-u", key)]
    return ["env", *unset, *argv]


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _write_private(path: Path, text: str) -> None:
    """Create ``path`` as 0600 (fails if it exists: templates never replace a filled-in file)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def _display(path: Path, home: Path) -> str:
    try:
        return "~/" + str(path.relative_to(home))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- setup


def setup(
    settings: Settings,
    *,
    layout: Layout,
    params: UnitParams,
    reload: bool = True,
    run: Runner = run_command,
    out: TextIO | None = None,
) -> int:
    """Write the env templates (if missing), ``mcp_inbox`` (if missing) and the units, then ``daemon-reload``. Returns
    an exit code. ``ValueError`` before anything is written for a non-loopback host, a relative tunnel-client path,
    a health port equal to ``mcp_port``, or a path a unit cannot hold."""
    from maf.mcp_server import check_loopback

    out = sys.stdout if out is None else out  # resolved per call: a default bound at import goes stale under capture
    check_loopback(params.mcp_host)  # ValueError -> usage error in the CLI
    if not params.tunnel_client.is_absolute():
        raise ValueError(f"--tunnel-client must be an absolute path, got {str(params.tunnel_client)!r}")
    if params.health_port == params.mcp_port:
        raise ValueError(f"the tunnel-client health port {params.health_port} is mcp_port: pick another --health-port")
    if dropped := dropped_sources(layout, params):
        raise ValueError(
            f"the installed {MCP_UNIT} runs with {', '.join(dropped)}, which this setup was not given: pass the same "
            "(global options before `chatgpt`, or MAF_* variables), or other values to change them"
        )
    units = render_units(params)  # may raise ValueError before anything is written

    created = not layout.maf_config_dir.exists()
    layout.maf_config_dir.mkdir(parents=True, exist_ok=True)
    if created:
        layout.maf_config_dir.chmod(0o700)
    for path, template, purpose in (
        (layout.tunnel_env, TUNNEL_ENV_TEMPLATE, "tunnel id + runtime key"),
        (layout.maf_env, MAF_ENV_TEMPLATE, "provider keys"),
    ):
        shown = _display(path, layout.home)
        if path.exists():
            if _mode(path) & 0o077:
                path.chmod(0o600)
                print(f"kept {shown} ({purpose}); mode tightened to 0600", file=out)
            else:
                print(f"kept {shown} ({purpose})", file=out)
        else:
            _write_private(path, template)
            print(f"wrote {shown} (0600 template for the {purpose}; fill it in)", file=out)

    inbox = settings.mcp_inbox
    if inbox is not None and not inbox.exists():
        inbox.mkdir(mode=0o700, parents=True)
        print(f"created {_display(inbox, layout.home)} (0700) for start_run input files (mcp_inbox)", file=out)

    layout.unit_dir.mkdir(parents=True, exist_ok=True)
    updated: list[str] = []
    for name, text in units.items():
        path = layout.unit_dir / name
        shown = _display(path, layout.home)
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            print(f"unchanged {shown}", file=out)
            continue
        if path.exists():
            updated.append(name)
        verb = "updated" if path.exists() else "installed"
        tmp = path.with_name(f"{name}.new")  # not a unit suffix, so systemd never loads it
        tmp.write_text(text, encoding="utf-8")
        tmp.chmod(0o644)
        tmp.replace(path)
        print(f"{verb} {shown}", file=out)

    status = 0
    if reload:
        result = run(["systemctl", "--user", "daemon-reload"], 30.0)
        if result.returncode != 0:
            print(f"systemctl --user daemon-reload failed: {(result.stderr or result.stdout).strip()}", file=out)
            status = 1
        else:
            print("systemctl --user daemon-reload: ok", file=out)
    _restart_hint(updated, reload=reload, run=run, out=out)

    print("", file=out)
    if not params.tunnel_client.is_file():
        print(f"tunnel-client not found at {params.tunnel_client}: run scripts/install-tunnel-client.sh", file=out)
    print("next: scripts/chatgpt-setup-wizard.sh walks through the manual steps; by hand (docs/CHATGPT.md):", file=out)
    print("  1. create the tunnel and its Restricted runtime key in the OpenAI Platform, then fill in "
          f"{_display(layout.tunnel_env, layout.home)} and {_display(layout.maf_env, layout.home)}", file=out)
    print(f"  2. systemctl --user enable {MCP_UNIT} {TUNNEL_UNIT} && systemctl --user restart {MCP_UNIT} {TUNNEL_UNIT}",
          file=out)
    print("     (restart, not start: a running unit keeps the env files it started with)", file=out)
    print("  3. loginctl enable-linger  (starts the units at boot without a login, and keeps them running after you "
          "log out)", file=out)
    print("  4. maf chatgpt status", file=out)
    return status


def _restart_hint(updated: Sequence[str], *, reload: bool, run: Runner, out: TextIO) -> None:
    """Tell the user that changed units keep running their old definition until restarted. Restarting maf-mcp
    restarts the tunnel too (``Requires=``); restarting only the tunnel never touches maf or its runs."""
    if not updated:
        return
    target = MCP_UNIT if MCP_UNIT in updated else TUNNEL_UNIT
    effect = ("restarts the tunnel too; an in-flight run stops at its next stage boundary, `maf resume` continues it"
              if target == MCP_UNIT else "runs continue")
    if not reload:
        print(f"if the units are running, apply the change: systemctl --user restart {target} ({effect})", file=out)
        return
    active = [name for name in updated if run(["systemctl", "--user", "is-active", name], 10.0).stdout.strip() == "active"]
    if active:
        print(f"{', '.join(active)} still run(s) the old unit: systemctl --user restart {target} ({effect})", file=out)


# --------------------------------------------------------------------------- status


def mcp_health(url: str, uds: Path | None = None, *, timeout: float = 10.0) -> tuple[bool, str]:
    """Local MCP round trip: initialize (the handshake-era protocol) + tools/list, to ``url`` or, with ``uds``, over
    that Unix socket (``url`` then only sets the Host header). Proxies from the environment are ignored (loopback).
    Returns ``(ok, one-line detail)``."""
    import anyio
    import httpx2
    import mcp
    from mcp.client.streamable_http import streamable_http_client

    async def probe() -> str:
        transport = httpx2.AsyncHTTPTransport(uds=str(uds)) if uds is not None else None
        with anyio.fail_after(timeout):
            async with httpx2.AsyncClient(trust_env=False, timeout=httpx2.Timeout(timeout), transport=transport) as http:
                async with mcp.Client(streamable_http_client(url, http_client=http), mode="legacy") as client:
                    tools = sorted(t.name for t in (await client.list_tools()).tools)
                    info = client.server_info
                    name = " ".join(x for x in (info.name, info.version) if x) if info is not None else "unnamed"
                    return f"{name}, protocol {client.protocol_version}, tools: {', '.join(tools) or 'none'}"

    started = time.monotonic()
    try:
        detail = anyio.run(probe)
    except BaseException as exc:  # noqa: BLE001 - report any failure (task groups raise ExceptionGroup)
        if isinstance(exc, KeyboardInterrupt):
            raise
        return False, _describe_exception(exc)
    return True, f"initialize + tools/list in {time.monotonic() - started:.2f} s: {detail}"


def _describe_exception(exc: BaseException) -> str:
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    text = " ".join(str(exc).split()) or "no details"
    if isinstance(exc, TimeoutError):
        text = "timed out"
    return f"{type(exc).__name__}: {text}"[:300]


def http_get(url: str, timeout: float = 5.0) -> tuple[int | None, str]:
    """GET a loopback URL without proxies: ``(status or None, first 200 characters of the body or the error)``."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as response:  # noqa: S310 - fixed loopback http URL
            return response.status, " ".join(response.read(2000).decode("utf-8", "replace").split())[:200]
    except urllib.error.HTTPError as exc:
        return exc.code, " ".join(exc.read(2000).decode("utf-8", "replace").split())[:200]
    except (urllib.error.URLError, OSError) as exc:
        return None, str(getattr(exc, "reason", exc))


def unit_state(unit: str, run: Runner) -> dict[str, str]:
    """``systemctl --user show`` properties; ``{"error": ...}`` if systemctl fails."""
    result = run(["systemctl", "--user", "show", unit, "--property=LoadState,ActiveState,SubState,UnitFileState,"
                  "NRestarts,Result"], 10.0)
    if result.returncode != 0:
        return {"error": (result.stderr or result.stdout).strip() or f"systemctl exit {result.returncode}"}
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def describe_unit(state: Mapping[str, str]) -> str:
    if "error" in state:
        return f"unknown ({state['error']})"
    if state.get("LoadState") == "not-found":
        return "not installed (maf chatgpt setup)"
    text = f"{state.get('ActiveState', '?')} ({state.get('SubState', '?')}), {state.get('UnitFileState') or 'static'}"
    restarts = state.get("NRestarts", "0")
    if restarts not in ("", "0"):
        text += f", {restarts} restart(s)"
    if state.get("Result") not in (None, "", "success"):
        text += f", last result {state['Result']}"
    return text


def journal(unit: str, lines: int, run: Runner) -> list[str]:
    result = run(["journalctl", "--user", "-u", unit, "-n", str(lines), "--no-pager", "-o", "short-iso"], 15.0)
    text = result.stdout if result.returncode == 0 else (result.stderr or result.stdout)
    return [line for line in text.splitlines() if line.strip() and not line.startswith("-- No entries --")]


def _exec_start(unit_path: Path) -> list[str]:
    """The words of an installed unit's ``ExecStart=``, or ``[]``."""
    try:
        text = unit_path.read_text(encoding="utf-8")
    except OSError:
        return []
    line = re.search(r"^ExecStart=(.*)$", text, re.MULTILINE)
    return line.group(1).split() if line else []


def _option(words: Sequence[str], name: str) -> str | None:
    for i, word in enumerate(words):
        if word == name and i + 1 < len(words):
            return words[i + 1]
        if word.startswith(name + "="):
            return word[len(name) + 1:]
    return None


def parse_upstream(value: str) -> tuple[str, str | None]:
    """``(url, unix socket or None)`` of a tunnel-client ``--mcp.server-url`` value: a bare URL or
    ``url=...,unix-socket=...[,...]``."""
    if not value.startswith(("url=", "unix-socket=", "channel=")):
        return value, None
    parts = dict(part.split("=", 1) for part in value.split(",") if "=" in part)
    return parts.get("url", ""), parts.get("unix-socket")


def installed_tunnel_target(unit_path: Path) -> tuple[str | None, str | None]:
    """``(--mcp.server-url, --health.listen-addr)`` from an installed maf-tunnel.service, or Nones. The server URL is
    the whole value (``url=...,unix-socket=...`` included; ``parse_upstream`` splits it)."""
    words = _exec_start(unit_path)
    return _option(words, "--mcp.server-url"), _option(words, "--health.listen-addr")


def installed_tunnel_options(layout: Layout) -> tuple[Path | None, int | None]:
    """``(tunnel-client binary, health port)`` of the installed maf-tunnel.service, so re-running setup keeps them."""
    words = _exec_start(layout.unit_dir / TUNNEL_UNIT)
    client = layout.expand(words[0]) if words else None
    _url, health = installed_tunnel_target(layout.unit_dir / TUNNEL_UNIT)
    port = None
    if health and (match := re.fullmatch(r".*:([0-9]{1,5})", health)):
        port = int(match.group(1))
    return client, port


SOURCE_OPTIONS = ("--config", "--vault", "--workspaces")


def installed_serve_sources(layout: Layout) -> dict[str, str]:
    """The settings sources the installed maf-mcp.service runs with: ``{"--config": path, ...,
    "MAF_BUDGET_USD": value}`` (paths with ``%h`` expanded), or ``{}``."""
    unit = layout.unit_dir / MCP_UNIT
    words = _exec_start(unit)
    found = {opt: str(layout.expand(value)) for opt in SOURCE_OPTIONS if (value := _option(words, opt)) is not None}
    try:
        text = unit.read_text(encoding="utf-8")
    except OSError:
        return found
    for variable in MAF_ENV_VARIABLES:
        if match := re.search(rf"^Environment={variable}=(\S+)$", text, re.MULTILINE):
            found[variable] = match.group(1)
    return found


def dropped_sources(layout: Layout, params: UnitParams) -> list[str]:
    """Settings sources of the installed maf-mcp.service that ``params`` leave out. Re-rendering without them would
    silently point the service at other settings (another config, vault or budget)."""
    given = {params.serve_args[i]: params.serve_args[i + 1] for i in range(0, len(params.serve_args) - 1, 2)}
    given |= dict(params.environment)
    return [f"{key} {value}" if key.startswith("--") else f"{key}={value}"
            for key, value in installed_serve_sources(layout).items() if key not in given]


def installed_mcp_socket(layout: Layout) -> tuple[str | None, Path | None]:
    """``(--uds word, expanded path)`` of the installed maf-mcp.service; ``(None, None)`` if it serves TCP or is absent."""
    word = _option(_exec_start(layout.unit_dir / MCP_UNIT), "--uds")
    return (word, layout.expand(word)) if word else (None, None)


_MISSING_KEY_RE = re.compile(r"\b([A-Z][A-Z0-9_]*_API_KEY) is not set")


def log_hints(mcp_lines: Iterable[str], tunnel_lines: Iterable[str], port: int) -> list[str]:
    """Fixes for recognizable failures in the journal lines."""
    hints: list[str] = []
    mcp_text = "\n".join(mcp_lines)
    for origin in dict.fromkeys(re.findall(r"Invalid Origin header: (\S+)", mcp_text)):
        hints.append(f"maf refused Origin {origin}: if that is the tunnel's traffic, add `mcp_allowed_origins: "
                     f"[\"{origin}\"]` to maf's config.yaml, then systemctl --user restart {MCP_UNIT}")
    for host in dict.fromkeys(re.findall(r"Invalid Host header: (\S+)", mcp_text)):
        hints.append(f"maf refused Host {host}: the tunnel's MCP URL must be http://127.0.0.1:{port}/mcp "
                     "(or localhost with that port)")
    for key in dict.fromkeys(_MISSING_KEY_RE.findall(mcp_text)):
        hints.append(f"maf serve runs without {key}: fill it in ~/.config/maf/maf.env, then systemctl --user restart "
                     f"{MCP_UNIT} (a running maf keeps the environment it started with)")
    tunnel_text = "\n".join(tunnel_lines).lower()
    if re.search(r"\b(401|403)\b|unauthori[sz]ed|forbidden|permission", tunnel_text):
        hints.append("the tunnel's control-plane calls are refused: use a Restricted key with Tunnels Read + Use in "
                     "the tunnel's organization (new role grants can take 30 minutes), and check CONTROL_PLANE_TUNNEL_ID; "
                     f"after editing ~/.config/maf/tunnel.env: systemctl --user restart {TUNNEL_UNIT} (a running "
                     "tunnel-client keeps retrying the key it started with)")
    return hints


def status(
    settings: Settings,
    *,
    layout: Layout,
    tunnel_client: Path | None = None,
    lines: int = 10,
    run: Runner = run_command,
    probe: Callable[[str, Path | None], tuple[bool, str]] = mcp_health,
    get: Callable[[str], tuple[int | None, str]] = http_get,
    out: TextIO | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Print the state of both units and endpoints. Exit 0 when nothing is wrong: maf answers MCP (over the installed
    unit's socket, else on its TCP port), every key is set, the tunnel unit is active, tunnel-client is ready and has
    polled the control plane successfully. Else 1, with the reasons on the last line."""
    out = sys.stdout if out is None else out
    secrets = secret_values(layout, environ)
    tunnel_client = tunnel_client or installed_tunnel_options(layout)[0] or default_tunnel_client(layout)
    url = mcp_url(settings.mcp_host, settings.mcp_port)
    socket_word, socket_path = installed_mcp_socket(layout)
    problems: list[str] = []

    def say(text: str = "") -> None:
        print(redact(text, secrets), file=out)

    def show_env(path: Path, keys: Sequence[str]) -> None:
        shown = _display(path, layout.home)
        if not path.exists():
            say(f"  env {shown}: missing (maf chatgpt setup)")
            problems.append(f"{shown} missing")
            return
        values = read_env_file(path)
        mode = _mode(path)
        parts = []
        for key in keys:
            value = values.get(key, "")
            if key == "CONTROL_PLANE_TUNNEL_ID" and value:
                ok = TUNNEL_ID_RE.fullmatch(value) is not None
                parts.append(f"{key} {value[:11]}...{value[-4:]}" + ("" if ok else " (not tunnel_ + 32 hex!)"))
            else:
                parts.append(f"{key} {'set' if value else 'missing'}")
            if not value:
                problems.append(f"{key} missing in {shown}")
        warn = "" if not mode & 0o077 else " (too open: chmod 600)"
        say(f"  env {shown} ({mode:04o}{warn}): " + ", ".join(parts))

    def show_journal(unit: str) -> list[str]:
        entries = journal(unit, lines, run)
        if entries:
            say(f"  last {len(entries)} journal line(s):")
            for entry in entries:
                say(f"    {entry}")
        return entries

    say(f"maf serve (MCP endpoint {url}" + (f" on the unix socket {socket_path})" if socket_path else ")"))
    mcp_state = unit_state(MCP_UNIT, run)
    say(f"  unit {MCP_UNIT}: {describe_unit(mcp_state)}")
    ok, detail = probe(url, socket_path)
    say(f"  MCP: {'ok' if ok else 'FAILED'}: {detail}")
    if not ok:
        problems.append("maf serve does not answer MCP")
    show_env(layout.maf_env, MAF_ENV_KEYS)
    mcp_lines = show_journal(MCP_UNIT)

    say()
    say("tunnel-client (OpenAI Secure MCP Tunnel)")
    if tunnel_client.is_file():
        version = run(without_keys([str(tunnel_client), "--version"]), 10.0)
        say(f"  binary {_display(tunnel_client, layout.home)}: {(version.stdout or version.stderr).strip()[:80]}")
    else:
        say(f"  binary {_display(tunnel_client, layout.home)}: missing (scripts/install-tunnel-client.sh)")
        problems.append("tunnel-client not installed")
    tunnel_state = unit_state(TUNNEL_UNIT, run)
    say(f"  unit {TUNNEL_UNIT}: {describe_unit(tunnel_state)}")
    if tunnel_state.get("ActiveState") != "active":
        problems.append(f"{TUNNEL_UNIT} is not active")
    target, health_addr = installed_tunnel_target(layout.unit_dir / TUNNEL_UNIT)
    if target is not None:
        target_url, target_socket = parse_upstream(target)
        if target_url != url or target_socket != socket_word:
            where = target_url + (f" via {target_socket}" if target_socket else "")
            here = url + (f" via {socket_word}" if socket_word else "")
            say(f"  WARNING: the tunnel forwards to {where}, but maf serve is configured for {here}: re-run "
                f"maf chatgpt setup, then systemctl --user restart {MCP_UNIT}")
            problems.append("tunnel target differs from maf serve")
    health_addr = health_addr or f"127.0.0.1:{DEFAULT_HEALTH_PORT}"
    code, body = get(f"http://{health_addr}/readyz")
    say(f"  readyz http://{health_addr}/readyz: {code if code is not None else 'unreachable'} {body}".rstrip())
    if code != 200:
        problems.append("tunnel-client is not ready")
    elif tunnel_client.is_file():
        poll = run(without_keys([str(tunnel_client), "health", "--url", f"http://{health_addr}",
                                 "--require-control-plane-poll"]), 15.0)
        report = [ln.strip() for ln in (poll.stdout + poll.stderr).splitlines() if ln.strip()]
        line = next((ln for ln in report if ln.startswith("Control-plane poll")), report[-1] if report else "")
        say(f"  control-plane poll: {'ok' if poll.returncode == 0 else 'NOT confirmed'} ({line[:160]})")
        if poll.returncode != 0:
            problems.append("no successful control-plane poll yet (key, tunnel id or permissions)")
    say(f"  admin UI: http://{health_addr}/ui")
    show_env(layout.tunnel_env, TUNNEL_ENV_KEYS)
    tunnel_lines = show_journal(TUNNEL_UNIT)

    say()
    user = (os.environ if environ is None else environ).get("USER", "")
    linger = run(["loginctl", "show-user", user, "--property=Linger", "--value"], 10.0) if user else None
    value = linger.stdout.strip() if linger is not None and linger.returncode == 0 else ""
    if value in ("yes", "no"):
        say(f"linger: {value}" + ("" if value == "yes" else " (the units start only once you log in, and stop when you "
                                                          "log out: loginctl enable-linger)"))
    for hint in log_hints(mcp_lines, tunnel_lines, settings.mcp_port):
        say(f"hint: {hint}")
    if problems:
        say("not ready: " + "; ".join(dict.fromkeys(problems)))
        return 1
    say("ready: ChatGPT can reach maf through the tunnel")
    return 0
