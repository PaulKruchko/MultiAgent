"""``maf chatgpt`` tests: unit rendering (and the contrib copies), setup, status with scripted system tools, redaction,
and a real loopback MCP round trip. No systemd, no network beyond 127.0.0.1."""

from __future__ import annotations

import io
import os
import re
import socket
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from maf import __version__, chatgpt, config, mcp_server
from maf.chatgpt import Layout, UnitParams
from maf.config import Settings
from maf.mcp_server import RunManager, build_server
from maf.pipeline import Pipeline
from maf.stages.base import StageContext, StageOutput
from maf.types import STAGE_ORDER

CONTRIB = Path(__file__).parents[1] / "contrib" / "systemd"
TUNNEL_ID = "tunnel_0123456789abcdef0123456789abcdef"


@pytest.fixture
def layout(tmp_path: Path) -> Layout:
    home = tmp_path / "home"
    home.mkdir()
    return Layout.default(environ={}, home=home)


def _params(layout: Layout, **kw: Any) -> UnitParams:
    base: dict[str, Any] = dict(
        maf_command=(str(layout.home / "MultiAgent" / ".venv" / "bin" / "maf"),),
        mcp_host="127.0.0.1",
        mcp_port=8765,
        tunnel_client=layout.home / ".local" / "bin" / "tunnel-client",
        maf_config_dir=layout.maf_config_dir,
        home=layout.home,
    )
    base.update(kw)
    return UnitParams(**base)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _directives(unit: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in unit.splitlines() if "=" in line and not line.startswith("#"))


# --------------------------------------------------------------------------- units


def test_contrib_units_are_the_default_rendering() -> None:
    for name, text in chatgpt.render_units(chatgpt.contrib_params()).items():
        assert (CONTRIB / name).read_text(encoding="utf-8") == text, f"re-render contrib/systemd/{name}"


def test_mcp_stop_timeout_covers_the_longest_stage(layout: Layout) -> None:
    """A stop must not SIGKILL a Claude Code session: its spend would never reach the ledger. Three sessions (one, its
    continuation after a timeout, a repair) plus a margin: 5h at the 90-minute default, and it follows the setting."""
    assert chatgpt.MCP_STOP_TIMEOUT == chatgpt.mcp_stop_timeout(5400.0) == "5h"
    assert chatgpt.mcp_stop_timeout(3600.0) == "210min" and chatgpt.mcp_stop_timeout(100.0) == "35min"
    settings = Settings(claude_code_timeout_s=7200.0)
    params = chatgpt.default_params(settings, layout, maf_command=("/usr/bin/maf",))
    assert _directives(chatgpt.render_mcp_unit(params))["TimeoutStopSec"] == "390min"


def test_mcp_unit(layout: Layout) -> None:
    unit = _directives(chatgpt.render_mcp_unit(_params(layout, mcp_port=9001)))
    assert unit["ExecStart"] == "%h/MultiAgent/.venv/bin/maf serve --host 127.0.0.1 --port 9001 --uds %t/maf/mcp.sock"
    assert (unit["RuntimeDirectory"], unit["RuntimeDirectoryMode"]) == ("maf", "0700")
    assert unit["EnvironmentFile"] == "%h/.config/maf/maf.env"
    assert (unit["Restart"], unit["KillMode"], unit["TimeoutStopSec"]) == ("on-failure", "mixed", "5h")
    assert unit["RestartPreventExitStatus"] == "2"  # a config or usage error must not restart every 5 s forever
    assert (unit["NoNewPrivileges"], unit["UMask"], unit["LimitCORE"]) == ("yes", "0077", "0")
    # These break Claude Code's bubblewrap sandbox in a user unit (verified with bwrap under systemd-run --user).
    for directive in ("RestrictNamespaces", "PrivateUsers", "ProtectSystem", "ProtectHome", "PrivateTmp",
                      "PrivateDevices", "MemoryDenyWriteExecute", "SystemCallArchitectures", "PrivateNetwork"):
        assert directive not in unit, directive


def test_tunnel_unit(layout: Layout) -> None:
    unit = _directives(chatgpt.render_tunnel_unit(_params(layout, mcp_port=9001, health_port=9002)))
    assert unit["ExecStart"].split() == [
        "%h/.local/bin/tunnel-client", "run", "--mcp.server-url",
        "url=http://127.0.0.1:9001/mcp,unix-socket=%t/maf/mcp.sock",
        "--mcp.startup-wait-timeout", "60s", "--health.listen-addr", "127.0.0.1:9002",
        "--log.format", "json", "--log.level", "info",
    ]
    assert (unit["Requires"], unit["After"]) == ("maf-mcp.service", "maf-mcp.service")
    assert "BindsTo" not in unit
    assert unit["EnvironmentFile"] == "%h/.config/maf/tunnel.env"
    assert {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"} <= set(unit["UnsetEnvironment"].split())
    for directive in ("NoNewPrivileges", "ProtectSystem", "RestrictNamespaces", "MemoryDenyWriteExecute"):
        assert directive in unit
    # These drop capabilities, which a user manager cannot do (the unit would fail with 218/CAPABILITIES).
    for directive in ("PrivateDevices", "ProtectKernelModules", "ProtectKernelLogs", "ProtectClock"):
        assert directive not in unit


def test_units_use_ipv6_brackets_and_absolute_paths_outside_home(layout: Layout) -> None:
    params = _params(layout, mcp_host="::1", maf_command=("/opt/maf/bin/maf",))
    assert "--mcp.server-url url=http://[::1]:8765/mcp,unix-socket=%t/maf/mcp.sock" in chatgpt.render_tunnel_unit(params)
    assert "ExecStart=/opt/maf/bin/maf serve --host ::1 --port 8765 --uds" in chatgpt.render_mcp_unit(params)


def test_mcp_unit_carries_the_settings_sources(layout: Layout, tmp_path: Path) -> None:
    """The service must read the config, vault and workspaces setup was given, not silently the defaults."""
    config = tmp_path / "c.yaml"
    args, environment = chatgpt.settings_sources(
        config=config, vault=None, workspaces=None,
        environ={"MAF_VAULT": str(tmp_path / "v"), "MAF_WORKSPACES": "", "MAF_BUDGET_USD": "7.50"},
    )
    assert args == ("--config", str(config), "--vault", str(tmp_path / "v"))
    assert environment == (("MAF_BUDGET_USD", "7.5"),)
    unit = _directives(chatgpt.render_mcp_unit(_params(layout, serve_args=args, environment=environment)))
    assert unit["ExecStart"].split()[1:5] == ["serve", "--config", str(config), "--vault"]
    assert unit["Environment"] == "MAF_BUDGET_USD=7.5"  # the last Environment= line
    assert chatgpt.settings_sources(config=None, vault=None, workspaces=None, environ={}) == ((), ())
    with pytest.raises(ValueError, match="MAF_BUDGET_USD"):
        chatgpt.settings_sources(config=None, vault=None, workspaces=None, environ={"MAF_BUDGET_USD": "lots"})


def _installed_copy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """maf as a non-editable install: no source checkout, so its default workspaces and inbox follow XDG_DATA_HOME."""
    package = tmp_path / "lib" / "site-packages" / "maf"
    package.mkdir(parents=True)
    monkeypatch.setattr(config, "PACKAGE_DIR", package)
    assert config.source_checkout() is None


def _unit_environment(unit: str) -> dict[str, str]:
    return dict(line.removeprefix("Environment=").split("=", 1)
                for line in unit.splitlines() if line.startswith("Environment="))


def test_an_installed_copy_passes_its_xdg_data_home_to_the_unit(
    layout: Layout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The systemd user manager has no XDG_DATA_HOME: without it in the unit, the service of an installed copy would use
    ~/.local/share/maf while setup created the inbox (and the CLI keeps workspaces) under the shell's value."""
    _installed_copy(monkeypatch, tmp_path)
    data = tmp_path / "xdg"
    monkeypatch.setenv("XDG_DATA_HOME", f"{data}/")
    settings = Settings(vault_path=tmp_path / "vault")
    assert settings.mcp_inbox == data / "maf" / "inbox"
    args, environment = chatgpt.settings_sources(config=None, vault=None, workspaces=None, environ=os.environ)
    assert (args, environment) == ((), (("XDG_DATA_HOME", str(data)),))
    params = chatgpt.default_params(settings, layout, maf_command=("/opt/maf/bin/maf",), serve_args=args,
                                    environment=environment)
    unit_env = _unit_environment(chatgpt.render_mcp_unit(params))
    assert unit_env["XDG_DATA_HOME"] == str(data)
    # What the service computes from the unit's environment alone is what setup and the CLI use.
    assert config.data_home(unit_env) / "inbox" == settings.mcp_inbox
    assert config.data_home(unit_env) / "workspaces" == settings.workspaces_path

    assert chatgpt.setup(settings, layout=layout, params=params, reload=False, out=io.StringIO()) == 0
    assert settings.mcp_inbox is not None and settings.mcp_inbox.is_dir()
    assert chatgpt.installed_serve_sources(layout) == {"XDG_DATA_HOME": str(data)}
    before = (layout.unit_dir / chatgpt.MCP_UNIT).read_text()

    monkeypatch.delenv("XDG_DATA_HOME")  # another shell, or a script, without it: refused, not silently moved
    rerun = chatgpt.default_params(Settings(vault_path=tmp_path / "vault"), layout, maf_command=("/opt/maf/bin/maf",),
                                   environment=chatgpt.settings_sources(config=None, vault=None, workspaces=None,
                                                                        environ=os.environ)[1])
    assert rerun.environment == ()
    with pytest.raises(ValueError, match=rf"runs with XDG_DATA_HOME={re.escape(str(data))}, which this setup was not "
                                         r"given: .*MAF_\* and XDG_DATA_HOME variables"):
        chatgpt.setup(settings, layout=layout, params=rerun, reload=False, out=io.StringIO())
    assert (layout.unit_dir / chatgpt.MCP_UNIT).read_text() == before

    monkeypatch.setattr(config, "PACKAGE_DIR", Path(chatgpt.__file__).resolve().parent)  # setup run from a checkout
    assert chatgpt.setup(settings, layout=layout, params=rerun, reload=False, out=io.StringIO()) == 0
    assert "XDG_DATA_HOME" not in _unit_environment((layout.unit_dir / chatgpt.MCP_UNIT).read_text())


@pytest.mark.parametrize("xdg", ["", "relative/data"])
def test_xdg_data_home_reaches_the_unit_only_when_the_defaults_use_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, xdg: str
) -> None:
    """A checkout keeps its data in the checkout whatever XDG_DATA_HOME says, and the XDG spec ignores an empty or
    relative value: none of them goes into the unit."""
    sources = {"config": None, "vault": None, "workspaces": None}
    assert config.source_checkout() is not None  # the tests run from this checkout
    assert chatgpt.settings_sources(**sources, environ={"XDG_DATA_HOME": str(tmp_path)}) == ((), ())
    _installed_copy(monkeypatch, tmp_path)
    assert chatgpt.settings_sources(**sources, environ={"XDG_DATA_HOME": xdg}) == ((), ())


@pytest.mark.parametrize("bad", ["/home/u/My Projects/maf", "/opt/$HOME/maf", "/opt/100%/maf", '/opt/"q"/maf'])
def test_unit_words_that_need_escaping_are_refused(layout: Layout, bad: str) -> None:
    with pytest.raises(ValueError, match="systemd unit"):
        chatgpt.render_mcp_unit(_params(layout, maf_command=(bad,)))


def test_default_maf_command_is_the_venv_script() -> None:
    command = chatgpt.default_maf_command()
    assert command[0].endswith("/maf") or command[1:] == ("-m", "maf.cli")


def test_setup_units_follow_the_venv_maf_runs_from(
    layout: Layout, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No checkout location is assumed: the unit runs the maf next to the running interpreter, here a checkout outside
    ``~/MultiAgent``; without that script it runs ``python -m maf.cli`` with the same interpreter."""
    venv_bin = layout.home / "src" / "maf-checkout" / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    monkeypatch.setattr(sys, "executable", str(venv_bin / "python3"))
    (venv_bin / "maf").touch()
    unit = _directives(chatgpt.render_mcp_unit(chatgpt.default_params(settings, layout)))
    assert unit["ExecStart"].startswith("%h/src/maf-checkout/.venv/bin/maf serve ")
    (venv_bin / "maf").unlink()
    unit = _directives(chatgpt.render_mcp_unit(chatgpt.default_params(settings, layout)))
    assert unit["ExecStart"].startswith("%h/src/maf-checkout/.venv/bin/python3 -m maf.cli serve ")


def test_layout_respects_xdg_config_home(tmp_path: Path) -> None:
    layout = Layout.default(environ={"XDG_CONFIG_HOME": str(tmp_path / "xdg")}, home=tmp_path)
    assert layout.unit_dir == tmp_path / "xdg" / "systemd" / "user"
    assert layout.maf_config_dir == tmp_path / ".config" / "maf"  # maf's own config dir stays put
    assert Layout.default(environ={"XDG_CONFIG_HOME": "relative"}, home=tmp_path).unit_dir == (
        tmp_path / ".config" / "systemd" / "user"
    )


# --------------------------------------------------------------------------- setup


class Recorder:
    """Scripted ``Runner``: records argv, answers by the first matching prefix."""

    def __init__(self, answers: dict[str, tuple[int, str]] | None = None) -> None:
        self.calls: list[list[str]] = []
        self.answers = answers or {}

    def __call__(self, argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
        argv = list(argv)
        self.calls.append(argv)
        text = " ".join(argv)
        for prefix, (code, out) in self.answers.items():
            if prefix in text:
                return subprocess.CompletedProcess(argv, code, out, "" if code == 0 else out)
        return subprocess.CompletedProcess(argv, 0, "", "")


def test_setup_writes_private_templates_units_and_reloads(layout: Layout, settings: Settings) -> None:
    run, out = Recorder(), io.StringIO()
    assert chatgpt.setup(settings, layout=layout, params=_params(layout), run=run, out=out) == 0
    assert _mode(layout.maf_config_dir) == 0o700
    for path, keys in ((layout.tunnel_env, chatgpt.TUNNEL_ENV_KEYS), (layout.maf_env, chatgpt.MAF_ENV_KEYS)):
        assert _mode(path) == 0o600
        assert chatgpt.read_env_file(path) == dict.fromkeys(keys, "")
    for name in chatgpt.UNITS:
        assert (layout.unit_dir / name).read_text(encoding="utf-8") == chatgpt.render_units(_params(layout))[name]
    assert run.calls == [["systemctl", "--user", "daemon-reload"]]
    assert "tunnel-client not found" in out.getvalue()


def test_setup_is_idempotent_and_never_overwrites_env_files(layout: Layout, settings: Settings) -> None:
    chatgpt.setup(settings, layout=layout, params=_params(layout), reload=False, out=io.StringIO())
    layout.tunnel_env.write_text(f"CONTROL_PLANE_TUNNEL_ID={TUNNEL_ID}\nCONTROL_PLANE_API_KEY=sk-kept\n")
    layout.tunnel_env.chmod(0o644)
    out = io.StringIO()
    assert chatgpt.setup(settings, layout=layout, params=_params(layout), reload=False, out=out) == 0
    assert chatgpt.read_env_file(layout.tunnel_env)["CONTROL_PLANE_API_KEY"] == "sk-kept"
    assert _mode(layout.tunnel_env) == 0o600
    text = out.getvalue()
    assert "mode tightened to 0600" in text and "unchanged ~/.config/systemd/user/maf-mcp.service" in text
    assert "sk-kept" not in text


def test_setup_updates_units_when_the_port_changes(layout: Layout, settings: Settings) -> None:
    chatgpt.setup(settings, layout=layout, params=_params(layout), reload=False, out=io.StringIO())
    out = io.StringIO()
    chatgpt.setup(settings, layout=layout, params=_params(layout, mcp_port=9100), reload=False, out=out)
    assert "updated ~/.config/systemd/user/maf-tunnel.service" in out.getvalue()
    assert "if the units are running, apply the change: systemctl --user restart maf-mcp.service" in out.getvalue()
    assert "http://127.0.0.1:9100/mcp" in (layout.unit_dir / chatgpt.TUNNEL_UNIT).read_text()
    assert sorted(p.name for p in layout.unit_dir.iterdir()) == sorted(chatgpt.UNITS)


def test_setup_says_when_an_active_unit_runs_the_old_definition(layout: Layout, settings: Settings) -> None:
    chatgpt.setup(settings, layout=layout, params=_params(layout), reload=False, out=io.StringIO())
    run, out = Recorder({"is-active maf-mcp": (0, "active\n"), "is-active maf-tunnel": (3, "inactive\n")}), io.StringIO()
    assert chatgpt.setup(settings, layout=layout, params=_params(layout, mcp_port=9100), run=run, out=out) == 0
    assert "maf-mcp.service still run(s) the old unit: systemctl --user restart maf-mcp.service" in out.getvalue()
    unchanged, out = Recorder(), io.StringIO()
    chatgpt.setup(settings, layout=layout, params=_params(layout, mcp_port=9100), run=unchanged, out=out)
    assert unchanged.calls == [["systemctl", "--user", "daemon-reload"]] and "restart" not in out.getvalue().split("next:")[0]


def test_setup_refuses_to_drop_the_units_settings_sources(layout: Layout, settings: Settings, tmp_path: Path) -> None:
    sources = ("--config", str(layout.home / "c.yaml"))
    chatgpt.setup(settings, layout=layout, params=_params(layout, serve_args=sources,
                                                          environment=(("MAF_BUDGET_USD", "9"),)),
                  reload=False, out=io.StringIO())
    assert chatgpt.installed_serve_sources(layout) == {"--config": str(layout.home / "c.yaml"), "MAF_BUDGET_USD": "9"}
    before = (layout.unit_dir / chatgpt.MCP_UNIT).read_text()
    with pytest.raises(ValueError, match=r"runs with --config .*c\.yaml, MAF_BUDGET_USD=9, which this setup was not given"):
        chatgpt.setup(settings, layout=layout, params=_params(layout), reload=False, out=io.StringIO())
    assert (layout.unit_dir / chatgpt.MCP_UNIT).read_text() == before
    other = ("--config", str(tmp_path / "other.yaml"))  # another value is an explicit change
    chatgpt.setup(settings, layout=layout, params=_params(layout, serve_args=other, environment=(("MAF_BUDGET_USD", "4"),)),
                  reload=False, out=io.StringIO())
    assert chatgpt.installed_serve_sources(layout)["--config"] == str(tmp_path / "other.yaml")


def test_setup_keeps_the_installed_tunnel_client_and_health_port(layout: Layout, settings: Settings) -> None:
    custom = Path("/opt/tc/tunnel-client")
    chatgpt.setup(settings, layout=layout, params=_params(layout, tunnel_client=custom, health_port=18999),
                  reload=False, out=io.StringIO())
    assert chatgpt.installed_tunnel_options(layout) == (custom, 18999)
    home_tc = layout.home / "bin" / "tunnel-client"
    chatgpt.setup(settings, layout=layout, params=_params(layout, tunnel_client=home_tc), reload=False, out=io.StringIO())
    assert chatgpt.installed_tunnel_options(layout) == (home_tc, chatgpt.DEFAULT_HEALTH_PORT)  # %h expanded back
    assert chatgpt.installed_tunnel_options(Layout.default(environ={}, home=layout.home / "none")) == (None, None)


@pytest.mark.parametrize(("kw", "message"), [
    ({"health_port": 8765}, "health port"),
    ({"tunnel_client": Path("bin/tunnel-client")}, "absolute"),
])
def test_setup_refuses_bad_tunnel_options_before_writing(layout: Layout, settings: Settings, kw: dict[str, Any],
                                                         message: str) -> None:
    with pytest.raises(ValueError, match=message):
        chatgpt.setup(settings, layout=layout, params=_params(layout, **kw), reload=False, out=io.StringIO())
    assert not layout.maf_config_dir.exists()


def test_setup_creates_the_inbox_private(layout: Layout, settings: Settings) -> None:
    out = io.StringIO()
    chatgpt.setup(settings, layout=layout, params=_params(layout), reload=False, out=out)
    assert settings.mcp_inbox is not None and _mode(settings.mcp_inbox) == 0o700
    assert "(0700) for start_run input files" in out.getvalue()
    settings.mcp_inbox.chmod(0o755)
    chatgpt.setup(settings, layout=layout, params=_params(layout), reload=False, out=io.StringIO())
    assert _mode(settings.mcp_inbox) == 0o755  # an existing inbox is left alone


def test_setup_next_steps_restart_and_explain_linger(layout: Layout, settings: Settings) -> None:
    out = io.StringIO()
    chatgpt.setup(settings, layout=layout, params=_params(layout), reload=False, out=out)
    text = out.getvalue()
    assert "systemctl --user restart maf-mcp.service maf-tunnel.service" in text
    assert "starts the units at boot without a login" in text
    assert text.index("Platform") < text.index("systemctl --user enable")  # the ids come before the start


def test_setup_reports_a_failed_reload(layout: Layout, settings: Settings) -> None:
    run, out = Recorder({"daemon-reload": (1, "Failed to connect to bus")}), io.StringIO()
    assert chatgpt.setup(settings, layout=layout, params=_params(layout), run=run, out=out) == 1
    assert "Failed to connect to bus" in out.getvalue()


def test_setup_refuses_non_loopback_before_writing(layout: Layout, settings: Settings) -> None:
    with pytest.raises(ValueError, match="loopback"):
        chatgpt.setup(settings, layout=layout, params=_params(layout, mcp_host="0.0.0.0"), out=io.StringIO())
    assert not layout.maf_config_dir.exists()


# --------------------------------------------------------------------------- env files and redaction


def test_read_env_file(tmp_path: Path) -> None:
    path = tmp_path / "x.env"
    path.write_text("# comment\n\nA=1\nB = 'two'\nC=\"three\"\nD=\nnot a pair\n;semi=x\nE=a=b\n")
    assert chatgpt.read_env_file(path) == {"A": "1", "B": "two", "C": "three", "D": "", "E": "a=b"}
    assert chatgpt.read_env_file(tmp_path / "missing") == {}


def test_redact_masks_known_values_and_key_shapes() -> None:
    text = ("got key=my-very-secret-value sk-proj-abcdefghijklmnop AIzaSyA1234567890abcdefghijklmn "
            "sk-ant-api03-abcdefghijklmnop Authorization: Bearer eyJhbGciOi.xyz tunnel_0123")
    masked = chatgpt.redact(text, {"my-very-secret-value"})
    assert "my-very-secret-value" not in masked and "sk-proj-abc" not in masked and "AIzaSy" not in masked
    assert "sk-ant-api03" not in masked
    assert "Bearer [redacted]" in masked and "eyJhbGciOi" not in masked
    assert "tunnel_0123" in masked


def test_secret_values_collects_env_files_and_shell_keys(layout: Layout) -> None:
    layout.maf_config_dir.mkdir(parents=True)
    layout.tunnel_env.write_text(f"CONTROL_PLANE_TUNNEL_ID={TUNNEL_ID}\nCONTROL_PLANE_API_KEY=runtime-key-123\n")
    layout.maf_env.write_text("OPENAI_API_KEY=openai-key-123\nGEMINI_API_KEY=short\n")
    found = chatgpt.secret_values(layout, environ={"ANTHROPIC_API_KEY": "anthropic-key-123", "HOME": "/home/x"})
    assert found == {"runtime-key-123", "openai-key-123", "anthropic-key-123"}  # ids and short values are not keys


def test_without_keys_unsets_every_key_variable() -> None:
    argv = chatgpt.without_keys(["/bin/tc", "--version"])
    assert argv[0] == "env" and argv[-2:] == ["/bin/tc", "--version"]
    assert {argv[i + 1] for i, a in enumerate(argv) if a == "-u"} == set(chatgpt.KEY_VARIABLES)


# --------------------------------------------------------------------------- status


def _installed(layout: Layout, settings: Settings, *, filled: bool = True) -> None:
    chatgpt.setup(settings, layout=layout, params=_params(layout, health_port=18766), reload=False, out=io.StringIO())
    if filled:
        layout.tunnel_env.write_text(f"CONTROL_PLANE_TUNNEL_ID={TUNNEL_ID}\nCONTROL_PLANE_API_KEY=runtime-key-secret\n")
        layout.maf_env.write_text("OPENAI_API_KEY=o-key-secret-1\nGEMINI_API_KEY=g-key-secret-1\n"
                                  "ANTHROPIC_API_KEY=a-key-secret-1\n")
    tc = layout.home / ".local" / "bin" / "tunnel-client"
    tc.parent.mkdir(parents=True, exist_ok=True)
    tc.write_text("#!/bin/sh\n")


ACTIVE = "LoadState=loaded\nActiveState=active\nSubState=running\nUnitFileState=enabled\nNRestarts=0\nResult=success\n"


def _status(layout: Layout, settings: Settings, run: Recorder, *, mcp_ok: bool = True, readyz: int | None = 200,
            environ: dict[str, str] | None = None) -> tuple[int, str]:
    out = io.StringIO()
    probed: list[str] = []
    fetched: list[str] = []

    def probe(url: str, uds: Path | None) -> tuple[bool, str]:
        probed.append(url)
        assert uds == layout.runtime_dir / "maf" / "mcp.sock"  # over the unit's socket, not a TCP port
        return mcp_ok, "fake detail" if mcp_ok else "ConnectError: refused"

    def get(url: str) -> tuple[int | None, str]:
        fetched.append(url)
        return readyz, "ready" if readyz == 200 else "down"

    code = chatgpt.status(settings, layout=layout, run=run, probe=probe, get=get, out=out,
                          environ=environ if environ is not None else {"USER": "u"})
    assert probed == ["http://127.0.0.1:8765/mcp"]
    assert fetched == ["http://127.0.0.1:18766/readyz"]  # the health port comes from the installed unit
    return code, out.getvalue()


def test_status_ready(layout: Layout, settings: Settings) -> None:
    _installed(layout, settings)
    run = Recorder({"show maf-": (0, ACTIVE), "--version": (0, "0.0.15+abc"), "Linger": (0, "yes\n"),
                    "health --url": (0, "Control-plane poll: PASS")})
    code, text = _status(layout, settings, run)
    assert code == 0, text
    assert text.rstrip().endswith("ready: ChatGPT can reach maf through the tunnel")
    assert "active (running), enabled" in text and "control-plane poll: ok" in text
    assert "CONTROL_PLANE_TUNNEL_ID tunnel_0123...cdef" in text and "CONTROL_PLANE_API_KEY set" in text
    tunnel_calls = [c for c in run.calls if any("tunnel-client" in a for a in c)]
    assert tunnel_calls and all(c[0] == "env" for c in tunnel_calls)  # never with a key in its environment


def test_status_never_prints_secrets_even_from_logs(layout: Layout, settings: Settings) -> None:
    _installed(layout, settings)
    leak = "runtime-key-secret o-key-secret-1 a-key-secret-1 sk-proj-abcdefghijklmnopq shell-key-12345"
    run = Recorder({"show maf-": (0, ACTIVE), "journalctl": (0, f"Sep 29 maf: {leak}\n"), "Linger": (0, "yes\n")})
    _code, text = _status(layout, settings, run, environ={"USER": "u", "OPENAI_API_KEY": "shell-key-12345"})
    for secret in ("runtime-key-secret", "o-key-secret-1", "a-key-secret-1", "sk-proj-abc", "shell-key-12345"):
        assert secret not in text
    assert "[redacted]" in text


def test_status_not_ready_lists_the_reasons(layout: Layout, settings: Settings) -> None:
    _installed(layout, settings, filled=False)
    inactive = ACTIVE.replace("ActiveState=active", "ActiveState=failed").replace("Result=success", "Result=exit-code")
    run = Recorder({"show maf-tunnel": (0, inactive), "show maf-mcp": (0, ACTIVE)})
    code, text = _status(layout, settings, run, mcp_ok=False, readyz=None)
    assert code == 1
    reasons = text.strip().splitlines()[-1]
    assert reasons.startswith("not ready:")
    for reason in ("maf serve does not answer MCP", "maf-tunnel.service is not active", "tunnel-client is not ready",
                   "CONTROL_PLANE_API_KEY missing", "OPENAI_API_KEY missing"):
        assert reason in reasons
    assert "last result exit-code" in text and "units stop when you log out" not in text  # linger unknown: no line


def test_status_readyz_alone_is_not_enough(layout: Layout, settings: Settings) -> None:
    """tunnel-client v0.0.15 reports /readyz 200 while every control-plane poll is refused (401); status needs a
    successful poll too."""
    _installed(layout, settings)
    run = Recorder({"show maf-": (0, ACTIVE), "health --url": (2, "Control-plane poll: FAIL | no successful poll")})
    code, text = _status(layout, settings, run)
    assert code == 1
    assert "control-plane poll: NOT confirmed (Control-plane poll: FAIL" in text


def test_status_hints_from_logs_and_target_mismatch(layout: Layout, settings: Settings) -> None:
    _installed(layout, settings)
    mcp_log = "WARNING mcp.server.transport_security: Invalid Origin header: https://chatgpt.com\n"
    tunnel_log = '{"level":"warn","msg":"poll failed; backing off","status":401}\n'
    run = Recorder({"show maf-": (0, ACTIVE), "-u maf-mcp.service": (0, mcp_log),
                    "-u maf-tunnel.service": (0, tunnel_log), "health --url": (0, "ok")})
    moved = settings.model_copy(update={"mcp_port": 9999})
    out = io.StringIO()
    code = chatgpt.status(moved, layout=layout, run=run, probe=lambda url, uds: (True, "ok"),
                          get=lambda url: (200, "ready"), out=out, environ={})
    text = out.getvalue()
    assert code == 1
    assert ("WARNING: the tunnel forwards to http://127.0.0.1:8765/mcp via %t/maf/mcp.sock, but maf serve is "
            "configured for http://127.0.0.1:9999/mcp via %t/maf/mcp.sock") in text
    assert 'mcp_allowed_origins: ["https://chatgpt.com"]' in text
    assert "Tunnels Read + Use" in text
    assert "after editing ~/.config/maf/tunnel.env: systemctl --user restart maf-tunnel.service" in text


def test_parse_upstream() -> None:
    assert chatgpt.parse_upstream("http://127.0.0.1:8765/mcp") == ("http://127.0.0.1:8765/mcp", None)
    assert chatgpt.parse_upstream("url=http://127.0.0.1:1/mcp,unix-socket=%t/maf/mcp.sock") == (
        "http://127.0.0.1:1/mcp", "%t/maf/mcp.sock")


def test_log_hints_host() -> None:
    hints = chatgpt.log_hints(["Invalid Host header: evil.example:8765"], [], 8765)
    assert hints == ["maf refused Host evil.example:8765: the tunnel's MCP URL must be http://127.0.0.1:8765/mcp "
                     "(or localhost with that port)"]
    assert chatgpt.log_hints(["all good"], ['{"msg":"poll ok"}'], 8765) == []


def test_log_hints_missing_key_says_to_restart() -> None:
    hints = chatgpt.log_hints(["run x failed: providers: ProviderError: OPENAI_API_KEY is not set"], [], 8765)
    assert hints == ["maf serve runs without OPENAI_API_KEY: fill it in ~/.config/maf/maf.env, then systemctl --user "
                     "restart maf-mcp.service (a running maf keeps the environment it started with)"]


def test_run_command_never_raises() -> None:
    assert chatgpt.run_command(["/nonexistent/binary"], 5).returncode == 127
    assert chatgpt.run_command(["sleep", "5"], 0.2).returncode == 124


# --------------------------------------------------------------------------- real loopback round trip


class _NoStages:
    def run_stage(self, ctx: StageContext) -> StageOutput:
        return StageOutput(notes=[])


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def live_server(settings: Settings, fake_providers: Any) -> Iterator[str]:
    """The HTTP app ``maf serve`` runs, on a real loopback port (uvicorn in a thread)."""
    import uvicorn

    pipeline = Pipeline(settings, providers_factory=fake_providers.factory(),
                        backends={s: _NoStages() for s in STAGE_ORDER})  # type: ignore[arg-type,misc]
    manager = RunManager(pipeline)
    port = _free_port()
    app = mcp_server.build_http_app(build_server(manager), "127.0.0.1", port)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.should_exit = True
        thread.join(10)
        manager.shutdown(wait=True)


def test_mcp_health_round_trip(live_server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")  # proxies are ignored for the loopback probe
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
    ok, detail = chatgpt.mcp_health(live_server)
    assert ok, detail
    assert re.search(rf"initialize \+ tools/list in [0-9.]+ s: maf {re.escape(__version__)}, protocol \S+, tools: get_run_result, "
                     r"get_run_status, list_runs, start_run", detail)


def test_mcp_health_over_a_unix_socket(settings: Settings, fake_providers: Any, tmp_path: Path) -> None:
    """What ``status`` does against the unit: the round trip over the socket, with the unit's Host header."""
    import uvicorn

    pipeline = Pipeline(settings, providers_factory=fake_providers.factory(),
                        backends={s: _NoStages() for s in STAGE_ORDER})  # type: ignore[arg-type,misc]
    manager = RunManager(pipeline)
    path = tmp_path / "rt" / "mcp.sock"
    sock = mcp_server.bind_unix(path)
    assert _mode(path) == 0o600 and _mode(path.parent) == 0o700
    app = mcp_server.build_http_app(build_server(manager), "127.0.0.1", 8765)
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "uvicorn did not start"
        time.sleep(0.05)
    try:
        ok, detail = chatgpt.mcp_health("http://127.0.0.1:8765/mcp", path)
        assert ok, detail
        assert "tools: get_run_result, get_run_status, list_runs, start_run" in detail
        wrong_host, _ = chatgpt.mcp_health("http://127.0.0.1:9999/mcp", path, timeout=5)
        assert not wrong_host  # the Host check still applies over the socket
    finally:
        server.should_exit = True
        thread.join(10)
        manager.shutdown(wait=True)


def test_mcp_health_reports_failures(live_server: str) -> None:
    ok, detail = chatgpt.mcp_health(f"http://127.0.0.1:{_free_port()}/mcp", timeout=5)
    assert not ok and detail
    ok, detail = chatgpt.mcp_health(live_server.replace("/mcp", "/nope"), timeout=5)
    assert not ok and detail


def test_http_get_ignores_proxies(live_server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    code, _body = chatgpt.http_get(live_server)
    assert code is not None and code != 200  # a plain GET without an MCP session is refused, but it arrived
    assert chatgpt.http_get(f"http://127.0.0.1:{_free_port()}/readyz")[0] is None


def test_env_file_modes_survive_umask(layout: Layout, settings: Settings) -> None:
    old = os.umask(0)
    try:
        chatgpt.setup(settings, layout=layout, params=_params(layout), reload=False, out=io.StringIO())
    finally:
        os.umask(old)
    assert _mode(layout.tunnel_env) == 0o600 and _mode(layout.maf_env) == 0o600
