"""Shell scripts: syntax, the tunnel-client installer against a fake local release (no network), and the setup
wizard traced with stubbed system tools in a temp HOME (it must never echo a secret)."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import stat
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
INSTALLER = ROOT / "scripts" / "install-tunnel-client.sh"
WIZARD = ROOT / "scripts" / "chatgpt-setup-wizard.sh"
TUNNEL_ID = "tunnel_0123456789abcdef0123456789abcdef"


@pytest.mark.parametrize("script", [INSTALLER, WIZARD], ids=lambda p: p.name)
def test_scripts_parse_and_are_executable(script: Path) -> None:
    subprocess.run(["bash", "-n", str(script)], check=True)
    assert script.stat().st_mode & stat.S_IXUSR


def test_installer_pins_the_verified_release() -> None:
    text = INSTALLER.read_text(encoding="utf-8")
    assert 'PINNED_VERSION="v0.0.15"' in text
    assert 'PINNED_SHA256="8c836dc5d68d68b663d9a5c5b28ff9fa780d9f7a3fffb1c306880b8f32fab5f1"' in text


needs_tools = pytest.mark.skipif(
    not all(shutil.which(t) for t in ("curl", "unzip", "sha256sum"))
    or platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"),
    reason="installer needs Linux amd64 with curl, unzip and sha256sum",
)


def _release(root: Path, version: str, *, reported: str | None = None) -> str:
    """A fake release dir: zip with a stub tunnel-client printing ``<version>+stub``, and SHA256SUMS.txt."""
    root.mkdir(parents=True, exist_ok=True)
    asset = root / f"tunnel-client-{version}-linux-amd64.zip"
    with zipfile.ZipFile(asset, "w") as archive:
        info = zipfile.ZipInfo("tunnel-client")
        info.external_attr = 0o755 << 16
        archive.writestr(info, f"#!/bin/sh\necho '{reported or version.lstrip('v')}+stub (fake)'\n")
        archive.writestr("LICENSE", "fake\n")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    (root / "SHA256SUMS.txt").write_text(f"{'0' * 64}  other-file.zip\n{digest}  {asset.name}\n")
    return digest


SIGKILLED = 137
"""A SIGKILLed installer. On the dev host (CrowdStrike Falcon running) ``mktemp -d`` of a hidden directory was killed
about half the time, so the installer no longer does that; the tests still retry, then skip rather than flake."""


def _install(prefix: Path, mirror: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "TUNNEL_CLIENT_PREFIX": str(prefix), "TUNNEL_CLIENT_BASE_URL": f"file://{mirror}"}
    for _attempt in range(3):
        result = subprocess.run([str(INSTALLER), *args], capture_output=True, text=True, env=env, timeout=60)
        if result.returncode != SIGKILLED:
            return result
    pytest.skip("the installer was SIGKILLed three times (endpoint security killing the fresh binary?)")


@needs_tools
def test_installer_installs_verifies_and_is_idempotent(tmp_path: Path) -> None:
    digest = _release(tmp_path / "mirror", "v9.9.9")
    prefix = tmp_path / "prefix"
    first = _install(prefix, tmp_path / "mirror", "--version", "v9.9.9", "--sha256", digest)
    assert first.returncode == 0, first.stderr
    link = prefix / "bin" / "tunnel-client"
    assert link.is_symlink() and os.readlink(link) == str(prefix / "opt" / "tunnel-client" / "v9.9.9" / "tunnel-client")
    assert subprocess.run([str(link)], capture_output=True, text=True).stdout.startswith("9.9.9+stub")
    assert (prefix / "opt" / "tunnel-client" / "v9.9.9" / ".verified-sha256").read_text().strip() == digest

    again = _install(prefix, tmp_path / "gone", "--version", "v9.9.9", "--sha256", digest)  # no download needed
    assert again.returncode == 0 and "already installed" in again.stdout

    link.unlink()
    relinked = _install(prefix, tmp_path / "gone", "--version", "v9.9.9", "--sha256", digest)
    assert relinked.returncode == 0 and "relinked" in relinked.stdout and link.is_symlink()


@needs_tools
def test_installer_refuses_a_tampered_zip(tmp_path: Path) -> None:
    digest = _release(tmp_path / "mirror", "v9.9.9")
    with (tmp_path / "mirror" / "tunnel-client-v9.9.9-linux-amd64.zip").open("ab") as handle:
        handle.write(b"x")
    result = _install(tmp_path / "prefix", tmp_path / "mirror", "--version", "v9.9.9", "--sha256", digest)
    assert result.returncode == 1 and "MISMATCH" in result.stderr and "refusing" in result.stderr
    assert not (tmp_path / "prefix" / "bin").exists()


@needs_tools
def test_installer_pinned_hash_beats_a_consistent_forged_release(tmp_path: Path) -> None:
    """A mirror whose SHA256SUMS.txt matches its (forged) zip still fails the pinned v0.0.15 hash."""
    _release(tmp_path / "mirror", "v0.0.15")
    result = _install(tmp_path / "prefix", tmp_path / "mirror")
    assert result.returncode == 1 and "pinned" in result.stderr
    assert not (tmp_path / "prefix" / "opt").exists()


@needs_tools
def test_installer_unpinned_version_warns_and_checks_sums(tmp_path: Path) -> None:
    _release(tmp_path / "mirror", "v9.9.8")
    result = _install(tmp_path / "prefix", tmp_path / "mirror", "--version", "v9.9.8")
    assert result.returncode == 0, result.stderr
    assert "no pinned hash" in result.stdout
    (tmp_path / "mirror" / "SHA256SUMS.txt").write_text("")
    missing = _install(tmp_path / "p2", tmp_path / "mirror", "--version", "v9.9.8")
    assert missing.returncode == 1 and "no entry" in missing.stderr


@needs_tools
def test_installer_refuses_a_binary_reporting_another_version(tmp_path: Path) -> None:
    digest = _release(tmp_path / "mirror", "v9.9.9", reported="1.0.0")
    result = _install(tmp_path / "prefix", tmp_path / "mirror", "--version", "v9.9.9", "--sha256", digest)
    assert result.returncode == 1 and "does not report version" in result.stderr
    assert not (prefix_bin := tmp_path / "prefix" / "bin" / "tunnel-client").exists(), prefix_bin


@pytest.mark.parametrize("args", [["--version", "1.0"], ["--sha256", "XYZ"], ["--bogus"]])
def test_installer_rejects_bad_arguments(args: list[str], tmp_path: Path) -> None:
    result = _install(tmp_path / "prefix", tmp_path, *args)
    assert result.returncode == 1 and result.stderr


# --------------------------------------------------------------------------- wizard, traced


def _stub(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_wizard_writes_only_private_env_files_and_never_echoes_secrets(tmp_path: Path) -> None:
    home, shim, calls = tmp_path / "home", tmp_path / "shim", tmp_path / "calls.log"
    (home / ".local" / "bin").mkdir(parents=True)
    shim.mkdir()
    log = f'echo "$(basename "$0") $*" >> {calls}\n'
    _stub(shim / "systemctl", log + "exit 0\n")
    _stub(shim / "loginctl", log + 'case "$*" in *show-user*) echo no;; esac\n')
    _stub(shim / "journalctl", log + 'echo "WARNING mcp.server.transport_security: Invalid Origin header: '
                                    'https://chatgpt.com"\n')
    _stub(shim / "xdg-open", "exit 0\n")
    maf = tmp_path / "maf"
    _stub(maf, log)
    _stub(home / ".local" / "bin" / "tunnel-client",
          f'printf "tunnel-client %s | OPENAI_API_KEY=%s CONTROL_PLANE_API_KEY=%s\\n" "$*" "${{OPENAI_API_KEY:+SET}}" '
          f'"${{CONTROL_PLANE_API_KEY:+SET}}" >> {calls}\n')
    (home / ".config" / "systemd" / "user").mkdir(parents=True)
    (home / ".config" / "systemd" / "user" / "maf-tunnel.service").write_text(
        (ROOT / "contrib" / "systemd" / "maf-tunnel.service").read_text()
    )
    runtime_key, openai_key, gemini_key = "sk-runtime-FAKE-0123456789", "sk-openai-FAKE-0123456789", "gem-FAKE-0123456"
    answers = "\n".join([
        "",              # banner
        "n",             # stage 1: re-check the installed tunnel-client? no
        "y", "",         # run maf chatgpt setup; continue
        "",              # stage 2
        "not-an-id", TUNNEL_ID,  # stage 3: one bad try, then a good id
        runtime_key,     # stage 4
        "y", gemini_key, "",     # stage 5: copy OPENAI_API_KEY from the shell; paste Gemini; skip Anthropic
        "y", "y", "",    # stage 6: start maf-mcp; it is already running with old keys: restart it; continue
        "y", "n", "",    # stage 7: start the tunnel; no linger; continue
        "",              # stage 8
        "", "y", "",     # stage 9: app created; allow the refused origin; continue
        "", "",          # stages 10, 11
    ]) + "\n"
    env = {"PATH": f"{shim}:{os.environ['PATH']}", "HOME": str(home), "USER": "tester", "MAF": str(maf),
           "OPENAI_API_KEY": openai_key, "WIZARD_ALLOW_NONTTY": "1"}
    result = subprocess.run([str(WIZARD)], input=answers, capture_output=True, text=True, env=env, timeout=120)
    transcript = result.stdout + result.stderr
    assert result.returncode == 0, transcript
    for secret in (runtime_key, openai_key, gemini_key):
        assert secret not in transcript

    config = home / ".config" / "maf"
    tunnel_env, maf_env = (config / "tunnel.env").read_text(), (config / "maf.env").read_text()
    assert f"CONTROL_PLANE_TUNNEL_ID={TUNNEL_ID}\n" in tunnel_env and f"CONTROL_PLANE_API_KEY={runtime_key}\n" in tunnel_env
    assert f"OPENAI_API_KEY={openai_key}\n" in maf_env and f"GEMINI_API_KEY={gemini_key}\n" in maf_env
    assert "ANTHROPIC_API_KEY" not in maf_env and openai_key not in tunnel_env
    for name in ("tunnel.env", "maf.env"):
        assert stat.S_IMODE((config / name).stat().st_mode) == 0o600
    assert (config / "config.yaml").read_text() == 'mcp_allowed_origins: ["https://chatgpt.com"]\n'

    recorded = calls.read_text().splitlines()
    doctor = next(line for line in recorded if line.startswith("tunnel-client doctor"))
    socket_path = f"/run/user/{os.getuid()}/maf/mcp.sock"  # %t expanded like systemd does
    assert f"--mcp.server-url url=http://127.0.0.1:8765/mcp,unix-socket={socket_path}" in doctor
    assert "--health.listen-addr 127.0.0.1:8766" in doctor
    for line in recorded:
        if line.startswith("tunnel-client"):
            assert "OPENAI_API_KEY= " in line  # tunnel-client never sees a provider key
    assert any(line.startswith("tunnel-client admin tunnels get " + TUNNEL_ID) for line in recorded)
    assert "systemctl --user enable maf-mcp.service" in recorded
    # maf.env changed while maf-mcp was running: restart (stage 6) so the service gets the new keys; the tunnel is
    # restarted, not started, so an edited tunnel.env reaches a running tunnel-client too.
    assert recorded.index("systemctl --user restart maf-mcp.service") < recorded.index(
        "systemctl --user restart maf-tunnel.service")
    assert "systemctl --user enable maf-tunnel.service" in recorded
    assert not any("--now" in line for line in recorded)
    assert not any(line.startswith("loginctl enable-linger") for line in recorded)
    assert "ANTHROPIC_API_KEY in" in transcript  # listed under "still to do by hand"
    assert "start the units at boot without a login" in transcript.lower()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_wizard_rerun_restarts_only_what_changed_and_edits_the_units_config(tmp_path: Path) -> None:
    """A re-run with unchanged keys leaves a running maf alone, still restarts the tunnel (tunnel.env may have been
    fixed), explains doctor's expected socket failures, and writes the Origin fix to the config the unit reads."""
    home, shim, calls = tmp_path / "home", tmp_path / "shim", tmp_path / "calls.log"
    (home / ".local" / "bin").mkdir(parents=True)
    shim.mkdir()
    log = f'echo "$(basename "$0") $*" >> {calls}\n'
    _stub(shim / "systemctl", log + "exit 0\n")  # every unit reads as active
    _stub(shim / "loginctl", log + 'case "$*" in *show-user*) echo yes;; esac\n')
    _stub(shim / "journalctl", log + 'echo "WARNING mcp.server.transport_security: Invalid Origin header: '
                                    'https://chatgpt.com"\n')
    _stub(shim / "xdg-open", "exit 0\n")
    maf = tmp_path / "maf"
    sees = f'echo "setup sees XDG_DATA_HOME=${{XDG_DATA_HOME:-}}" >> {calls}'
    _stub(maf, log + f'case "$*" in *"chatgpt setup"*) {sees};; esac\n')
    _stub(home / ".local" / "bin" / "tunnel-client",
          log + 'case "$1" in doctor) echo "CHECK mcp_server_reachable FAIL dial tcp: refused"; '
                'echo "FAILED_CHECKS mcp_server_reachable,oauth_metadata"; exit 2;; esac\n')
    units = home / ".config" / "systemd" / "user"
    units.mkdir(parents=True)
    (units / "maf-tunnel.service").write_text((ROOT / "contrib" / "systemd" / "maf-tunnel.service").read_text())
    custom = tmp_path / "chatgpt.yaml"
    data = tmp_path / "data"  # an installed maf's XDG_DATA_HOME, which this shell does not set
    unbuffered = "Environment=PYTHONUNBUFFERED=1\n"
    (units / "maf-mcp.service").write_text(
        (ROOT / "contrib" / "systemd" / "maf-mcp.service").read_text().replace(" serve ", f" serve --config {custom} ")
        .replace(unbuffered, f"{unbuffered}Environment=XDG_DATA_HOME={data}\n")
    )
    config = home / ".config" / "maf"
    config.mkdir(parents=True)
    (config / "tunnel.env").write_text(f"CONTROL_PLANE_TUNNEL_ID={TUNNEL_ID}\nCONTROL_PLANE_API_KEY=sk-rt-FAKE-0123456789\n")
    keys = "OPENAI_API_KEY=o-FAKE-0123456\nGEMINI_API_KEY=g-FAKE-0123456\nANTHROPIC_API_KEY=a-FAKE-0123456\n"
    (config / "maf.env").write_text(keys)
    answers = "\n".join([
        "", "n", "y", "",  # banner; stage 1: no re-check, run setup, continue
        "",                # stage 2
        "", "",            # stages 3, 4: keep the saved id and key
        "", "", "",        # stage 5: keep all three keys
        "y", "",           # stage 6: start maf-mcp (already running, unchanged: no restart prompt); continue
        "y", "",           # stage 7: (re)start the tunnel; linger is on already; continue
        "",                # stage 8
        "", "y", "",       # stage 9: allow the refused origin
        "", "",
    ]) + "\n"
    env = {"PATH": f"{shim}:{os.environ['PATH']}", "HOME": str(home), "USER": "tester", "MAF": str(maf),
           "WIZARD_ALLOW_NONTTY": "1"}
    result = subprocess.run([str(WIZARD)], input=answers, capture_output=True, text=True, env=env, timeout=120)
    transcript = result.stdout + result.stderr
    assert result.returncode == 0, transcript
    assert (config / "maf.env").read_text() == keys
    recorded = calls.read_text().splitlines()
    stage6 = recorded[:recorded.index("systemctl --user restart maf-tunnel.service")]
    assert "systemctl --user restart maf-mcp.service" not in stage6
    assert "already running with the current keys and unit" in transcript
    assert "are expected: maf listens on a Unix socket" in transcript and "doctor reported problems" not in transcript
    assert custom.read_text() == 'mcp_allowed_origins: ["https://chatgpt.com"]\n'
    assert not (config / "config.yaml").exists()
    assert f"maf --config {custom} chatgpt setup" in recorded  # the unit's settings sources are passed on
    assert f"setup sees XDG_DATA_HOME={data}" in recorded


def test_wizard_refuses_without_a_terminal(tmp_path: Path) -> None:
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    result = subprocess.run([str(WIZARD)], input="", capture_output=True, text=True, env=env, timeout=30)
    assert result.returncode == 1 and "terminal" in result.stderr
