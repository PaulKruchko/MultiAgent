"""Claude Code headless adapter. A fake ``Runner`` stands in for the CLI; the real ``claude`` never runs."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from conftest import load_provider_fixture

from maf.providers.base import (
    CompletionRequest,
    CompletionResult,
    Message,
    ProviderError,
    SandboxUnavailable,
    StructuredOutputError,
)
from maf.providers.claude_code import (
    CLI_CHILD_TMPDIR_MAX_BYTES,
    DISALLOWED_TOOLS,
    PREFLIGHT_COMMAND,
    PREFLIGHT_FILE,
    PREFLIGHT_OUTPUT,
    PREFLIGHT_SCHEMA,
    SENSITIVE_READ_PATHS,
    ClaudeCodeBudgetExhausted,
    ClaudeCodeProvider,
    ClaudeCodeTimeout,
    CompletedProcess,
    builtin_tools,
    check_tmpdir,
    ensure_private_dir,
    format_budget,
    max_tmpdir_bytes,
    preflight_budget_usd,
    preflight_request,
    reported_digest,
    sandbox_failure,
    sandbox_settings,
    scratch_tmpdir,
    subprocess_runner,
    written_digest,
)
from maf.types import Usage

MODEL = "claude-opus-5-5"
TOOLS = ("Read(./**)", "Edit(./**)", "Bash(make *)")
SCHEMA = {
    "type": "object",
    "properties": {"artifacts": {"type": "array", "items": {"type": "string"}}, "tests_passed": {"type": "boolean"}},
    "required": ["artifacts", "tests_passed"],
}


@pytest.fixture(autouse=True)
def tmp_base(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A short private ``tmp_base`` for every provider built here (``/tmp/mXXXXXXXX``, 14 bytes, so ``TMPDIR`` is
    31), so no test creates real ``/tmp/maf-*`` directories. ``tmp_path`` is too long for ``max_tmpdir_bytes()``."""
    base = Path(tempfile.mkdtemp(prefix="m", dir="/tmp"))
    monkeypatch.setitem(ClaudeCodeProvider.__init__.__kwdefaults__, "tmp_base", base)
    yield base
    shutil.rmtree(base, ignore_errors=True)


class FakeRunner:
    def __init__(self, reply: CompletedProcess | Exception) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, argv: Sequence[str], stdin_text: str, cwd: Path, env: dict[str, str], timeout_s: float
    ) -> CompletedProcess:
        self.calls.append({"argv": list(argv), "stdin": stdin_text, "cwd": cwd, "env": env, "timeout": timeout_s})
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _out(name: str, returncode: int = 0, stderr: str = "") -> CompletedProcess:
    return CompletedProcess(returncode=returncode, stdout=json.dumps(load_provider_fixture(name)), stderr=stderr)


def _req(**kw: Any) -> CompletionRequest:
    kw.setdefault("max_budget_usd", 8.0)
    return CompletionRequest.simple(MODEL, "Implement the allocator.", max_output_tokens=64_000, **kw)


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-must-not-leak")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-must-not-leak")
    monkeypatch.setenv("CLAUDECODE", "1")


def _provider(tmp_path: Path, reply: CompletedProcess | Exception, **kw: Any) -> tuple[ClaudeCodeProvider, FakeRunner]:
    runner = FakeRunner(reply)
    kw.setdefault("allowed_tools", TOOLS)
    provider = ClaudeCodeProvider(tmp_path / "ws", executable=Path("/opt/claude"), timeout_s=120.0, runner=runner, **kw)
    return provider, runner


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


# --- sandbox / argv / stdin ---------------------------------------------------------------------


def test_sandbox_settings(tmp_path: Path) -> None:
    assert sandbox_settings(tmp_path, ("~/.ssh", "/srv/vault/")) == {
        "permissions": {
            "deny": ["Read(~/.ssh/**)", "Edit(~/.ssh/**)", "Read(//srv/vault/**)", "Edit(//srv/vault/**)"]
        },
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "allowUnsandboxedCommands": False,
            "autoAllowBashIfSandboxed": True,
            "filesystem": {"allowWrite": [str(tmp_path.resolve())], "denyRead": ["~/.ssh", "/srv/vault/"]},
            "network": {"allowedDomains": []},
        },
    }


def test_default_sandbox_denies_credentials(tmp_path: Path) -> None:
    settings = sandbox_settings(tmp_path)
    # Credentials, shell files that export keys, histories and browser profiles: what a prompt injection in a
    # ChatGPT-written brief would read first.
    for path in ("~/.ssh", "~/.config", "~/.bashrc", "~/.profile", "~/.bash_profile", "~/.zshrc", "~/.bash_history",
                 "~/.python_history", "~/.pypirc", "~/.npmrc", "~/.pgpass", "~/.mozilla", "~/snap/firefox", "~/.var/app"):
        assert path in SENSITIVE_READ_PATHS, path
    assert len(set(SENSITIVE_READ_PATHS)) == len(SENSITIVE_READ_PATHS)
    assert settings["sandbox"]["filesystem"]["denyRead"] == list(SENSITIVE_READ_PATHS)
    assert "Read(~/.ssh/**)" in settings["permissions"]["deny"]


@pytest.mark.parametrize("bare", ["Read", "Edit", "Write", "MultiEdit", "NotebookEdit"])
def test_unscoped_file_tools_are_rejected(tmp_path: Path, bare: str) -> None:
    with pytest.raises(ValueError, match="path scope"):
        _provider(tmp_path, _out("claude_code_success"), allowed_tools=(bare, "Bash(make *)"))


def test_default_tools_have_no_unscoped_file_rules(tmp_path: Path) -> None:
    from maf.config import DEFAULT_CLAUDE_CODE_TOOLS

    provider = ClaudeCodeProvider(tmp_path, executable=Path("/opt/claude"), allowed_tools=DEFAULT_CLAUDE_CODE_TOOLS)
    argv = provider.build_argv(_req())
    listed = argv[argv.index("--allowedTools") + 1 : argv.index("--disallowedTools")]
    assert not {"Read", "Edit", "Write", "MultiEdit", "NotebookEdit"} & set(listed)
    assert {"Read(./**)", "Edit(./**)"} <= set(listed)
    assert "Write" in _flag(argv, "--tools").split(",")  # available, governed by the Edit rule


def test_builtin_tools_strips_rule_patterns() -> None:
    assert builtin_tools(("Read(./**)", "Bash", "Bash(make *)", "Glob")) == ["Read", "Bash", "Glob"]


def test_build_argv_minimal(tmp_path: Path) -> None:
    provider, _ = _provider(tmp_path, _out("claude_code_success"))
    argv = provider.build_argv(_req())
    assert argv == [
        "/opt/claude",
        "-p",
        "--output-format", "json",
        "--model", MODEL,
        "--effort", "high",
        "--max-budget-usd", "8.0000",
        "--permission-mode", "dontAsk",
        "--permission-prompts", "none",
        "--no-session-persistence",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--tools", ",".join(builtin_tools(TOOLS)),
        "--settings", json.dumps(sandbox_settings(tmp_path / "ws", tmpdir=provider.tmpdir), separators=(",", ":")),
        "--allowedTools", *TOOLS,
        "--disallowedTools", "WebFetch", "WebSearch",
    ]  # fmt: skip
    assert "--add-dir" not in argv
    assert "--bare" not in argv  # simple mode would hide Write/Glob/Grep
    assert DISALLOWED_TOOLS == ("WebFetch", "WebSearch")


def test_build_argv_optional_flags_precede_variadic_lists(tmp_path: Path) -> None:
    provider, _ = _provider(tmp_path, _out("claude_code_success"))
    argv = provider.build_argv(_req(system="Be terse.", json_schema=SCHEMA, effort="max", max_budget_usd=1.23456))
    assert _flag(argv, "--append-system-prompt") == "Be terse."
    assert json.loads(_flag(argv, "--json-schema")) == SCHEMA
    assert _flag(argv, "--effort") == "max"
    assert _flag(argv, "--max-budget-usd") == "1.2345"  # rounded down, never above the ledger clamp
    assert argv.index("--json-schema") < argv.index("--allowedTools") < argv.index("--disallowedTools")


def test_build_argv_requires_budget(tmp_path: Path) -> None:
    provider, _ = _provider(tmp_path, _out("claude_code_success"))
    with pytest.raises(ValueError, match="max_budget_usd"):
        provider.build_argv(_req(max_budget_usd=None))
    with pytest.raises(ValueError, match="too small"):
        provider.build_argv(_req(max_budget_usd=0.0))


def test_build_argv_without_allowed_tools(tmp_path: Path) -> None:
    provider = ClaudeCodeProvider(tmp_path, executable=Path("claude"), allowed_tools=())
    argv = provider.build_argv(_req())
    assert "--allowedTools" not in argv
    assert argv[-3:] == ["--disallowedTools", "WebFetch", "WebSearch"]


def test_build_stdin(tmp_path: Path) -> None:
    provider, _ = _provider(tmp_path, _out("claude_code_success"))
    assert provider.build_stdin(_req()) == "Implement the allocator."
    multi = CompletionRequest(
        model=MODEL,
        messages=(
            Message(role="user", content="a"),
            Message(role="assistant", content="b"),
            Message(role="user", content="c"),
        ),
        max_output_tokens=10,
        max_budget_usd=1.0,
    )
    assert provider.build_stdin(multi) == "## User\n\na\n\n## Assistant (earlier turn)\n\nb\n\n## User\n\nc"


def test_build_env_drops_other_keys_and_scopes_temp_dirs(
    tmp_path: Path, api_key: None, monkeypatch: pytest.MonkeyPatch, tmp_base: Path
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-must-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-must-not-leak")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    provider, _ = _provider(
        tmp_path, _out("claude_code_success"), extra_env={"FOO": "bar"}, path_prepend=[Path("/venv/bin")]
    )
    env = provider.build_env()
    assert "ANTHROPIC_API_KEY" not in env  # sandboxed Bash inherits the env; the key goes via apiKeyHelper
    assert "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB" not in env  # disables the OS sandbox in CLI 2.1.284
    assert env["LC_ALL"] == "C.UTF-8"
    for dropped in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "CLAUDECODE", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY"):
        assert dropped not in env
    ws = (tmp_path / "ws").resolve()
    assert env["TMPDIR"] == str(provider.tmpdir) and provider.tmpdir.parent == tmp_base
    assert len(os.fsencode(env["TMPDIR"])) <= max_tmpdir_bytes()
    assert env["MPLCONFIGDIR"] == str(ws / ".maf" / "mpl")  # matplotlib's cache stays in the workspace
    assert not (tmp_path / "ws").exists() and not provider.tmpdir.exists()  # pure: nothing created
    assert env["PATH"].startswith("/venv/bin:")
    assert env["FOO"] == "bar"
    assert "BASH_MAX_TIMEOUT_MS" not in env and "BASH_DEFAULT_TIMEOUT_MS" not in env  # the CLI's own 10 minutes


def test_bash_timeout_raises_the_cli_cap(tmp_path: Path, tmp_base: Path) -> None:
    """Claude Code 2.1.284 kills a Bash command after 10 minutes unless ``BASH_MAX_TIMEOUT_MS`` says otherwise, so a
    long reproduction never reached ``; echo $? > REPRO_EXIT``. The default per command stays 2 minutes."""
    provider, _ = _provider(tmp_path, _out("claude_code_success"), bash_timeout_s=2700)
    env = provider.build_env()
    assert env["BASH_MAX_TIMEOUT_MS"] == "2700000" and "BASH_DEFAULT_TIMEOUT_MS" not in env


def test_bound_to_rebinds_the_directory_and_denies_the_workspace(tmp_path: Path, tmp_base: Path, api_key: None) -> None:
    """The clean room's session: another cwd and writable directory, the workspace unreadable, long commands by
    default; the original provider is unchanged and both share one ``TMPDIR`` and the preflight verdict."""
    provider, runner = _provider(tmp_path, _out("claude_code_success"), bash_timeout_s=1800)
    provider.sandbox_verified = True
    room = tmp_path / "rooms" / "r1"
    clone = provider.bound_to(room, deny_read=(str(tmp_path / "ws"),), bash_default_is_max=True)
    assert (clone.workspace, clone.sandbox_verified, clone.tmpdir) == (room, True, provider.tmpdir)
    settings = json.loads(_flag(clone.build_argv(_req()), "--settings"))
    assert settings["sandbox"]["filesystem"]["allowWrite"] == [str(room.resolve()), str(provider.tmpdir)]
    assert str(tmp_path / "ws") in settings["sandbox"]["filesystem"]["denyRead"]
    assert f"Read(/{tmp_path / 'ws'}/**)" in settings["permissions"]["deny"]
    env = clone.build_env()
    assert env["BASH_MAX_TIMEOUT_MS"] == env["BASH_DEFAULT_TIMEOUT_MS"] == "1800000"
    assert env["MPLCONFIGDIR"] == str(room.resolve() / ".maf" / "mpl")
    assert provider.workspace == tmp_path / "ws" and "BASH_DEFAULT_TIMEOUT_MS" not in provider.build_env()
    original = json.loads(_flag(provider.build_argv(_req()), "--settings"))
    assert str(tmp_path / "ws") not in original["sandbox"]["filesystem"]["denyRead"]
    clone.complete(_req())
    assert runner.calls[-1]["cwd"] == room


# --- parse_output -------------------------------------------------------------------------------


def test_parse_output_plain_and_with_log_noise() -> None:
    payload = load_provider_fixture("claude_code_success")
    parsed = ClaudeCodeProvider.parse_output(json.dumps(payload))
    assert parsed.total_cost_usd == 3.2175
    assert parsed.subtype == "success"
    assert parsed.model_extra is not None and "modelUsage" in parsed.model_extra
    noisy = "warning: something\n{not json\n" + json.dumps(payload, indent=2) + "\n"
    assert ClaudeCodeProvider.parse_output(noisy).session_id == payload["session_id"]


def test_parse_output_event_list_takes_result() -> None:
    payload = [{"type": "system", "subtype": "init"}, load_provider_fixture("claude_code_success")]
    assert ClaudeCodeProvider.parse_output(json.dumps(payload)).num_turns == 37


@pytest.mark.parametrize("stdout", ["", "Error: not logged in\n", "[]", '"str"'])
def test_parse_output_without_result(stdout: str) -> None:
    with pytest.raises(ProviderError, match="no JSON result"):
        ClaudeCodeProvider.parse_output(stdout)


# --- complete -----------------------------------------------------------------------------------


def test_complete_success(tmp_path: Path, api_key: None, tmp_base: Path) -> None:
    provider, runner = _provider(tmp_path, _out("claude_code_success"), secrets_dir=tmp_path / "secrets")
    req = _req(system="sys")
    result = provider.complete(req)
    call = runner.calls[0]
    settings = json.loads(_flag(call["argv"], "--settings"))
    helper = settings["apiKeyHelper"]
    assert call["argv"] == provider.build_argv(req, api_key_helper=helper)
    assert helper.startswith(f"cat {tmp_path / 'secrets'}/anthropic-")
    assert not list((tmp_path / "secrets").iterdir())  # key file removed after the call
    assert "ANTHROPIC_API_KEY" not in call["env"]
    assert call["stdin"] == "Implement the allocator."
    assert call["cwd"] == tmp_path / "ws"
    assert call["timeout"] == 120.0
    assert "OPENAI_API_KEY" not in call["env"]
    tmpdir = provider.tmpdir
    assert tmpdir.parent == tmp_base and call["env"]["TMPDIR"] == str(tmpdir)
    assert stat.S_IMODE(tmpdir.stat().st_mode) == 0o700
    assert settings["sandbox"]["filesystem"]["allowWrite"] == [str((tmp_path / "ws").resolve()), str(tmpdir)]
    assert (tmp_path / "ws" / ".maf" / "mpl").is_dir()
    assert not (tmp_path / "ws" / ".maf" / "tmp").exists()
    assert result.cost_usd == 3.2175  # authoritative, not re-priced
    assert result.text.startswith("## Summary")
    assert result.session_id == "4f0c2a8e-9b1d-4c3e-a7f5-2d6e8b9c0a1f"
    assert result.provider == "claude_code"
    assert result.model == MODEL
    assert result.stop_reason == "success"
    assert result.usage.input_tokens == 45 + 21000 + 480000
    assert result.usage.cached_input_tokens == 480000
    assert result.raw["num_turns"] == 37
    assert result.parsed is None


def test_key_file_holds_key_with_private_mode_during_call(tmp_path: Path, api_key: None) -> None:
    seen: dict[str, Any] = {}

    def runner(argv: list[str], stdin: str, cwd: Path, env: dict[str, str], timeout: float) -> CompletedProcess:
        path = Path(json.loads(_flag(argv, "--settings"))["apiKeyHelper"].split(" ", 1)[1])
        seen["key"], seen["mode"] = path.read_text(), path.stat().st_mode & 0o777
        return _out("claude_code_success")

    provider = ClaudeCodeProvider(
        tmp_path / "ws", executable=Path("/opt/claude"), allowed_tools=TOOLS, runner=runner, secrets_dir=tmp_path / "s"
    )
    provider.complete(_req())
    assert seen == {"key": "sk-ant-test-not-real", "mode": 0o600}
    assert not list((tmp_path / "s").iterdir())


def test_complete_structured(tmp_path: Path, api_key: None) -> None:
    provider, _ = _provider(tmp_path, _out("claude_code_structured"))
    result = provider.complete(_req(json_schema=SCHEMA))
    assert result.parsed == {"artifacts": ["src/tlsf.c", "tests/test_tlsf.c"], "tests_passed": True}
    assert result.cost_usd == 1.05


def test_complete_structured_falls_back_to_result_text(tmp_path: Path, api_key: None) -> None:
    payload = load_provider_fixture("claude_code_success") | {"result": '{"artifacts": [], "tests_passed": false}'}
    provider, _ = _provider(tmp_path, CompletedProcess(returncode=0, stdout=json.dumps(payload)))
    assert provider.complete(_req(json_schema=SCHEMA)).parsed == {"artifacts": [], "tests_passed": False}


def test_complete_structured_invalid(tmp_path: Path, api_key: None) -> None:
    payload = load_provider_fixture("claude_code_structured") | {"structured_output": {"artifacts": "nope"}}
    provider, _ = _provider(tmp_path, CompletedProcess(returncode=0, stdout=json.dumps(payload)))
    with pytest.raises(StructuredOutputError) as info:
        provider.complete(_req(json_schema=SCHEMA))
    assert info.value.cost_usd == 1.05


def test_complete_budget_exhausted_reports_spend(tmp_path: Path, api_key: None) -> None:
    provider, _ = _provider(tmp_path, _out("claude_code_budget", returncode=1))
    with pytest.raises(ClaudeCodeBudgetExhausted, match="error_max_budget_usd") as info:
        provider.complete(_req())
    assert info.value.cost_usd == 8.0012
    assert info.value.retryable is False
    assert info.value.provider == "claude_code"


def test_complete_is_error_with_zero_exit(tmp_path: Path, api_key: None) -> None:
    provider, _ = _provider(tmp_path, _out("claude_code_error", returncode=0))
    with pytest.raises(ProviderError, match="529 overloaded") as info:
        provider.complete(_req())
    assert info.value.cost_usd == 0.42


def test_complete_nonzero_exit_with_success_json(tmp_path: Path, api_key: None) -> None:
    provider, _ = _provider(tmp_path, _out("claude_code_success", returncode=2, stderr="crash"))
    with pytest.raises(ProviderError, match="exit 2") as info:
        provider.complete(_req())
    assert info.value.cost_usd == 3.2175


def test_complete_crash_without_json_charges_worst_case(tmp_path: Path, api_key: None) -> None:
    """No JSON result means no cost report: the session may have spent its whole budget (plus a turn)."""
    provider, _ = _provider(tmp_path, CompletedProcess(returncode=-9, stdout="", stderr="Killed"))
    with pytest.raises(ProviderError, match="Killed") as info:
        provider.complete(_req(max_budget_usd=2.5))
    assert info.value.cost_usd == pytest.approx(provider.worst_case_cost(_req(max_budget_usd=2.5)))


def test_complete_timeout_charges_full_budget(tmp_path: Path, api_key: None) -> None:
    """A killed session reports no cost: it is charged its worst case, and stays a ``ClaudeCodeTimeout`` so the
    stages can run one continuation (``maf.stages.base.work_session``)."""
    provider, _ = _provider(tmp_path, ClaudeCodeTimeout("timed out", provider="claude_code"))
    with pytest.raises(ClaudeCodeTimeout, match="timed out") as info:
        provider.complete(_req(max_budget_usd=2.5))
    assert info.value.cost_usd == pytest.approx(2.5 + provider.turn_headroom_usd(_req()))
    assert not info.value.retryable


def test_complete_runner_crash_is_wrapped_and_charged(tmp_path: Path, api_key: None) -> None:
    provider, _ = _provider(tmp_path, RuntimeError("pipe broke"))
    with pytest.raises(ProviderError, match="pipe broke") as info:
        provider.complete(_req(max_budget_usd=2.5))
    assert info.value.cost_usd == pytest.approx(provider.worst_case_cost(_req(max_budget_usd=2.5)))


def test_complete_cli_not_startable_is_free(tmp_path: Path, api_key: None) -> None:
    provider, _ = _provider(tmp_path, ProviderError("cannot start Claude Code", provider="claude_code"))
    with pytest.raises(ProviderError, match="cannot start") as info:
        provider.complete(_req())
    assert info.value.cost_usd == 0.0


def test_complete_requires_budget_and_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider, runner = _provider(tmp_path, _out("claude_code_success"))
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        provider.complete(_req())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with pytest.raises(ValueError):
        provider.complete(_req(max_budget_usd=None))
    assert runner.calls == []


def test_worst_case_cost_is_the_budget_plus_one_turn(tmp_path: Path) -> None:
    provider, _ = _provider(tmp_path, _out("claude_code_success"), turn_context_tokens=100_000, turn_output_tokens=10_000)
    headroom = provider.turn_headroom_usd(_req())
    assert headroom == pytest.approx(100_000 * 5.00 / 1e6 + 10_000 * 20.00 / 1e6)  # opus: cache-write input rate
    assert provider.worst_case_cost(_req(max_budget_usd=3.75)) == pytest.approx(3.75 + headroom)
    with pytest.raises(ValueError):
        provider.worst_case_cost(_req(max_budget_usd=None))


# --- short private TMPDIR ------------------------------------------------------------------------


def test_scratch_tmpdir_is_short_and_random() -> None:
    first, second = scratch_tmpdir(), scratch_tmpdir(Path("/var/t"))
    assert re.fullmatch(r"/tmp/maf-[0-9a-f]{12}", str(first)) and len(str(first)) == 21
    assert second.parent == Path("/var/t") and re.fullmatch(r"maf-[0-9a-f]{12}", second.name)
    assert len({scratch_tmpdir() for _ in range(50)}) == 50  # nothing to predict from the workspace path


def test_provider_tmpdir_is_stable_per_instance_and_unrelated_to_the_workspace(tmp_path: Path, tmp_base: Path) -> None:
    provider, _ = _provider(tmp_path, _out("claude_code_success"))
    again, _ = _provider(tmp_path, _out("claude_code_success"))  # same workspace
    assert provider.tmpdir == provider.tmpdir and provider.tmpdir != again.tmpdir
    assert provider.tmpdir.parent == tmp_base and not provider.tmpdir.exists()  # picked, not created
    argv_settings = json.loads(_flag(provider.build_argv(_req()), "--settings"))
    assert argv_settings["sandbox"]["filesystem"]["allowWrite"][1] == provider.build_env()["TMPDIR"]


def test_max_tmpdir_bytes_follows_the_cli_child_tmpdir_budget() -> None:
    assert CLI_CHILD_TMPDIR_MAX_BYTES == 44
    assert max_tmpdir_bytes(1000) == 44 - len("/claude-1000") == 32
    assert max_tmpdir_bytes(0) == 35 and max_tmpdir_bytes(1234567890) == 26
    assert max_tmpdir_bytes() == 44 - len(f"/claude-{os.getuid()}")
    assert len(str(scratch_tmpdir())) <= max_tmpdir_bytes(4_294_967_294)  # the default fits any uid


def test_check_tmpdir_guards_length_and_absoluteness() -> None:
    limit = max_tmpdir_bytes()
    check_tmpdir(Path("/" + "x" * (limit - 1)))  # exactly the limit is fine
    with pytest.raises(ProviderError, match=f"{limit + 1} bytes long") as info:
        check_tmpdir(Path("/" + "x" * limit))
    assert (info.value.cost_usd, info.value.retryable) == (0.0, False)
    assert f"claude-{os.getuid()}" in str(info.value) and "claude_code_tmp_base" in str(info.value)
    with pytest.raises(ProviderError, match="bytes long"):
        check_tmpdir(Path("/" + "é" * (limit // 2)))  # fewer characters than the limit, but more bytes
    with pytest.raises(ProviderError, match="absolute"):
        check_tmpdir(Path("tmp/maf-x"))


def test_check_tmpdir_refuses_a_base_the_old_40_character_cap_allowed() -> None:
    """``/var/tmp/maf-work`` passed the former 40-character cap, but gives sandboxed commands a 50-byte TMPDIR."""
    tmpdir = scratch_tmpdir(Path("/var/tmp/maf-work"))
    assert len(str(tmpdir)) == 34 <= 40
    if len(os.fsencode(tmpdir)) > max_tmpdir_bytes():  # true for every uid of 3 or more digits
        with pytest.raises(ProviderError, match="shorter directory"):
            check_tmpdir(tmpdir)


def test_complete_refuses_long_tmpdir_before_spawning(tmp_path: Path, api_key: None) -> None:
    base = tmp_path / "a-long-temp-base"
    base.mkdir()
    provider, runner = _provider(tmp_path, _out("claude_code_success"), tmp_base=base, secrets_dir=tmp_path / "s")
    with pytest.raises(ProviderError, match="claude_code_tmp_base") as info:
        provider.complete(_req())
    assert not isinstance(info.value, SandboxUnavailable)
    assert (info.value.cost_usd, info.value.retryable) == (0.0, False)
    assert runner.calls == []
    assert not provider.tmpdir.exists() and not (tmp_path / "s").exists()  # nothing created, no key written


def test_ensure_private_dir_creates_0700(tmp_path: Path) -> None:
    target = tmp_path / "t"
    old = os.umask(0)
    try:
        ensure_private_dir(target)
    finally:
        os.umask(old)
    assert target.is_dir() and stat.S_IMODE(target.stat().st_mode) == 0o700


def test_ensure_private_dir_repairs_mode(tmp_path: Path) -> None:
    target = tmp_path / "t"
    target.mkdir(mode=0o755)
    target.chmod(0o775)
    ensure_private_dir(target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700


def test_ensure_private_dir_refuses_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o755)
    real.chmod(0o755)
    os.symlink(real, tmp_path / "t")
    with pytest.raises(ProviderError, match="symlink") as info:
        ensure_private_dir(tmp_path / "t")
    assert info.value.cost_usd == 0.0
    assert stat.S_IMODE(real.stat().st_mode) == 0o755  # the target was not touched


def test_ensure_private_dir_refuses_dangling_symlink_and_files(tmp_path: Path) -> None:
    os.symlink(tmp_path / "nowhere", tmp_path / "dangling")
    with pytest.raises(ProviderError, match="symlink"):
        ensure_private_dir(tmp_path / "dangling")
    (tmp_path / "file").write_text("x")
    with pytest.raises(ProviderError, match="refusing"):
        ensure_private_dir(tmp_path / "file")


def test_ensure_private_dir_refuses_other_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "t"
    target.mkdir(mode=0o755)
    target.chmod(0o755)
    real_uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: real_uid + 1)  # the directory now belongs to "someone else"
    with pytest.raises(ProviderError, match="owned by uid"):
        ensure_private_dir(target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o755  # no chmod on another user's directory


def test_ensure_private_dir_needs_the_parent(tmp_path: Path) -> None:
    with pytest.raises(ProviderError, match="cannot create"):
        ensure_private_dir(tmp_path / "missing" / "t")


def test_complete_moves_to_a_fresh_tmpdir_when_the_name_was_taken(tmp_path: Path, api_key: None) -> None:
    """Someone created the path first (here a symlink): the call goes ahead under a new random name."""
    provider, runner = _provider(tmp_path, _out("claude_code_success"), secrets_dir=tmp_path / "s")
    taken = provider.tmpdir
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o755)
    elsewhere.chmod(0o755)
    os.symlink(elsewhere, taken)
    assert provider.complete(_req()).cost_usd == 3.2175
    (call,) = runner.calls
    assert provider.tmpdir != taken and call["env"]["TMPDIR"] == str(provider.tmpdir)
    assert json.loads(_flag(call["argv"], "--settings"))["sandbox"]["filesystem"]["allowWrite"][1] == str(provider.tmpdir)
    assert stat.S_IMODE(provider.tmpdir.lstat().st_mode) == 0o700 and not provider.tmpdir.is_symlink()
    assert taken.is_symlink() and stat.S_IMODE(elsewhere.stat().st_mode) == 0o755  # not touched


def test_complete_refuses_when_a_fresh_tmpdir_is_unusable_too(
    tmp_path: Path, api_key: None, tmp_base: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    taken = tmp_base / "maf-000000000000"
    os.symlink(tmp_path, taken)
    monkeypatch.setattr("maf.providers.claude_code.scratch_tmpdir", lambda base: taken)
    provider, runner = _provider(tmp_path, _out("claude_code_success"), secrets_dir=tmp_path / "s")
    with pytest.raises(ProviderError, match="symlink") as info:
        provider.complete(_req())
    assert (info.value.cost_usd, runner.calls) == (0.0, [])


def test_tmpdir_is_removed_with_the_provider(tmp_path: Path, api_key: None) -> None:
    provider, _ = _provider(tmp_path, _out("claude_code_success"), secrets_dir=tmp_path / "s")
    provider.complete(_req())
    provider.complete(_req())  # the same directory is reused within an instance
    tmpdir = provider.tmpdir
    (tmpdir / "claude-1000").mkdir()
    assert tmpdir.is_dir()
    del provider
    gc.collect()
    assert not tmpdir.exists()


def test_sandbox_settings_allows_writes_to_tmpdir(tmp_path: Path) -> None:
    settings = sandbox_settings(tmp_path / "ws", tmpdir=Path("/tmp/maf-0123456789ab"))
    assert settings["sandbox"]["filesystem"]["allowWrite"] == [str((tmp_path / "ws").resolve()), "/tmp/maf-0123456789ab"]
    provider, _ = _provider(tmp_path, _out("claude_code_success"))
    argv_settings = json.loads(_flag(provider.build_argv(_req()), "--settings"))
    assert argv_settings["sandbox"]["filesystem"]["allowWrite"][1] == str(provider.tmpdir)


# --- sandbox failure detection -------------------------------------------------------------------

BRIDGE_FAILURE = "Sandbox is required but failed to initialize: Failed to create bridge sockets after 5 attempts"


@pytest.mark.parametrize(
    "text",
    [
        BRIDGE_FAILURE,
        "Error: Failed to create bridge sockets after 5 attempts",
        "bwrap: Creating new namespace failed: Operation not permitted",
        "Every command failed: `bwrap: setting up uid map: Permission denied`",
        "- The Bash sandbox failed to initialize, so no test was run.",
        "SANDBOX HAS FAILED TO INITIALISE",
        "Claude Code failed to initialize the OS sandbox for Bash.",
    ],
)
def test_sandbox_failure_signatures(text: str) -> None:
    assert sandbox_failure("all good\n", f"line one\n  {text}  \nline three") == text.strip()


@pytest.mark.parametrize(
    "text",
    [
        "Ran `make test` inside the sandbox: 12 passed.",
        "The sandbox is enabled; the allocator failed to initialize its pool when size=0 (fixed).",
        "The sandbox is enabled and the allocator failed to initialize its pool when size=0 (fixed).",
        "Uses bubblewrap (bwrap) for isolation.",
        "Wrote a sandbox allocator test: tlsf_init failed to initialize with a 0-byte arena, as specified.",
        "",
    ],
)
def test_sandbox_failure_ignores_ordinary_text(text: str) -> None:
    assert sandbox_failure(text) is None


def _result_with(**fields: Any) -> CompletedProcess:
    payload = load_provider_fixture("claude_code_success") | fields
    return CompletedProcess(returncode=0, stdout=json.dumps(payload))


def test_sandbox_failure_in_result_text_raises_with_real_cost(tmp_path: Path, api_key: None) -> None:
    body = f"## Summary\n\nCould not build.\n\n## Known Limitations\n\n- {BRIDGE_FAILURE}\n"
    provider, _ = _provider(tmp_path, _result_with(result=body))
    provider.sandbox_verified = True
    with pytest.raises(SandboxUnavailable, match="Failed to create bridge sockets") as info:
        provider.complete(_req())
    assert info.value.cost_usd == 3.2175  # total_cost_usd, not the worst case
    assert info.value.retryable is False and info.value.provider == "claude_code"
    assert str(provider.tmpdir) in str(info.value)
    assert provider.sandbox_verified is False


def test_sandbox_failure_in_stderr_of_failed_session(tmp_path: Path, api_key: None) -> None:
    provider, _ = _provider(tmp_path, _out("claude_code_error", returncode=1, stderr="bwrap: No permissions to create a new namespace"))
    with pytest.raises(SandboxUnavailable, match="bwrap: No permissions") as info:
        provider.complete(_req())
    assert info.value.cost_usd == 0.42


def test_sandbox_failure_without_json_charges_worst_case(tmp_path: Path, api_key: None) -> None:
    provider, _ = _provider(tmp_path, CompletedProcess(returncode=1, stdout="", stderr=BRIDGE_FAILURE))
    with pytest.raises(SandboxUnavailable) as info:
        provider.complete(_req(max_budget_usd=2.5))
    assert info.value.cost_usd == pytest.approx(provider.worst_case_cost(_req(max_budget_usd=2.5)))


def test_sandbox_failure_beats_structured_output_error(tmp_path: Path, api_key: None) -> None:
    """The fix pass tolerates StructuredOutputError; a broken sandbox must not hide behind one."""
    provider, _ = _provider(tmp_path, _result_with(structured_output={"summary": BRIDGE_FAILURE}))
    with pytest.raises(SandboxUnavailable) as info:
        provider.complete(_req(json_schema=SCHEMA))
    assert not isinstance(info.value, StructuredOutputError)
    assert info.value.cost_usd == 3.2175


def test_sandbox_failure_nested_in_structured_output(tmp_path: Path, api_key: None) -> None:
    report = {"artifacts": ["x"], "tests_passed": False, "notes": [{"detail": "bwrap: execvp make: No such file"}]}
    provider, _ = _provider(tmp_path, _result_with(structured_output=report))
    with pytest.raises(SandboxUnavailable, match="execvp"):
        provider.complete(_req(json_schema=SCHEMA))


def test_ordinary_failure_is_not_a_sandbox_failure(tmp_path: Path, api_key: None) -> None:
    provider, _ = _provider(tmp_path, _out("claude_code_budget", returncode=1))
    with pytest.raises(ProviderError) as info:
        provider.complete(_req())
    assert not isinstance(info.value, SandboxUnavailable)


# --- preflight -----------------------------------------------------------------------------------


def _preflight_runner(
    answer: Callable[[bytes], Any],
    cost: float = 0.0412,
    seen: list[dict[str, Any]] | None = None,
    writes: bool = True,
) -> Callable[..., CompletedProcess]:
    """A fake CLI that reads the probe file from its cwd and answers ``structured_output={"digest": answer(data)}``
    (``answer`` may return a whole ``structured_output`` dict instead). With ``writes``, it leaves what
    ``PREFLIGHT_COMMAND``'s ``tee`` writes in a working sandbox."""

    def runner(argv: list[str], stdin: str, cwd: Path, env: dict[str, str], timeout: float) -> CompletedProcess:
        data = (cwd / PREFLIGHT_FILE).read_bytes()
        if seen is not None:
            seen.append({"argv": list(argv), "stdin": stdin, "size": len(data), "env": env})
        if writes:
            line = f"{hashlib.sha256(data).hexdigest()}  {env['TMPDIR']}/claude-1000/maf-preflight\n"
            (cwd / PREFLIGHT_OUTPUT).write_text(line)
        value = answer(data)
        structured = value if isinstance(value, dict) else {"digest": value}
        payload = load_provider_fixture("claude_code_structured") | {
            "structured_output": structured,
            "total_cost_usd": cost,
            "num_turns": 3,
        }
        return CompletedProcess(returncode=0, stdout=json.dumps(payload))

    return runner


def _preflight_provider(tmp_path: Path, runner: Callable[..., CompletedProcess]) -> ClaudeCodeProvider:
    return ClaudeCodeProvider(
        tmp_path / "ws",
        executable=Path("/opt/claude"),
        allowed_tools=TOOLS,
        runner=runner,
        secrets_dir=tmp_path / "secrets",
    )


def test_preflight_request_is_cheap_and_structured() -> None:
    request = preflight_request(MODEL, 0.15)
    assert request.effort == "low"
    assert request.max_budget_usd == 0.15
    assert request.max_output_tokens <= 1_024
    assert request.json_schema == PREFLIGHT_SCHEMA == {
        "type": "object",
        "properties": {"digest": {"type": "string"}},
        "required": ["digest"],
        "additionalProperties": False,
    }
    assert PREFLIGHT_COMMAND in request.messages[0].content
    # read the workspace, write $TMPDIR (only a copy that worked is hashed), write the workspace
    assert PREFLIGHT_COMMAND == (
        'cp .maf/preflight.bin "$TMPDIR/maf-preflight" && sha256sum "$TMPDIR/maf-preflight" | tee .maf/preflight.out'
    )


def test_preflight_budget_scales_with_the_model() -> None:
    opus, fable = preflight_budget_usd("claude-opus-5-5"), preflight_budget_usd("claude-fable-5-1")
    assert opus == pytest.approx(3 * (25_000 * 5.00 + 1_024 * 20.00) / 1e6)  # about $0.44
    assert fable == pytest.approx(3 * (25_000 * 12.50 + 1_024 * 50.00) / 1e6)  # about $1.09
    assert 13_000 * 12.50 / 1e6 > 0.15  # a Fable first turn alone overran the former flat $0.15
    assert preflight_budget_usd("gemini-3.8-flash") == 0.15  # the floor


def test_preflight_passes_on_matching_digest(tmp_path: Path, api_key: None) -> None:
    seen: list[dict[str, Any]] = []
    provider = _preflight_provider(tmp_path, _preflight_runner(lambda d: hashlib.sha256(d).hexdigest(), seen=seen))
    assert provider.sandbox_verified is False
    result = provider.preflight(MODEL, 0.15)
    assert provider.sandbox_verified is True
    assert result.cost_usd == 0.0412
    assert not (tmp_path / "ws" / PREFLIGHT_FILE).exists()  # deleted afterwards
    assert not (tmp_path / "ws" / PREFLIGHT_OUTPUT).exists()
    (call,) = seen
    assert call["size"] == 4096
    assert PREFLIGHT_COMMAND in call["stdin"]
    argv = call["argv"]
    assert (_flag(argv, "--effort"), _flag(argv, "--max-budget-usd"), _flag(argv, "--model")) == ("low", "0.1500", MODEL)
    assert json.loads(_flag(argv, "--json-schema")) == PREFLIGHT_SCHEMA


def test_preflight_never_writes_through_a_planted_symlink(tmp_path: Path, api_key: None) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    probe = tmp_path / "ws" / PREFLIGHT_FILE
    probe.parent.mkdir(parents=True)
    os.symlink(victim, probe)
    os.symlink(victim, tmp_path / "ws" / PREFLIGHT_OUTPUT)
    provider = _preflight_provider(tmp_path, _preflight_runner(lambda d: hashlib.sha256(d).hexdigest()))
    provider.preflight(MODEL, 0.15)
    assert victim.read_text() == "keep me"
    assert not probe.exists() and not probe.is_symlink()
    assert not (tmp_path / "ws" / PREFLIGHT_OUTPUT).is_symlink()


def test_preflight_default_budget_scales_with_the_model(tmp_path: Path, api_key: None) -> None:
    seen: list[dict[str, Any]] = []
    provider = _preflight_provider(tmp_path, _preflight_runner(lambda d: hashlib.sha256(d).hexdigest(), seen=seen))
    provider.preflight(MODEL)
    assert _flag(seen[0]["argv"], "--max-budget-usd") == format_budget(preflight_budget_usd(MODEL)) == "0.4364"


def test_preflight_fails_when_sandboxed_bash_cannot_write_the_workspace(tmp_path: Path, api_key: None) -> None:
    """``tee`` could not write (read-only workspace): the digest still reaches the answer, but not the file."""
    runner = _preflight_runner(lambda d: hashlib.sha256(d).hexdigest(), cost=0.05, writes=False)
    provider = _preflight_provider(tmp_path, runner)
    provider.sandbox_verified = True
    with pytest.raises(SandboxUnavailable, match=r"\.maf/preflight\.out does not hold it") as info:
        provider.preflight(MODEL, 0.15)
    assert (info.value.cost_usd, info.value.retryable, provider.sandbox_verified) == (0.05, False, False)
    assert not (tmp_path / "ws" / PREFLIGHT_FILE).exists()


def test_preflight_fails_when_the_written_digest_is_wrong(tmp_path: Path, api_key: None) -> None:
    def runner(argv: list[str], stdin: str, cwd: Path, env: dict[str, str], timeout: float) -> CompletedProcess:
        digest = hashlib.sha256((cwd / PREFLIGHT_FILE).read_bytes()).hexdigest()
        (cwd / PREFLIGHT_OUTPUT).write_text("0" * 64 + "  x\n")  # e.g. a stale copy
        payload = load_provider_fixture("claude_code_structured") | {"structured_output": {"digest": digest}}
        return CompletedProcess(returncode=0, stdout=json.dumps(payload))

    with pytest.raises(SandboxUnavailable, match="does not hold it"):
        _preflight_provider(tmp_path, runner).preflight(MODEL, 0.15)
    assert not (tmp_path / "ws" / PREFLIGHT_OUTPUT).exists()


def test_preflight_reports_a_tmpdir_write_failure(tmp_path: Path, api_key: None) -> None:
    """``cp`` into ``$TMPDIR`` failed, so nothing was hashed: the answer quotes the error."""
    error = "cp: cannot create regular file '/tmp/maf-x/claude-1000/maf-preflight': Read-only file system"
    provider = _preflight_provider(tmp_path, _preflight_runner(lambda d: error, writes=False))
    with pytest.raises(SandboxUnavailable, match="Read-only file system"):
        provider.preflight(MODEL, 0.15)


def test_written_digest(tmp_path: Path) -> None:
    h = "cd" * 32
    (tmp_path / "out").write_text(f"{h.upper()}  /tmp/maf-x/claude-1000/maf-preflight\n")
    assert written_digest(tmp_path / "out") == h
    (tmp_path / "none").write_text("sha256sum: No such file\n")
    os.symlink(tmp_path / "out", tmp_path / "link")
    os.mkfifo(tmp_path / "fifo")
    for name in ("none", "link", "fifo", "missing"):
        assert written_digest(tmp_path / name) is None


def test_preflight_bytes_are_random(tmp_path: Path, api_key: None) -> None:
    contents: list[bytes] = []

    def answer(data: bytes) -> str:
        contents.append(data)
        return hashlib.sha256(data).hexdigest()

    provider = _preflight_provider(tmp_path, _preflight_runner(answer))
    provider.preflight(MODEL, 0.15)
    provider.preflight(MODEL, 0.15)
    assert contents[0] != contents[1]


@pytest.mark.parametrize(
    "shape",
    [
        lambda h: h.upper(),
        lambda h: f"{h}  .maf/preflight.bin",
        lambda h: f"The digest is `{h}`.",
    ],
)
def test_preflight_tolerates_digest_formatting(tmp_path: Path, api_key: None, shape: Callable[[str], str]) -> None:
    provider = _preflight_provider(tmp_path, _preflight_runner(lambda d: shape(hashlib.sha256(d).hexdigest())))
    provider.preflight(MODEL, 0.15)
    assert provider.sandbox_verified is True


@pytest.mark.parametrize(
    "answer",
    [
        lambda d: hashlib.sha256(d + b"x").hexdigest(),  # wrong file or invented digest
        lambda d: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",  # sha256 of nothing
        lambda d: "I could not run the command.",
        lambda d: "",
    ],
)
def test_preflight_mismatch_raises_sandbox_unavailable(
    tmp_path: Path, api_key: None, answer: Callable[[bytes], str]
) -> None:
    provider = _preflight_provider(tmp_path, _preflight_runner(answer, cost=0.05))
    provider.sandbox_verified = True
    with pytest.raises(SandboxUnavailable, match="preflight failed") as info:
        provider.preflight(MODEL, 0.15)
    assert info.value.cost_usd == 0.05
    assert info.value.retryable is False
    assert provider.sandbox_verified is False
    assert not (tmp_path / "ws" / PREFLIGHT_FILE).exists()


def test_preflight_missing_digest_raises_sandbox_unavailable(tmp_path: Path, api_key: None) -> None:
    provider = _preflight_provider(tmp_path, _preflight_runner(lambda d: {"answer": "no"}, cost=0.06))
    with pytest.raises(SandboxUnavailable, match="no digest") as info:
        provider.preflight(MODEL, 0.15)
    assert info.value.cost_usd == 0.06
    assert isinstance(info.value.__cause__, StructuredOutputError)
    assert not (tmp_path / "ws" / PREFLIGHT_FILE).exists()


def test_preflight_reporting_the_sandbox_error(tmp_path: Path, api_key: None) -> None:
    provider = _preflight_provider(tmp_path, _preflight_runner(lambda d: BRIDGE_FAILURE, cost=0.07))
    with pytest.raises(SandboxUnavailable, match="bridge sockets") as info:
        provider.preflight(MODEL, 0.15)
    assert info.value.cost_usd == 0.07
    assert not (tmp_path / "ws" / PREFLIGHT_FILE).exists()


def test_preflight_out_of_budget_says_so(tmp_path: Path, api_key: None) -> None:
    """A too-small preflight cap is a budget problem, not a broken sandbox; the message must say which."""
    provider, _ = _provider(tmp_path, _out("claude_code_budget", returncode=1))
    with pytest.raises(ProviderError, match="preflight ran out of its budget on claude-opus-5-5") as info:
        provider.preflight(MODEL, 0.15)
    assert not isinstance(info.value, (SandboxUnavailable, ClaudeCodeBudgetExhausted))
    assert "claude_code_preflight_budget_usd" in str(info.value) and "error_max_budget_usd" in str(info.value)
    assert (info.value.cost_usd, info.value.retryable) == (8.0012, False)
    assert isinstance(info.value.__cause__, ClaudeCodeBudgetExhausted)
    assert provider.sandbox_verified is False
    assert not (tmp_path / "ws" / PREFLIGHT_FILE).exists()


def test_preflight_other_failures_propagate_unchanged(tmp_path: Path, api_key: None) -> None:
    provider, _ = _provider(tmp_path, _out("claude_code_error", returncode=1))
    with pytest.raises(ProviderError, match="529 overloaded") as info:
        provider.preflight(MODEL, 0.15)
    assert type(info.value) is ProviderError
    assert provider.sandbox_verified is False
    assert not (tmp_path / "ws" / PREFLIGHT_FILE).exists()


def test_preflight_goes_through_the_given_call(tmp_path: Path) -> None:
    """Stages pass a metered ``ctx.call``; the provider's own ``complete`` is then not used directly."""
    requests: list[CompletionRequest] = []
    probe = tmp_path / "ws" / PREFLIGHT_FILE

    def call(request: CompletionRequest) -> CompletionResult:
        requests.append(request)
        digest = hashlib.sha256(probe.read_bytes()).hexdigest()
        (tmp_path / "ws" / PREFLIGHT_OUTPUT).write_text(f"{digest}  -\n")
        return CompletionResult(
            text="", parsed={"digest": digest}, usage=Usage(), cost_usd=0.03, model=request.model, provider="claude_code"
        )

    provider, runner = _provider(tmp_path, _out("claude_code_success"))
    result = provider.preflight(MODEL, 0.2, call=call)
    assert runner.calls == []
    assert requests == [preflight_request(MODEL, 0.2)]
    assert result.cost_usd == 0.03 and provider.sandbox_verified is True
    assert not probe.exists() and not (tmp_path / "ws" / PREFLIGHT_OUTPUT).exists()


def test_reported_digest() -> None:
    h = "ab" * 32
    assert reported_digest({"digest": h}) == h
    assert reported_digest({"digest": f"{h.upper()}  file"}) == h
    for bad in (None, {}, {"digest": 42}, {"digest": "ab" * 31}, {"digest": "ab" * 33}):
        assert reported_digest(bad) is None


# --- default runner (harmless local subprocesses, never claude) ---------------------------------


def test_subprocess_runner_round_trip(tmp_path: Path) -> None:
    script = "import os, sys; print(sys.stdin.read().upper()); print(os.getcwd(), file=sys.stderr); sys.exit(3)"
    proc = subprocess_runner([sys.executable, "-c", script], "hello", tmp_path, {"PATH": "/usr/bin:/bin"}, 30.0)
    assert proc.returncode == 3
    assert proc.stdout.strip() == "HELLO"
    assert proc.stderr.strip() == str(tmp_path)


def test_subprocess_runner_timeout_kills_process_group(tmp_path: Path) -> None:
    marker = tmp_path / "child-survived"
    child = tmp_path / "child.py"
    child.write_text(f"import time\ntime.sleep(1.5)\nopen({str(marker)!r}, 'w').close()\n")
    parent = tmp_path / "parent.py"
    parent.write_text(
        f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, {str(child)!r}])\ntime.sleep(30)\n"
    )
    started = time.monotonic()
    with pytest.raises(ClaudeCodeTimeout, match="timed out"):
        subprocess_runner([sys.executable, str(parent)], "", tmp_path, {"PATH": "/usr/bin:/bin"}, 0.5)
    assert time.monotonic() - started < 10
    time.sleep(2.0)
    assert not marker.exists()  # the grandchild died with the group


def test_subprocess_runner_missing_executable(tmp_path: Path) -> None:
    with pytest.raises(ProviderError, match="cannot start"):
        subprocess_runner([str(tmp_path / "no-claude")], "", tmp_path, {}, 5.0)
