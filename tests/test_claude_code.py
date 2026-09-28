"""Claude Code headless adapter. A fake ``Runner`` stands in for the CLI; the real ``claude`` never runs."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from conftest import load_provider_fixture

from maf.providers.base import CompletionRequest, Message, ProviderError, StructuredOutputError
from maf.providers.claude_code import (
    DISALLOWED_TOOLS,
    SENSITIVE_READ_PATHS,
    ClaudeCodeProvider,
    ClaudeCodeTimeout,
    CompletedProcess,
    sandbox_settings,
    subprocess_runner,
)

MODEL = "claude-opus-5-5"
TOOLS = ("Read(./**)", "Edit(./**)", "Bash(make *)")
SCHEMA = {
    "type": "object",
    "properties": {"artifacts": {"type": "array", "items": {"type": "string"}}, "tests_passed": {"type": "boolean"}},
    "required": ["artifacts", "tests_passed"],
}


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
            "autoAllowBashIfSandboxed": False,
            "filesystem": {"allowWrite": [str(tmp_path.resolve())], "denyRead": ["~/.ssh", "/srv/vault/"]},
            "network": {"allowedDomains": []},
        },
    }


def test_default_sandbox_denies_credentials(tmp_path: Path) -> None:
    settings = sandbox_settings(tmp_path)
    assert "~/.ssh" in SENSITIVE_READ_PATHS
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
    assert {"Read(./**)", "Edit(./**)", "Write(./**)"} <= set(listed)


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
        "--bare",
        "--settings", json.dumps(sandbox_settings(tmp_path / "ws"), separators=(",", ":")),
        "--allowedTools", *TOOLS,
        "--disallowedTools", "WebFetch", "WebSearch",
    ]  # fmt: skip
    assert "--add-dir" not in argv
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
    tmp_path: Path, api_key: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp-must-not-leak")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-must-not-leak")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    provider, _ = _provider(
        tmp_path, _out("claude_code_success"), extra_env={"FOO": "bar"}, path_prepend=[Path("/venv/bin")]
    )
    env = provider.build_env()
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-test-not-real"
    assert env["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"] == "1"
    assert env["LC_ALL"] == "C.UTF-8"
    for dropped in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "CLAUDECODE", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY"):
        assert dropped not in env
    ws = (tmp_path / "ws").resolve()
    assert env["TMPDIR"] == str(ws / ".maf" / "tmp")
    assert env["MPLCONFIGDIR"] == str(ws / ".maf" / "mpl")
    assert env["PATH"].startswith("/venv/bin:")
    assert env["FOO"] == "bar"


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


def test_complete_success(tmp_path: Path, api_key: None) -> None:
    provider, runner = _provider(tmp_path, _out("claude_code_success"))
    req = _req(system="sys")
    result = provider.complete(req)
    call = runner.calls[0]
    assert call["argv"] == provider.build_argv(req)
    assert call["stdin"] == "Implement the allocator."
    assert call["cwd"] == tmp_path / "ws"
    assert call["timeout"] == 120.0
    assert "OPENAI_API_KEY" not in call["env"]
    assert (tmp_path / "ws" / ".maf" / "tmp").is_dir()
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
    with pytest.raises(ProviderError, match="error_max_budget_usd") as info:
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
    provider, _ = _provider(tmp_path, ClaudeCodeTimeout("timed out", provider="claude_code"))
    with pytest.raises(ProviderError, match="timed out") as info:
        provider.complete(_req(max_budget_usd=2.5))
    assert info.value.cost_usd == pytest.approx(2.5 + provider.turn_headroom_usd(_req()))


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
