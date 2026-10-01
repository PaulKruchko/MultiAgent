"""CLI tests: argparse tree, exit codes, and each command against a real temp vault with scripted backends."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from maf import cli
from maf.config import Settings
from maf.handoff import HandoffInvalid, HandoffKind
from maf.ledger import BudgetExceeded
from maf.pipeline import Pipeline
from maf.stages.base import StageContext, StageOutput
from maf.types import STAGE_ORDER, RunStatus, StageName


@dataclass
class Backend:
    """Minimal stage backend: no notes, optionally raising once. ``unresolved`` > 0 makes a crosscheck loop back;
    ``unmet`` lines are the acceptance criteria a final reports as not met. Ingestion picks ``mode``."""

    name: StageName
    error: BaseException | None = None
    unresolved: int = 0
    unmet: list[str] = field(default_factory=list)
    mode: str = "code"
    calls: int = 0

    def run_stage(self, ctx: StageContext) -> StageOutput:
        self.calls += 1
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        if self.name == "ingestion":
            return StageOutput(notes=[], index_updates={"mode": self.mode})
        if self.name == "crosscheck":
            return StageOutput(
                notes=[], index_updates={"unresolved_critical": self.unresolved}, loop_back=self.unresolved > 0
            )
        if self.name == "final":
            return StageOutput(notes=[], index_updates={"criteria_unmet": len(self.unmet), "unmet_criteria": self.unmet})
        return StageOutput(notes=[])


class Env:
    def __init__(self, tmp_path: Path, fake_providers: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        self.vault = tmp_path / "vault"
        self.workspaces = tmp_path / "ws"
        self.backends: dict[StageName, Backend] = {s: Backend(s) for s in STAGE_ORDER}
        self.settings: Settings | None = None

        def make_pipeline(settings: Settings) -> Pipeline:
            self.settings = settings
            return Pipeline(settings, providers_factory=fake_providers.factory(), backends=self.backends)

        monkeypatch.setattr(cli, "make_pipeline", make_pipeline)

    def argv(self, *args: str) -> list[str]:
        return ["--vault", str(self.vault), "--workspaces", str(self.workspaces), *args]

    def pipeline(self) -> Pipeline:
        assert self.settings is not None
        return Pipeline(self.settings, backends=self.backends, providers_factory=lambda s, w: None)  # type: ignore[arg-type,return-value]


@pytest.fixture
def env(tmp_path: Path, fake_providers: Any, monkeypatch: pytest.MonkeyPatch) -> Env:
    return Env(tmp_path, fake_providers, monkeypatch)


# --------------------------------------------------------------------------- parser and exit codes


def test_parser_run_options() -> None:
    args = cli.build_parser().parse_args(
        ["run", "brief text", "--file", "a.pdf", "--file", "b.png", "--budget", "12.5", "--tier", "max", "--review", "--no-wait"]
    )
    assert args.command == "run"
    assert args.brief == "brief text"
    assert args.files == [Path("a.pdf"), Path("b.png")]
    assert (args.budget, args.tier, args.review, args.no_wait) == (12.5, "max", True, True)


def test_parser_defaults_leave_settings_in_charge() -> None:
    args = cli.build_parser().parse_args(["run", "b"])
    assert (args.budget, args.tier, args.review, args.no_wait, args.files) == (None, None, None, False, [])
    assert (args.vault, args.workspaces, args.config) == (None, None, None)


@pytest.mark.parametrize(
    "argv",
    [
        ["--vault", "/v", "status", "r1"],
        ["status", "r1", "--vault", "/v"],
    ],
)
def test_global_options_work_before_or_after_the_subcommand(argv: list[str]) -> None:
    args = cli.build_parser().parse_args(argv)
    assert args.vault == Path("/v")
    assert args.run_id == "r1"


def test_parser_other_commands() -> None:
    p = cli.build_parser()
    r = p.parse_args(["resume", "r1", "--note", "go", "--budget", "40"])
    assert (r.command, r.run_id, r.note, r.budget, r.extra_round) == ("resume", "r1", "go", 40.0, False)
    assert p.parse_args(["resume", "r1", "--extra-round"]).extra_round is True
    s = p.parse_args(["status", "r1", "--json"])
    assert (s.command, s.json) == ("status", True)
    ls = p.parse_args(["list", "--limit", "5"])
    assert (ls.command, ls.limit, ls.json) == ("list", 5, False)
    sv = p.parse_args(["serve", "--port", "9000"])
    assert (sv.command, sv.host, sv.port) == ("serve", None, 9000)
    ex = p.parse_args(["export", "r1"])
    assert (ex.command, ex.run_id) == ("export", "r1")


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["bogus"],
        ["run"],
        ["run", "b", "--budget", "0"],
        ["run", "b", "--budget", "abc"],
        ["run", "b", "--budget", "inf"],
        ["run", "b", "--budget", "nan"],
        ["resume", "r", "--budget", "inf"],
        ["run", "b", "--tier", "ultra"],
        ["list", "--limit", "0"],
        ["serve", "--port", "70000"],
        ["export"],
    ],
)
def test_usage_errors_exit_2(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(argv) == cli.EXIT_USAGE
    assert capsys.readouterr().err


def test_help_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--help"]) == 0
    assert "resume" in capsys.readouterr().out


def test_version_prints_the_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    import maf

    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"maf {maf.__version__}"


def test_parser_review_and_no_review() -> None:
    p = cli.build_parser()
    assert p.parse_args(["run", "b", "--review"]).review is True
    assert p.parse_args(["run", "b", "--no-review"]).review is False
    assert p.parse_args(["run", "b"]).review is None  # unset: the config's review decides


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["list", "--help"], ["--json", "spent_usd", "--limit N", "(default: 20)"]),
        (["status", "--help"], ["--json", "run.md frontmatter", "JSON object"]),
        (["run", "--help"], ["--review, --no-review", "config"]),
    ],
)
def test_option_help_strings(argv: list[str], expected: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(argv) == 0
    out = " ".join(capsys.readouterr().out.split())
    for text in expected:
        assert text in out, text


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (RunStatus.COMPLETED, 0),
        (RunStatus.AWAITING_REVIEW, 0),
        (RunStatus.PENDING, 0),
        (RunStatus.RUNNING, 0),
        (RunStatus.COMPLETED_WITH_ISSUES, 2),
        (RunStatus.FAILED, 1),
        (RunStatus.BUDGET_EXCEEDED, 1),
    ],
)
def test_exit_code_for(status: RunStatus, code: int) -> None:
    assert cli.exit_code_for(status) == code


def test_exit_code_for_covers_every_status() -> None:
    for status in RunStatus:
        assert cli.exit_code_for(status) in (cli.EXIT_OK, cli.EXIT_FAILED, cli.EXIT_WITH_ISSUES)


def test_bad_config_file_exits_2(env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text("budget_usd: [unclosed\n", encoding="utf-8")
    assert cli.main(["--config", str(bad), "list"]) == cli.EXIT_USAGE
    assert "configuration" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("text", "named"),
    [
        ("output_limits: {claude_code_budget: 20}\n", "claude_code_budget (allowed keys: chatgpt, gemini, claude"),
        ("stage_model_overrides: {final: {claude: claude-sonnet-9}}\n", "unknown model 'claude-sonnet-9'"),
        ("claude_code_tmp_base: /var/tmp/a-much-too-long-base\n", "is too long"),
        ("claude_code_tools: [Read]\n", "need a path scope"),
    ],
)
def test_invalid_config_fails_before_any_run_is_created(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str], text: str, named: str
) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(text, encoding="utf-8")
    assert cli.main(["--config", str(cfg), *env.argv("run", "x")]) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert "bad configuration" in err and named in err
    assert env.settings is None and not env.vault.exists() and not env.workspaces.exists()


def test_missing_python_executable_is_one_warning_line_and_the_run_goes_on(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "old-venv" / "bin" / "python"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"python_executable: {missing}\n", encoding="utf-8")
    assert cli.main(["--config", str(cfg), *env.argv("run", "x")]) == 0
    captured = capsys.readouterr()
    warnings = [line for line in captured.err.splitlines() if "does not exist" in line]
    assert warnings == [f"maf: warning: python_executable {missing} does not exist, so Claude Code runs without that "
                        "venv first on its PATH (simulations get the system python3 and its packages); set "
                        "python_executable in the config to the venv's python, or recreate the venv"]
    assert env.pipeline().status(captured.out.splitlines()[0]).status == RunStatus.COMPLETED


def test_no_warning_with_the_default_python_executable(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(env.argv("list")) == 0
    assert "warning" not in capsys.readouterr().err


def test_global_paths_reach_settings(env: Env) -> None:
    assert cli.main(env.argv("list")) == 0
    assert env.settings is not None
    assert env.settings.vault_path == env.vault
    assert env.settings.workspaces_path == env.workspaces


# --------------------------------------------------------------------------- run


def test_run_completes_and_prints_run_id_progress_and_final_path(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(env.argv("run", "Portable allocator", "--budget", "5")) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    run_id = lines[0]
    assert run_id.endswith("portable-allocator")
    assert any(f"[{run_id}] ingestion" in line for line in lines[1:])
    assert lines[-1] == str(env.vault / "runs" / run_id / "05-final.md")
    index = env.pipeline().status(run_id)
    assert index.status == RunStatus.COMPLETED
    assert index.budget_usd == 5.0


def test_run_with_review_pauses_and_exits_0(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(env.argv("run", "x", "--review")) == 0
    out = capsys.readouterr().out
    assert "awaiting review" in out and "maf resume" in out


def test_no_review_overrides_review_true_in_the_config(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("review: true\n", encoding="utf-8")
    assert cli.main(["--config", str(cfg), *env.argv("run", "paused by the config")]) == 0
    paused = capsys.readouterr().out.splitlines()[0]
    assert env.pipeline().status(paused).status == RunStatus.AWAITING_REVIEW
    assert cli.main(["--config", str(cfg), *env.argv("run", "straight through", "--no-review")]) == 0
    straight = capsys.readouterr().out.splitlines()[0]
    index = env.pipeline().status(straight)
    assert (index.status, index.review) == (RunStatus.COMPLETED, False)


def test_run_failure_exits_1(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    env.backends["execution"].error = HandoffInvalid(HandoffKind.EXECUTION, ["missing section 'Summary'"])
    assert cli.main(env.argv("run", "x")) == cli.EXIT_FAILED
    assert "failed: execution" in capsys.readouterr().err


def test_run_budget_exceeded_exits_1_then_resume_exits_0(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    env.backends["strategy"].error = BudgetExceeded(cap_usd=1.0, spent_usd=0.9, requested_usd=0.5, what="strategy")
    assert cli.main(env.argv("run", "x", "--budget", "1")) == cli.EXIT_FAILED
    captured = capsys.readouterr()
    run_id = captured.out.splitlines()[0]
    assert "--budget" in captured.err
    assert cli.main(env.argv("resume", run_id, "--budget", "3", "--note", "carry on")) == 0
    index = env.pipeline().status(run_id)
    assert (index.status, index.budget_usd) == (RunStatus.COMPLETED, 3.0)
    assert (Path(index.workspace) / ".maf" / "review-note.md").read_text(encoding="utf-8").strip() == "carry on"


def test_run_completed_with_issues_exits_2_with_a_clear_final_line(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    env.backends["crosscheck"].unresolved = 3
    assert cli.main(env.argv("run", "x")) == cli.EXIT_WITH_ISSUES == 2
    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    run_id = lines[0]
    assert any("run completed with issues: 3 unresolved critical issue(s)" in line for line in lines)
    assert lines[-1] == str(env.vault / "runs" / run_id / "05-final.md")
    last = captured.err.strip().splitlines()[-1]
    assert last.startswith("completed with issues: 3 unresolved critical issue(s) after the cross-check loop cap")
    assert f"maf resume {run_id} --extra-round" in last
    assert env.pipeline().status(run_id).status == RunStatus.COMPLETED_WITH_ISSUES
    assert env.backends["execution"].calls == 3


def test_run_with_unmet_criteria_exits_2_and_names_them(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    env.backends["final"].unmet = ["AC-2 [unmet]: every ITER claim is sourced", "clean-room [unmet]: rebuild"]
    assert cli.main(env.argv("run", "x")) == cli.EXIT_WITH_ISSUES
    captured = capsys.readouterr()
    run_id = captured.out.splitlines()[0]
    assert "run completed with issues: 2 acceptance criteria not met (AC-2, clean-room)" in captured.out
    final = env.vault / "runs" / run_id / "05-final.md"
    assert captured.err.strip().splitlines()[-1] == (
        f"completed with issues: 2 acceptance criteria not met (AC-2, clean-room); see {final} "
        f"(spent $0.0000 of $25.00; one more pass: maf resume {run_id} --extra-round)"
    )

    assert cli.main(env.argv("status", run_id)) == 0
    out = capsys.readouterr().out
    assert "status:   completed_with_issues" in out and "unresolved critical" not in out
    assert "criteria unmet: 2\n  - AC-2 [unmet]: every ITER claim is sourced\n  - clean-room [unmet]: rebuild\n" in out
    assert cli.main(env.argv("resume", run_id)) == cli.EXIT_WITH_ISSUES  # refused like any completed_with_issues run


def test_both_kinds_of_issue_share_one_stderr_line(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    env.backends["crosscheck"].unresolved = 1
    env.backends["final"].unmet = ["clean-room [unmet]: rebuild"]
    assert cli.main(env.argv("run", "x")) == cli.EXIT_WITH_ISSUES
    captured = capsys.readouterr()
    run_id = captured.out.splitlines()[0]
    last = captured.err.strip().splitlines()[-1]
    assert last.startswith("completed with issues: 1 unresolved critical issue(s) after the cross-check loop cap; ")
    assert f"; 1 acceptance criterion not met (clean-room); see {env.vault / 'runs' / run_id / '05-final.md'}" in last


def test_resume_refuses_completed_with_issues_without_extra_round(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    env.backends["crosscheck"].unresolved = 1
    cli.main(env.argv("run", "x"))
    run_id = capsys.readouterr().out.splitlines()[0]
    calls = {name: b.calls for name, b in env.backends.items()}

    assert cli.main(env.argv("resume", run_id, "--note", "try harder")) == cli.EXIT_WITH_ISSUES
    err = capsys.readouterr().err
    assert f"maf: {run_id} is completed_with_issues; nothing to resume" in err
    assert {name: b.calls for name, b in env.backends.items()} == calls
    assert not (Path(env.pipeline().status(run_id).workspace) / ".maf" / "review-note.md").exists()

    env.backends["crosscheck"].unresolved = 0
    assert cli.main(env.argv("resume", run_id, "--extra-round", "--note", "try harder")) == cli.EXIT_OK
    index = env.pipeline().status(run_id)
    assert (index.status, index.round) == (RunStatus.COMPLETED, 4)
    assert (env.backends["execution"].calls, env.backends["final"].calls) == (4, 2)



def test_legacy_completed_run_with_open_criticals_can_take_an_extra_round(
    env: Env, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run.md written before completed_with_issues existed (``status: completed``, ``unresolved_critical: 9``)
    shows as completed_with_issues everywhere, and ``--extra-round`` is allowed on it."""
    cli.main(env.argv("run", "legacy"))
    run_id = capsys.readouterr().out.splitlines()[0]
    run_md = env.vault / "runs" / run_id / "run.md"
    text = run_md.read_text(encoding="utf-8")
    assert "\nstatus: completed\n" in text and "\nunresolved_critical: 0\n" in text
    run_md.write_text(text.replace("\nunresolved_critical: 0\n", "\nunresolved_critical: 9\n"), encoding="utf-8")

    assert cli.main(env.argv("status", run_id)) == 0
    out = capsys.readouterr().out
    assert "status:   completed_with_issues" in out and "unresolved critical: 9" in out
    assert cli.main(env.argv("list", "--json")) == 0
    assert json.loads(capsys.readouterr().out)[0]["status"] == "completed_with_issues"
    assert cli.main(env.argv("resume", run_id)) == cli.EXIT_WITH_ISSUES
    assert f"maf: {run_id} is completed_with_issues; nothing to resume" in capsys.readouterr().err
    assert cli.main(env.argv("resume", run_id, "--extra-round")) == cli.EXIT_OK
    index = env.pipeline().status(run_id)
    assert (index.status, index.round, index.unresolved_critical) == (RunStatus.COMPLETED, 2, 0)
    assert env.backends["execution"].calls == 2

def test_resume_extra_round_on_other_statuses_is_a_usage_error(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(env.argv("run", "x"))
    run_id = capsys.readouterr().out.splitlines()[0]
    assert cli.main(env.argv("resume", run_id, "--extra-round")) == cli.EXIT_USAGE
    assert "only applies to completed_with_issues" in capsys.readouterr().err


def test_run_failure_reports_error_and_spend_on_one_line(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    env.backends["execution"].error = RuntimeError("Sandbox is required but failed to initialize:\n  bridge sockets")
    assert cli.main(env.argv("run", "x", "--budget", "4")) == cli.EXIT_FAILED
    last = capsys.readouterr().err.strip().splitlines()[-1]
    assert last == (
        "failed: execution: RuntimeError: Sandbox is required but failed to initialize: bridge sockets (spent $0.0000 of $4.00)"
    )


def test_run_missing_file_exits_2(env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(env.argv("run", "x", "--file", str(tmp_path / "nope.pdf"))) == cli.EXIT_USAGE
    assert "not found" in capsys.readouterr().err


def test_run_no_wait_launches_detached_resume(env: Env, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    launched: list[dict[str, Any]] = []

    def fake_popen(argv: list[str], **kw: Any) -> None:
        launched.append({"argv": argv, **kw})

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    assert cli.main(env.argv("run", "x", "--no-wait")) == 0
    run_id = capsys.readouterr().out.splitlines()[0]
    (call,) = launched
    assert call["argv"][-2:] == ["resume", run_id]
    assert call["argv"][1:3] == ["-m", "maf.cli"]
    assert "--vault" in call["argv"] and str(env.vault.resolve()) in call["argv"]
    assert call["start_new_session"] is True
    assert env.pipeline().status(run_id).status == RunStatus.PENDING


# --------------------------------------------------------------------------- resume, status, list, serve


def test_a_workspace_recorded_under_another_root_is_a_usage_error_and_a_status_warning(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without the --workspaces the run was made with (or after the default moved with the installation), resume and
    export stop with exit 2 before anything runs or is written, and status warns next to the recorded path."""
    env.backends["execution"].error = RuntimeError("crash")
    assert cli.main(env.argv("run", "x")) == cli.EXIT_FAILED
    run_id = capsys.readouterr().out.splitlines()[0]
    run_md = env.vault / "runs" / run_id / "run.md"
    before = run_md.read_text(encoding="utf-8")
    calls = {name: b.calls for name, b in env.backends.items()}
    other = ["--vault", str(env.vault), "--workspaces", str(tmp_path / "other")]
    hint = f"Pass --workspaces {env.workspaces}"

    assert cli.main([*other, "resume", run_id, "--budget", "9", "--note", "go"]) == cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert err.startswith(f"maf: run {run_id} has its workspace at {env.workspaces / run_id} (run.md)")
    assert str(tmp_path / "other" / run_id) in err and hint in err
    assert cli.main([*other, "export", run_id]) == cli.EXIT_USAGE
    assert hint in capsys.readouterr().err
    assert run_md.read_text(encoding="utf-8") == before
    assert {name: b.calls for name, b in env.backends.items()} == calls
    assert not (tmp_path / "other").exists()

    assert cli.main([*other, "status", run_id]) == cli.EXIT_OK
    captured = capsys.readouterr()
    assert captured.err.startswith("maf: warning: run ") and hint in captured.err
    assert f"workspace: {env.workspaces / run_id}" in captured.out
    assert cli.main(env.argv("status", run_id)) == cli.EXIT_OK
    assert capsys.readouterr().err == ""
    assert cli.main(env.argv("resume", run_id)) == cli.EXIT_OK  # with the recorded root it continues


def test_resume_unknown_run_exits_2(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(env.argv("resume", "2026-01-01-nope")) == cli.EXIT_USAGE
    assert "no such run" in capsys.readouterr().err


def test_status_text_and_json(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(env.argv("run", "status probe"))
    run_id = capsys.readouterr().out.splitlines()[0]

    assert cli.main(env.argv("status", run_id)) == 0
    out = capsys.readouterr().out
    assert f"run_id:   {run_id}" in out and "completed" in out

    assert cli.main(env.argv("status", run_id, "--json")) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["run_id"] == run_id and data["status"] == "completed"


def test_status_and_list_show_completed_with_issues(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    env.backends["crosscheck"].unresolved = 2
    cli.main(env.argv("run", "issues probe"))
    run_id = capsys.readouterr().out.splitlines()[0]
    assert cli.main(env.argv("status", run_id)) == 0
    out = capsys.readouterr().out
    assert "status:   completed_with_issues" in out and "unresolved critical: 2" in out
    assert cli.main(env.argv("list")) == 0
    assert "completed_with_issues" in capsys.readouterr().out
    assert cli.main(env.argv("list", "--json")) == 0
    assert json.loads(capsys.readouterr().out)[0]["status"] == "completed_with_issues"


# --------------------------------------------------------------------------- export


def test_export_reexports_and_notes_it(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(env.argv("run", "x"))
    run_id = capsys.readouterr().out.splitlines()[0]
    index = env.pipeline().status(run_id)
    workspace = Path(index.workspace)
    (workspace / "src").mkdir()
    (workspace / "src" / "alloc.c").write_text("int x;\n")
    (workspace / ".env").write_text("")
    (workspace / "leak").symlink_to("/etc/hostname")
    (workspace / "src" / "alloc.o").write_bytes(b"\x7fELF")
    calls = {name: b.calls for name, b in env.backends.items()}

    assert cli.main(env.argv("export", run_id)) == cli.EXIT_OK

    out, err = capsys.readouterr()
    deliverables = env.vault / "runs" / run_id / "deliverables"
    assert out.splitlines() == [
        f"exported 1 file(s), 7 B, to {deliverables}",
        "left out 1 empty sandbox placeholder file(s): .env",
        "excluded: src/alloc.o (export_include brings a file back)",
    ]
    assert err.strip() == "skipped 1 unsafe entry: leak (symlink out of the workspace)"
    assert (deliverables / "src" / "alloc.c").is_file()
    assert {name: b.calls for name, b in env.backends.items()} == calls  # no stage (and no model) ran
    after = env.pipeline().status(run_id)
    note = "maf export: 1 file(s), 7 B; 1 unsafe entry skipped; excluded: src/alloc.o"
    assert (after.status, after.export_note) == (RunStatus.COMPLETED, note)
    assert cli.main(env.argv("status", run_id)) == 0
    assert f"exported: {after.exported_at.isoformat()} ({note})" in capsys.readouterr().out  # type: ignore[union-attr]


def test_export_failures_exit_1_and_unknown_runs_exit_2(env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(env.argv("export", "2026-01-01-nope")) == cli.EXIT_USAGE
    assert "no such run: 2026-01-01-nope" in capsys.readouterr().err
    assert cli.main(env.argv("export", "../escape")) == cli.EXIT_USAGE
    capsys.readouterr()

    env.backends["ingestion"].error = RuntimeError("triage down")  # the run never gets a mode
    cli.main(env.argv("run", "x"))
    no_mode = capsys.readouterr().out.splitlines()[0]
    assert cli.main(env.argv("export", no_mode)) == cli.EXIT_FAILED
    assert "maf: export failed: " in capsys.readouterr().err

    cli.main(env.argv("run", "y"))
    run_id = capsys.readouterr().out.splitlines()[0]
    (Path(env.pipeline().status(run_id).workspace) / "big.bin").write_bytes(b"x" * 2000)
    config = tmp_path / "small.yaml"
    config.write_text("export_max_mb: 0.001\n", encoding="utf-8")
    assert cli.main(["--config", str(config), *env.argv("export", run_id)]) == cli.EXIT_FAILED
    assert "maf: export failed: the workspace export of" in capsys.readouterr().err
    assert env.pipeline().status(run_id).exported_at is None

    with env.pipeline()._guard(run_id):
        assert cli.main(env.argv("export", run_id)) == cli.EXIT_FAILED
    assert "already running" in capsys.readouterr().err


def test_status_unknown_exits_2(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(env.argv("status", "2026-01-01-nope")) == cli.EXIT_USAGE


def test_list_empty_text_and_json_limit(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(env.argv("list")) == 0
    assert "no runs" in capsys.readouterr().out
    for brief in ("one", "two", "three"):
        cli.main(env.argv("run", brief, "--review"))
    capsys.readouterr()
    assert cli.main(env.argv("list", "--json", "--limit", "2")) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 2
    assert set(rows[0]) == {"run_id", "status", "stage", "spent_usd", "created"}
    assert rows[0]["status"] == "awaiting_review"
    assert cli.main(env.argv("list")) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 3


def test_serve_passes_host_and_port(env: Env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import maf.mcp_server

    seen: dict[str, Any] = {}
    monkeypatch.setattr(maf.mcp_server, "serve", lambda pipeline, host, port, transport, uds: seen.update(
        host=host, port=port, transport=transport, uds=uds))
    assert cli.main(env.argv("serve", "--port", "9123")) == 0
    assert seen == {"host": "127.0.0.1", "port": 9123, "transport": "http", "uds": None}
    assert cli.main(env.argv("serve", "--uds", str(tmp_path / "s.sock"))) == 0
    assert seen["uds"] == tmp_path / "s.sock"
    assert cli.main(env.argv("serve", "--stdio")) == 0
    assert seen["transport"] == "stdio"


def test_serve_stdio_rejects_host_and_port(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(env.argv("serve", "--stdio", "--port", "9000")) == cli.EXIT_USAGE
    assert "--stdio" in capsys.readouterr().err
    assert cli.main(env.argv("serve", "--stdio", "--uds", "/tmp/x.sock")) == cli.EXIT_USAGE


def test_serve_exits_1_when_it_cannot_listen(env: Env, monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture[str]) -> None:
    """A taken address is transient (the unit restarts on 1); configuration errors exit 2 (no restart)."""
    import maf.mcp_server

    def busy(*args: Any, **kwargs: Any) -> None:
        raise OSError(98, "Address already in use")

    monkeypatch.setattr(maf.mcp_server, "serve", busy)
    assert cli.main(env.argv("serve")) == cli.EXIT_FAILED
    assert "cannot listen" in capsys.readouterr().err


def test_serve_env_file_loads_keys_into_maf_only(env: Env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    """The stdio recipe: tunnel-client spawns maf without provider keys; maf reads them itself."""
    import os

    import maf.mcp_server

    seen: dict[str, str | None] = {}
    monkeypatch.setattr(maf.mcp_server, "serve", lambda *a, **k: seen.update(
        openai=os.environ.get("OPENAI_API_KEY"), runtime=os.environ.get("CONTROL_PLANE_API_KEY")))
    monkeypatch.setenv("CONTROL_PLANE_API_KEY", "runtime-key-inherited")
    env_file = tmp_path / "maf.env"
    env_file.write_text("OPENAI_API_KEY=from-file-123\nGEMINI_API_KEY=\n# c\n")
    env_file.chmod(0o644)
    assert cli.main(env.argv("serve", "--stdio", "--env-file", str(env_file))) == cli.EXIT_USAGE
    assert "chmod 600" in capsys.readouterr().err and seen == {}
    env_file.chmod(0o600)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cli.main(env.argv("serve", "--stdio", "--env-file", str(env_file))) == 0
    assert seen == {"openai": "from-file-123", "runtime": None}  # the tunnel's runtime key is dropped
    assert "GEMINI_API_KEY" not in os.environ  # empty template lines set nothing
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cli.main(env.argv("serve", "--env-file", str(tmp_path / "missing.env"))) == cli.EXIT_USAGE


def test_chatgpt_commands_never_build_a_pipeline(
    env: Env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from maf import chatgpt

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(cli, "make_pipeline", lambda settings: pytest.fail("chatgpt must not build a pipeline"))
    seen: dict[str, Any] = {}

    def fake_setup(settings: Settings, *, layout: chatgpt.Layout, params: chatgpt.UnitParams, reload: bool) -> int:
        seen.update(action="setup", port=params.mcp_port, health=params.health_port, reload=reload,
                    unit_dir=layout.unit_dir)
        return 0

    def fake_status(settings: Settings, *, layout: chatgpt.Layout, tunnel_client: Path | None, lines: int) -> int:
        seen.update(action="status", lines=lines)
        return 1

    monkeypatch.setattr(chatgpt, "setup", fake_setup)
    monkeypatch.setattr(chatgpt, "status", fake_status)
    assert cli.main(env.argv("chatgpt", "setup", "--no-reload", "--health-port", "18766")) == 0
    assert (seen["action"], seen["port"], seen["health"], seen["reload"]) == ("setup", 8765, 18766, False)
    assert seen["unit_dir"] == tmp_path / "home" / ".config" / "systemd" / "user"
    assert cli.main(env.argv("chatgpt", "status", "--lines", "3")) == 1
    assert (seen["action"], seen["lines"]) == ("status", 3)
    assert not env.vault.exists() and not env.workspaces.exists()


@pytest.mark.parametrize("argv", [["chatgpt"], ["chatgpt", "bogus"], ["chatgpt", "status", "--lines", "0"]])
def test_chatgpt_usage_errors_exit_2(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(argv) == cli.EXIT_USAGE
    assert capsys.readouterr().err


def test_chatgpt_setup_refuses_public_host(env: Env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = tmp_path / "config.yaml"
    config.write_text("mcp_host: 0.0.0.0\n", encoding="utf-8")
    assert cli.main(env.argv("--config", str(config), "chatgpt", "setup", "--no-reload")) == cli.EXIT_USAGE
    assert "loopback" in capsys.readouterr().err
    assert not (tmp_path / "home" / ".config").exists()


def test_chatgpt_setup_writes_the_given_config_and_vault_into_the_unit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--config``/``--vault`` (and MAF_* variables) must reach the service, or it silently reads other settings."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("MAF_BUDGET_USD", "12")
    config = tmp_path / "c.yaml"
    config.write_text(f"mcp_max_budget_usd: 3\nmcp_inbox: {tmp_path / 'inbox'}\n", encoding="utf-8")
    assert cli.main(["--config", str(config), "--vault", str(tmp_path / "v"), "chatgpt", "setup", "--no-reload",
                     "--health-port", "18777"]) == 0
    unit = (home / ".config" / "systemd" / "user" / "maf-mcp.service").read_text()
    exec_start = next(line for line in unit.splitlines() if line.startswith("ExecStart="))
    assert f" serve --config {config} --vault {tmp_path / 'v'} --host 127.0.0.1 --port 8765 --uds " in exec_start
    assert "Environment=MAF_BUDGET_USD=12\n" in unit
    # A re-run that leaves out a source the unit has is refused, not silently re-rendered with other settings.
    monkeypatch.setenv("MAF_CONFIG", str(config))
    assert cli.main(["chatgpt", "setup", "--no-reload"]) == cli.EXIT_USAGE
    assert f"runs with --vault {tmp_path / 'v'}, which this setup was not given" in capsys.readouterr().err
    # With the same sources it goes through, and keeps the installed health port (and binary) instead of resetting them.
    monkeypatch.setenv("MAF_VAULT", str(tmp_path / "v"))
    assert cli.main(["chatgpt", "setup", "--no-reload"]) == 0
    tunnel = (home / ".config" / "systemd" / "user" / "maf-tunnel.service").read_text()
    assert "--health.listen-addr 127.0.0.1:18777" in tunnel
    assert f" serve --config {config} --vault" in (home / ".config" / "systemd" / "user" / "maf-mcp.service").read_text()


def test_chatgpt_setup_refuses_the_mcp_port_as_health_port(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert cli.main(["--vault", str(tmp_path / "v"), "--workspaces", str(tmp_path / "w"), "chatgpt", "setup",
                     "--no-reload", "--health-port", "8765"]) == cli.EXIT_USAGE
    assert "health port" in capsys.readouterr().err
    assert cli.main(["chatgpt", "setup", "--no-reload", "--tunnel-client", "tc"]) == cli.EXIT_USAGE
    assert "absolute" in capsys.readouterr().err
    assert not (tmp_path / "home" / ".config").exists()


def test_serve_refuses_public_host(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(env.argv("serve", "--host", "0.0.0.0")) == cli.EXIT_USAGE
    assert "loopback" in capsys.readouterr().err
