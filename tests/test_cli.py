"""CLI tests: argparse tree, exit codes, and each command against a real temp vault with scripted backends."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    """Minimal stage backend: no notes, optionally raising once. ``unresolved`` > 0 makes a crosscheck loop back."""

    name: StageName
    error: BaseException | None = None
    unresolved: int = 0
    calls: int = 0

    def run_stage(self, ctx: StageContext) -> StageOutput:
        self.calls += 1
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        if self.name == "crosscheck":
            return StageOutput(
                notes=[], index_updates={"unresolved_critical": self.unresolved}, loop_back=self.unresolved > 0
            )
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


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["bogus"],
        ["run"],
        ["run", "b", "--budget", "0"],
        ["run", "b", "--budget", "abc"],
        ["run", "b", "--tier", "ultra"],
        ["list", "--limit", "0"],
        ["serve", "--port", "70000"],
    ],
)
def test_usage_errors_exit_2(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(argv) == cli.EXIT_USAGE
    assert capsys.readouterr().err


def test_help_exits_0(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--help"]) == 0
    assert "resume" in capsys.readouterr().out


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


def test_serve_passes_host_and_port(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    import maf.mcp_server

    seen: dict[str, Any] = {}
    monkeypatch.setattr(maf.mcp_server, "serve", lambda pipeline, host, port: seen.update(host=host, port=port))
    assert cli.main(env.argv("serve", "--port", "9123")) == 0
    assert seen == {"host": "127.0.0.1", "port": 9123}


def test_serve_refuses_public_host(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(env.argv("serve", "--host", "0.0.0.0")) == cli.EXIT_USAGE
    assert "loopback" in capsys.readouterr().err
