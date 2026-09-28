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
    """Minimal stage backend: no notes, optionally raising once."""

    name: StageName
    error: BaseException | None = None

    def run_stage(self, ctx: StageContext) -> StageOutput:
        if self.error is not None:
            error, self.error = self.error, None
            raise error
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
    assert (r.command, r.run_id, r.note, r.budget) == ("resume", "r1", "go", 40.0)
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
        (RunStatus.FAILED, 1),
        (RunStatus.BUDGET_EXCEEDED, 3),
    ],
)
def test_exit_code_for(status: RunStatus, code: int) -> None:
    assert cli.exit_code_for(status) == code


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


def test_run_budget_exceeded_exits_3_then_resume_exits_0(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    env.backends["strategy"].error = BudgetExceeded(cap_usd=1.0, spent_usd=0.9, requested_usd=0.5, what="strategy")
    assert cli.main(env.argv("run", "x", "--budget", "1")) == cli.EXIT_BUDGET
    captured = capsys.readouterr()
    run_id = captured.out.splitlines()[0]
    assert "--budget" in captured.err
    assert cli.main(env.argv("resume", run_id, "--budget", "3", "--note", "carry on")) == 0
    index = env.pipeline().status(run_id)
    assert (index.status, index.budget_usd) == (RunStatus.COMPLETED, 3.0)
    assert (Path(index.workspace) / ".maf" / "review-note.md").read_text(encoding="utf-8").strip() == "carry on"


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
