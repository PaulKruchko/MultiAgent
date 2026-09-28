"""``maf`` command line.

Owner: orchestration.

    maf run "<brief>" [--file PATH ...] [--budget USD] [--tier default|max] [--review] [--no-wait]
    maf resume RUN_ID [--note TEXT] [--budget USD]
    maf status RUN_ID [--json]
    maf list [--json] [--limit N]
    maf serve [--host 127.0.0.1] [--port 8765]

Global options: ``--vault PATH``, ``--workspaces PATH``, ``--config PATH``.
Exit codes: 0 completed or awaiting review; 1 failed; 2 usage error; 3 budget exceeded.
``run`` prints the run_id first, then progress lines per stage, then the path of 05-final.md.
``run --no-wait`` creates the run, starts ``maf resume RUN_ID`` as a detached process (output in
``workspace/.maf/run.log``) and returns immediately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from maf.config import Settings, load_settings
from maf.handoff import HandoffKind
from maf.pipeline import Pipeline
from maf.types import RunStatus
from maf.vault import RunIndex, note_name

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_BUDGET = 3

_GLOBAL_OPTIONS = ("vault", "workspaces", "config")


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be positive: {text!r}")
    return value


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive: {text!r}")
    return value


def _port(text: str) -> int:
    value = _positive_int(text)
    if value > 65535:
        raise argparse.ArgumentTypeError(f"not a TCP port: {text!r}")
    return value


def _global_options(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Global options; on subcommands they default to SUPPRESS so either position works."""
    default = argparse.SUPPRESS if suppress else None
    parser.add_argument("--vault", type=Path, default=default, help="Obsidian vault root")
    parser.add_argument("--workspaces", type=Path, default=default, help="workspaces root (outside the vault)")
    parser.add_argument("--config", type=Path, default=default, help="config YAML (default ~/.config/maf/config.yaml)")


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree described in the module docstring (subcommand dest: ``command``)."""
    parser = argparse.ArgumentParser(
        prog="maf",
        description="Multi-agent pipeline coordinating ChatGPT, Gemini and Claude over an Obsidian vault.",
    )
    _global_options(parser, suppress=False)
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    common = argparse.ArgumentParser(add_help=False)
    _global_options(common, suppress=True)

    run = sub.add_parser("run", parents=[common], help="create a run and execute it")
    run.add_argument("brief", help="the request, in plain language")
    run.add_argument("--file", dest="files", type=Path, action="append", default=[], metavar="PATH",
                     help="input file for ingestion (repeatable)")
    run.add_argument("--budget", type=_positive_float, metavar="USD", help="per-run spend cap")
    run.add_argument("--tier", choices=("default", "max"), help="model tier")
    run.add_argument("--review", action="store_true", default=None, help="pause after strategy for review")
    run.add_argument("--no-wait", action="store_true", help="start in the background and return immediately")

    resume = sub.add_parser("resume", parents=[common], help="continue a paused, failed or crashed run")
    resume.add_argument("run_id")
    resume.add_argument("--note", help="direction for later stages (stored as the review note)")
    resume.add_argument("--budget", type=_positive_float, metavar="USD", help="new spend cap")

    status = sub.add_parser("status", parents=[common], help="show one run")
    status.add_argument("run_id")
    status.add_argument("--json", action="store_true", help="print run.md frontmatter as JSON")

    list_ = sub.add_parser("list", parents=[common], help="list runs, newest first")
    list_.add_argument("--json", action="store_true")
    list_.add_argument("--limit", type=_positive_int, default=20, metavar="N")

    serve = sub.add_parser("serve", parents=[common], help="serve the MCP endpoint for the ChatGPT app")
    serve.add_argument("--host", help="bind address (loopback only; default from settings)")
    serve.add_argument("--port", type=_port, help="TCP port (default from settings)")
    return parser


def exit_code_for(status: RunStatus) -> int:
    if status == RunStatus.FAILED:
        return EXIT_FAILED
    if status == RunStatus.BUDGET_EXCEEDED:
        return EXIT_BUDGET
    # COMPLETED and AWAITING_REVIEW are successes; PENDING/RUNNING mean "started" (--no-wait).
    return EXIT_OK


def make_pipeline(settings: Settings) -> Pipeline:
    """Seam for tests: the pipeline the CLI drives."""
    return Pipeline(settings)


def _settings_from(args: argparse.Namespace) -> Settings:
    return load_settings(
        getattr(args, "config", None),
        vault_path=getattr(args, "vault", None),
        workspaces_path=getattr(args, "workspaces", None),
    )


def _print_progress(index: RunIndex, message: str) -> None:
    print(f"[{index.run_id}] {message}", flush=True)


def _report(pipeline: Pipeline, index: RunIndex, out: TextIO, err: TextIO) -> int:
    paths = pipeline.vault.paths(index.run_id)
    if index.status == RunStatus.COMPLETED:
        print(paths.note(note_name(HandoffKind.FINAL)), file=out)
    elif index.status == RunStatus.AWAITING_REVIEW:
        print(f"awaiting review: edit {paths.note(note_name(HandoffKind.STRATEGY))} then run: maf resume {index.run_id}", file=out)
    elif index.status == RunStatus.BUDGET_EXCEEDED:
        print(f"budget exceeded: {index.error}", file=err)
        print(f"raise the cap with: maf resume {index.run_id} --budget USD", file=err)
    elif index.status == RunStatus.FAILED:
        print(f"failed: {index.error}", file=err)
    return exit_code_for(index.status)


def _launch_detached(args: argparse.Namespace, pipeline: Pipeline, run_id: str) -> Path:
    """Start ``maf resume run_id`` in a new session, logging to ``workspace/.maf/run.log``."""
    argv = [sys.executable, "-m", "maf.cli"]
    for option in _GLOBAL_OPTIONS:
        value = getattr(args, option, None)
        if value is not None:
            argv += [f"--{option}", str(Path(value).expanduser().resolve())]
    argv += ["resume", run_id]
    log_path = pipeline.vault.paths(run_id).workspace / ".maf" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_file:
        subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    return log_path


def _cmd_run(args: argparse.Namespace, pipeline: Pipeline) -> int:
    try:
        index = pipeline.create(args.brief, args.files, budget_usd=args.budget, tier=args.tier, review=args.review)
    except FileNotFoundError as exc:
        print(f"maf: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except ValueError as exc:
        print(f"maf: {exc}", file=sys.stderr)
        return EXIT_USAGE
    print(index.run_id, flush=True)
    if args.no_wait:
        log_path = _launch_detached(args, pipeline, index.run_id)
        print(f"started in the background; log: {log_path}", flush=True)
        return EXIT_OK
    index = pipeline.run(index.run_id, progress=_print_progress)
    return _report(pipeline, index, sys.stdout, sys.stderr)


def _cmd_resume(args: argparse.Namespace, pipeline: Pipeline) -> int:
    try:
        index = pipeline.resume(args.run_id, note=args.note, budget_usd=args.budget, progress=_print_progress)
    except FileNotFoundError:
        print(f"maf: no such run: {args.run_id}", file=sys.stderr)
        return EXIT_USAGE
    except RuntimeError as exc:
        print(f"maf: {exc}", file=sys.stderr)
        return EXIT_FAILED
    return _report(pipeline, index, sys.stdout, sys.stderr)


def _status_lines(index: RunIndex) -> list[str]:
    lines = [
        f"run_id:   {index.run_id}",
        f"status:   {index.status.value}",
        f"stage:    {index.stage} (round {index.round})",
        f"mode:     {index.mode or '-'}",
        f"tier:     {index.tier}",
        f"spent:    ${index.spent_usd:.4f} of ${index.budget_usd:.2f}",
    ]
    if index.spend_by_agent:
        lines.append("by agent: " + ", ".join(f"{a} ${usd:.4f}" for a, usd in sorted(index.spend_by_agent.items())))
    if index.handoffs:
        lines.append("handoffs: " + ", ".join(index.handoffs))
    if index.error:
        lines.append(f"error:    {index.error}")
    lines.append(f"workspace: {index.workspace}")
    return lines


def _cmd_status(args: argparse.Namespace, pipeline: Pipeline) -> int:
    try:
        index = pipeline.status(args.run_id)
    except FileNotFoundError:
        print(f"maf: no such run: {args.run_id}", file=sys.stderr)
        return EXIT_USAGE
    if args.json:
        print(json.dumps(index.model_dump(mode="json"), indent=2))
    else:
        print("\n".join(_status_lines(index)))
    return EXIT_OK


def _cmd_list(args: argparse.Namespace, pipeline: Pipeline) -> int:
    runs = pipeline.list_runs()[: args.limit]
    if args.json:
        rows = [
            {
                "run_id": r.run_id,
                "status": r.status.value,
                "stage": r.stage,
                "spent_usd": r.spent_usd,
                "created": r.created.isoformat(),
            }
            for r in runs
        ]
        print(json.dumps(rows, indent=2))
        return EXIT_OK
    if not runs:
        print("no runs")
        return EXIT_OK
    width = max(len(r.run_id) for r in runs)
    for r in runs:
        print(f"{r.run_id:<{width}}  {r.status.value:<15}  {r.stage:<10}  ${r.spent_usd:>8.4f}  {r.created:%Y-%m-%d %H:%M}")
    return EXIT_OK


def _cmd_serve(args: argparse.Namespace, pipeline: Pipeline) -> int:
    from maf.mcp_server import serve

    host = args.host or pipeline.settings.mcp_host
    port = args.port or pipeline.settings.mcp_port
    try:
        serve(pipeline, host=host, port=port)
    except ValueError as exc:
        print(f"maf: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        pass
    return EXIT_OK


_COMMANDS = {
    "run": _cmd_run,
    "resume": _cmd_resume,
    "status": _cmd_status,
    "list": _cmd_list,
    "serve": _cmd_serve,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point (``maf = maf.cli:main``). Returns the exit code; ``__main__`` calls ``sys.exit``."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 on usage errors and 0 on --help
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE
    try:
        settings = _settings_from(args)
    except ValueError as exc:
        print(f"maf: bad configuration: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        pipeline = make_pipeline(settings)
    except Exception as exc:  # noqa: BLE001 - report setup problems without a traceback
        print(f"maf: cannot start: {exc}", file=sys.stderr)
        return EXIT_FAILED
    return _COMMANDS[args.command](args, pipeline)


if __name__ == "__main__":
    raise SystemExit(main())
