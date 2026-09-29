"""Deterministic stage state machine: create, run, resume, review gate, crosscheck loop.

Owner: orchestration (pipeline + cli + mcp_server).

State lives only in run.md (``RunIndex``) and ledger.jsonl, so a crash at any point resumes by
re-running the recorded ``stage`` (stages are idempotent). Transitions (``next_step``)::

    ingestion -> strategy -> [review gate if index.review] -> execution -> crosscheck
    crosscheck --loop_back and round <= max_crosscheck_loops--> execution (round += 1)
    crosscheck --pass, or loops exhausted--> final -> COMPLETED (unresolved_critical == 0 and criteria_unmet == 0)
                                                   -> COMPLETED_WITH_ISSUES (unresolved_critical > 0: loops exhausted;
                                                      or criteria_unmet > 0: acceptance criteria or clean-room not met)

Review gate: after strategy with ``review=True``, the status becomes AWAITING_REVIEW with ``stage="execution"``.
``resume(run_id, note=...)`` re-reads 02-strategy.md (the user may have edited it in Obsidian),
re-validates it (invalid means FAILED with the errors, and the user can fix and resume again),
and continues.

COMPLETED and COMPLETED_WITH_ISSUES are both terminal: ``run`` and ``resume`` return them unchanged.
``export(run_id)`` (``maf export``) rewrites ``deliverables/`` from the workspace at any status, under the run lock,
with no model calls, and records it in run.md (``exported_at``, ``export_note``).
``resume(run_id, extra_round=True)`` is the one way to continue a COMPLETED_WITH_ISSUES run: it schedules one
more execution + crosscheck pass (round + 1, reading the last cross-check's unresolved issues) and then final
again. The loop cap still applies, so a pass that leaves critical issues open goes straight to final.

Error mapping for a stage attempt:
- ``BudgetExceeded``: BUDGET_EXCEEDED (resumable after raising the budget: ``maf resume --budget``)
- ``HandoffInvalid`` (after its one repair): FAILED
- ``ProviderError``: FAILED at once. Stages never swallow a non-retryable one (``StageContext.call`` retries
  only ``retryable`` token-provider errors, never Claude Code), so an infrastructure failure such as a Claude
  Code sandbox that cannot start ends the run before any crosscheck loop can re-pay against it.
- any other exception: FAILED (``error`` records type and message)
The ledger totals are mirrored into run.md after every stage and on every failure.
"""

from __future__ import annotations

import fcntl
import logging
import math
import os
import stat
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from maf.config import Settings
from maf.handoff import HandoffInvalid, HandoffKind, validate_handoff
from maf.ledger import BudgetExceeded, Ledger
from maf.providers import ProviderError, Providers
from maf.stages.base import StageBackend, StageContext, StageOutput
from maf.types import STAGE_ORDER, RunStatus, StageName, Tier
from maf.vault import RunIndex, RunPaths, Vault, atomic_write_text, describe_issues, note_name

if TYPE_CHECKING:
    from maf.stages.final import Export

log = logging.getLogger(__name__)

ProvidersFactory = Callable[[Settings, Path], Providers]
"""``(settings, workspace) -> Providers``. Production uses ``maf.providers.build_providers``; tests inject fakes."""

Clock = Callable[[], datetime]

ProgressFn = Callable[[RunIndex, str], None]
"""Optional observer: called with the current index and a one-line human-readable message."""

ALLOWED_INDEX_UPDATES: frozenset[str] = frozenset(
    {"mode", "unresolved_critical", "criteria_unmet", "unmet_criteria", "exported_at", "export_note"}
)
"""``StageOutput.index_updates`` keys the pipeline applies: ``mode`` (ingestion), ``unresolved_critical``
(crosscheck), and the final stage's acceptance and export fields."""

REVIEW_NOTE_REL = Path(".maf") / "review-note.md"
"""Workspace-relative location of the persisted ``resume --note`` text."""

ORPHAN_GRACE_S = 120.0
"""A PENDING run with no recorded owner (a CLI run between ``create`` and taking its lock, or one queued by an older
maf) whose lock is free is only treated as orphaned after run.md is this old."""

BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")

RUN_LOCK_NAME = ".lock"
"""Per-run lock file in the run folder (a dotfile, so Obsidian ignores it)."""

_CREATE_ATTEMPTS = 5

# Process-wide guard: (vault root, run_id) pairs currently being advanced by any Pipeline instance.
_ACTIVE: set[tuple[str, str]] = set()
_ACTIVE_LOCK = threading.Lock()


@dataclass(frozen=True)
class Step:
    """The next transition: run ``stage`` (with ``round``), or stop with ``status``."""

    stage: StageName | None
    round: int
    status: RunStatus


def next_step(index: RunIndex, finished: StageName, output: StageOutput, max_loops: int) -> Step:
    """Pure transition after ``finished`` completed with ``output``.

    - After strategy with ``index.review``: ``Step("execution", round, AWAITING_REVIEW)``.
    - After crosscheck with ``loop_back`` and ``index.round <= max_loops``: ``Step("execution", round+1, RUNNING)``.
    - After crosscheck otherwise: ``Step("final", round, RUNNING)``.
    - After final: ``Step(None, round, COMPLETED)``, or ``COMPLETED_WITH_ISSUES`` when ``index.unresolved_critical > 0``
      (final was reached only because the loop cap was hit) or ``index.criteria_unmet > 0`` (acceptance criteria,
      including the clean-room reproduction, not met).
    - Otherwise the next stage in ``STAGE_ORDER``, RUNNING.
    """
    if finished == "final":
        status = RunStatus.COMPLETED_WITH_ISSUES if index.has_issues else RunStatus.COMPLETED
        return Step(None, index.round, status)
    if finished == "strategy" and index.review:
        return Step("execution", index.round, RunStatus.AWAITING_REVIEW)
    if finished == "crosscheck":
        if output.loop_back and index.round <= max_loops:
            return Step("execution", index.round + 1, RunStatus.RUNNING)
        return Step("final", index.round, RunStatus.RUNNING)
    following = STAGE_ORDER[STAGE_ORDER.index(finished) + 1]
    return Step(following, index.round, RunStatus.RUNNING)


_FORBIDDEN_INPUT_ROOTS = (Path("/proc"), Path("/sys"), Path("/dev"))


def check_confined_input(path: Path, root: Path) -> None:
    """``ValueError`` unless ``path`` (already resolved, so symlinks are followed) is a regular, non-empty file
    inside ``root``, outside /proc, /sys and /dev, with no dot-directory or dotfile component below ``root``.

    Pseudo-files that report size 0 but still yield bytes (procfs) are refused, so they cannot be exfiltrated
    or slip past the size-based budget estimate."""
    base = root.expanduser().resolve()
    if any(path == forbidden or path.is_relative_to(forbidden) for forbidden in _FORBIDDEN_INPUT_ROOTS):
        raise ValueError(f"input file {path} is in a system pseudo-filesystem")
    if not path.is_relative_to(base):
        raise ValueError(f"input file {path} is outside the allowed inbox {base}")
    if any(part.startswith(".") for part in path.relative_to(base).parts):
        raise ValueError(f"input file {path} is hidden or inside a hidden directory")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"input file {path} is not a regular file")
    if info.st_size == 0:
        with path.open("rb") as fh:
            if fh.read(1):
                raise ValueError(f"input file {path} reports size 0 but is not empty (pseudo-file)")
        raise ValueError(f"input file {path} is empty")


def not_an_inbox_file(given: str | Path) -> ValueError:
    """The one refusal a remote caller gets for any input file problem: it echoes only the caller's own string, so
    it tells nothing about files outside the inbox (whether they exist, where a symlink points)."""
    return ValueError(f"not an allowed inbox file: {given}")


def boot_id() -> str:
    """This boot's id (``/proc/sys/kernel/random/boot_id``), or "" where it cannot be read."""
    try:
        return BOOT_ID_PATH.read_text(encoding="ascii").strip()
    except OSError:
        return ""


def process_owner() -> str:
    """``<boot id>:<pid>`` of this process, recorded as ``RunIndex.owner`` of the runs a server queues."""
    return f"{boot_id()}:{os.getpid()}"


def owner_alive(owner: str) -> bool:
    """True if ``owner`` (``process_owner()`` of some process) names another process that is still running. This
    process counts as dead: when it recovers orphans it has queued nothing yet, so a match is an earlier incarnation
    whose pid was reused. A pid reused by an unrelated process reads as alive (the run then waits for that process)."""
    boot, _, pid_text = owner.rpartition(":")
    try:
        pid = int(pid_text)
    except ValueError:
        return False
    if pid <= 0 or pid == os.getpid() or boot != boot_id():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def _describe(exc: BaseException) -> str:
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


class Pipeline:
    """Runs one run at a time per call. Different runs may execute concurrently in different threads."""

    def __init__(
        self,
        settings: Settings,
        *,
        vault: Vault | None = None,
        providers_factory: ProvidersFactory | None = None,
        backends: dict[StageName, StageBackend] | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Defaults: ``Vault(settings.vault_path, settings.workspaces_path)``,
        ``maf.providers.build_providers``, ``maf.stages.default_backends()``, and local-time ``datetime.now``."""
        self.settings = settings
        self.vault = vault or Vault(settings.vault_path, settings.workspaces_path)
        if providers_factory is None:
            from maf.providers import build_providers

            providers_factory = build_providers
        self.providers_factory = providers_factory
        if backends is None:
            from maf.stages import default_backends

            backends = default_backends()
        missing = [stage for stage in STAGE_ORDER if stage not in backends]
        if missing:
            raise ValueError(f"no backend for stage(s): {', '.join(missing)}")
        self.backends = backends
        self.clock: Clock = clock or datetime.now
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        """Ask every run this instance is advancing to stop at the next stage boundary (server shutdown).
        A stopped run is marked FAILED with an ``interrupted`` error; ``resume`` continues it."""
        self._stop_event.set()

    # ------------------------------------------------------------------ creation and queries

    def create(
        self,
        brief: str,
        files: list[Path] | None = None,
        *,
        budget_usd: float | None = None,
        tier: Tier | None = None,
        review: bool | None = None,
        input_root: Path | None = None,
        origin: str | None = None,
        owner: str | None = None,
    ) -> RunIndex:
        """Create the run folder, workspace, and run.md (status PENDING, stage ingestion). No model calls.
        Missing input files raise ``FileNotFoundError`` before anything is created.

        ``input_root`` (set for remote callers such as MCP) confines input files to that directory; see
        ``check_confined_input``. Then every problem with a file, a missing one included, raises the same
        ``ValueError`` (``not_an_inbox_file``), which names only the string the caller gave: the confinement check
        runs before any existence check, so a refusal says nothing about files outside the inbox. The details go to
        the debug log. ``origin`` and ``owner`` are recorded in run.md (``RunIndex``)."""
        brief = brief.strip()
        if not brief:
            raise ValueError("brief must not be empty")
        budget = self.settings.budget_usd if budget_usd is None else budget_usd
        if not (math.isfinite(budget) and budget > 0):  # NaN would slip past every cap comparison
            raise ValueError(f"budget must be positive and finite, got {budget}")
        sources: list[Path] = []
        for file in files or []:
            if input_root is not None:
                sources.append(self._confined_source(file, input_root))
                continue
            path = Path(file).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"input file not found: {file}")
            sources.append(path)

        for _ in range(_CREATE_ATTEMPTS):
            now = self.clock()
            run_id = self.vault.new_run_id(brief, now.date())
            paths = self.vault.paths(run_id)
            index = RunIndex(
                run_id=run_id,
                tier=tier or self.settings.tier,
                review=self.settings.review if review is None else review,
                budget_usd=budget,
                created=now,
                updated=now,
                workspace=str(paths.workspace),
                brief=brief,
                origin=origin,
                owner=owner,
            )
            try:
                self.vault.create_run(index, sources)
            except FileExistsError:
                continue  # another creator took this id between new_run_id and create_run
            return self.vault.read_index(run_id)
        raise RuntimeError(f"could not allocate a unique run id for brief {brief[:60]!r}")

    @staticmethod
    def _confined_source(file: str | Path, input_root: Path) -> Path:
        """``file`` resolved, if ``check_confined_input`` accepts it; else ``not_an_inbox_file(file)``."""
        try:
            path = Path(file).expanduser().resolve()
            check_confined_input(path, input_root)
        except (OSError, ValueError, RuntimeError) as exc:  # RuntimeError: a symlink loop
            log.debug("refused input file %r: %s", str(file), exc)
            raise not_an_inbox_file(file) from None
        return path

    def status(self, run_id: str) -> RunIndex:
        return self.vault.read_index(run_id)

    def list_runs(self) -> list[RunIndex]:
        return self.vault.list_runs()

    def ledger_for(self, index: RunIndex) -> Ledger:
        """``Ledger.load(run_id, index.budget_usd, paths.ledger)``."""
        return Ledger.load(index.run_id, index.budget_usd, self.vault.paths(index.run_id).ledger)

    def run_settings(self, index: RunIndex) -> Settings:
        """Settings for one run: the process-wide settings with the run's own ``tier`` (from run.md)."""
        if index.tier == self.settings.tier:
            return self.settings
        return self.settings.model_copy(update={"tier": index.tier})

    def review_note_path(self, run_id: str) -> Path:
        return self.vault.paths(run_id).workspace / REVIEW_NOTE_REL

    # ------------------------------------------------------------------ run and resume

    def run(self, run_id: str, *, progress: ProgressFn | None = None) -> RunIndex:
        """Advance ``run_id`` from its recorded stage until COMPLETED, COMPLETED_WITH_ISSUES, AWAITING_REVIEW,
        FAILED or BUDGET_EXCEEDED. Returns the final index. Never raises for stage errors (they are recorded);
        raises only for a missing run, or ``RuntimeError`` if another thread or process is advancing the run.

        A run in any of those statuses is returned unchanged: continuing one is an explicit decision, made
        with ``resume``. A run left RUNNING by a crashed process is continued from its recorded stage."""
        with self._guard(run_id):
            index = self.vault.read_index(run_id)
            if index.status not in (RunStatus.PENDING, RunStatus.RUNNING):
                return index
            return self._advance(index, progress)

    def resume(
        self,
        run_id: str,
        *,
        note: str | None = None,
        budget_usd: float | None = None,
        extra_round: bool = False,
        progress: ProgressFn | None = None,
    ) -> RunIndex:
        """Continue a non-completed run. ``note`` is stored as the review note passed to later stages
        (persisted in ``workspace/.maf/review-note.md``). ``budget_usd`` raises the cap.

        COMPLETED and COMPLETED_WITH_ISSUES runs are returned unchanged (nothing is written), except that
        ``extra_round=True`` on a COMPLETED_WITH_ISSUES run schedules one more execution + crosscheck pass at
        ``round + 1`` followed by final (see the module docstring). ``extra_round`` on any other status raises
        ``ValueError``."""
        if budget_usd is not None and not (math.isfinite(budget_usd) and budget_usd > 0):
            raise ValueError(f"budget must be positive and finite, got {budget_usd}")
        with self._guard(run_id):
            index = self.vault.read_index(run_id)
            if extra_round and index.status != RunStatus.COMPLETED_WITH_ISSUES:
                raise ValueError(
                    f"extra_round only applies to {RunStatus.COMPLETED_WITH_ISSUES.value} runs; "
                    f"{run_id} is {index.status.value}"
                )
            if index.status.finished and not extra_round:
                return index
            if budget_usd is not None:
                index.budget_usd = budget_usd
            if note is not None:
                atomic_write_text(self.review_note_path(run_id), note.strip() + "\n")
            if extra_round:
                self._schedule_extra_round(index)

            if self._past_review_gate(index):
                errors = self._strategy_errors(run_id)
                if errors:
                    index.status = RunStatus.FAILED
                    index.error = "02-strategy invalid after review: " + "; ".join(errors)
                    index.updated = self.clock()
                    self.vault.write_index(index)
                    return index

            index.status = RunStatus.PENDING
            index.error = None
            index.updated = self.clock()
            self.vault.write_index(index)
            return self._advance(index, progress)

    def export(self, run_id: str) -> tuple[RunIndex, Export]:
        """Rewrite ``deliverables/`` from the run's workspace with the final stage's export
        (``maf.stages.final.export_run``, excludes ``export_excludes(settings)``, cap ``settings.export_max_bytes``) and
        record it in run.md (``exported_at``, ``export_note`` ``maf export: ...``). The status is unchanged and no model
        is called. Raises ``FileNotFoundError`` for an unknown run, ``RuntimeError`` if the run is being advanced,
        ``maf.vault.ExportError`` (``ExportTooLarge`` included) when the export is refused, leaving run.md as it was."""
        from maf.stages.final import export_excludes, export_run

        with self._guard(run_id):
            index = self.vault.read_index(run_id)
            export = export_run(
                self.vault, index, excludes=export_excludes(self.settings), max_bytes=self.settings.export_max_bytes
            )
            index.exported_at = index.updated = self.clock()
            index.export_note = f"maf export: {export.describe()}"
            self.vault.write_index(index)
            return index, export

    def fail_orphans(self, *, grace_s: float = ORPHAN_GRACE_S) -> list[str]:
        """Mark FAILED every run left PENDING or RUNNING by a process that is gone. Returns the affected run ids.
        Never spends money: the user decides with ``maf resume`` whether to continue them.

        A run whose lock another process holds is being advanced and is left alone. With the lock free:
        - RUNNING: orphaned at once. RUNNING is only ever written under the run lock, so a free lock means the
          process advancing it died.
        - PENDING with an ``owner`` (queued by a ``maf serve``): orphaned at once if that process is gone (another
          boot, a dead pid, or this process's own pid); left alone while it runs, since it is that server's queue.
        - PENDING without an owner (a CLI run about to take its lock, or queued by an older maf): orphaned only once
          run.md has not changed for ``grace_s`` seconds."""
        failed: list[str] = []
        for index in self.list_runs():
            if not self._maybe_orphaned(index, grace_s):
                continue
            try:
                with self._guard(index.run_id):
                    current = self.vault.read_index(index.run_id)
                    if not self._maybe_orphaned(current, grace_s):
                        continue
                    previous = current.status.value
                    current.status = RunStatus.FAILED
                    current.error = f"interrupted: left {previous} by a process that exited; maf resume continues it"
                    current.updated = self.clock()
                    self.vault.write_index(current)
                    failed.append(current.run_id)
            except RuntimeError:
                continue  # another process is advancing it right now
        return failed

    def _maybe_orphaned(self, index: RunIndex, grace_s: float) -> bool:
        """``fail_orphans``'s test, short of the lock (the caller takes it)."""
        if index.status == RunStatus.RUNNING:
            return True
        if index.status != RunStatus.PENDING:
            return False
        if index.owner:
            return not owner_alive(index.owner)
        return (self.clock() - index.updated).total_seconds() >= grace_s

    # ------------------------------------------------------------------ internals

    @contextmanager
    def _guard(self, run_id: str) -> Iterator[None]:
        """Exclusive right to advance ``run_id``: an in-process set plus an OS lock on ``runs/<id>/.lock``,
        so a second process (``maf resume`` next to ``maf serve``, a detached run) cannot drive it too."""
        key = (str(self.vault.root.resolve()), run_id)
        with _ACTIVE_LOCK:
            if key in _ACTIVE:
                raise RuntimeError(f"run {run_id} is already running in this process")
            _ACTIVE.add(key)
        try:
            with self._run_lock(run_id):
                yield
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE.discard(key)

    @contextmanager
    def _run_lock(self, run_id: str) -> Iterator[None]:
        root = self.vault.paths(run_id).root
        if not root.is_dir():
            yield  # unknown run: read_index raises FileNotFoundError inside the guard
            return
        fd = os.open(root / RUN_LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"run {run_id} is already running in another process") from None
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def is_locked(self, run_id: str) -> bool:
        """True while some process (this one included) holds ``run_id``'s run lock."""
        key = (str(self.vault.root.resolve()), run_id)
        with _ACTIVE_LOCK:
            if key in _ACTIVE:
                return True
        try:
            with self._run_lock(run_id):
                return False
        except RuntimeError:
            return True

    @staticmethod
    def _schedule_extra_round(index: RunIndex) -> None:
        """Point a COMPLETED_WITH_ISSUES index at execution ``round + 1``. 05-final leaves ``handoffs`` so the
        list stays chronological; the note itself stays on disk until the new final pass overwrites it."""
        final = note_name(HandoffKind.FINAL)
        index.stage = "execution"
        index.round += 1
        index.handoffs = [name for name in index.handoffs if name != final]

    @staticmethod
    def _past_review_gate(index: RunIndex) -> bool:
        """True when the next stage is the first execution pass of a reviewed run."""
        return index.review and index.stage == "execution" and index.round == 1 and "strategy" in index.completed_stages

    def _strategy_errors(self, run_id: str) -> list[str]:
        name = note_name(HandoffKind.STRATEGY)
        try:
            handoff = self.vault.read_handoff(run_id, name)
        except FileNotFoundError:
            return [f"{name}.md is missing"]
        except HandoffInvalid as exc:
            return list(exc.errors) or [str(exc)]
        if handoff.meta.stage != HandoffKind.STRATEGY:
            return [f"frontmatter stage must be {HandoffKind.STRATEGY.value!r}, got {handoff.meta.stage.value!r}"]
        return validate_handoff(handoff)

    def _read_review_note(self, run_id: str) -> str | None:
        path = self.review_note_path(run_id)
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8").strip()
        return text or None

    def _advance(self, index: RunIndex, progress: ProgressFn | None) -> RunIndex:
        """Run stages until a stop status. Every exit path writes run.md."""
        run_id = index.run_id
        paths = self.vault.paths(run_id)
        ledger = self.ledger_for(index)
        settings = self.run_settings(index)

        index.status = RunStatus.RUNNING
        index.error = None
        self._mirror(index, ledger)
        self.vault.write_index(index)

        try:
            providers = self.providers_factory(settings, paths.workspace)
        except Exception as exc:  # noqa: BLE001 - configuration errors become a FAILED run
            log.exception("run %s: building providers failed", run_id)
            return self._stop(index, ledger, RunStatus.FAILED, f"providers: {_describe(exc)}", progress)

        review_note = self._read_review_note(run_id)
        while True:
            stage = index.stage
            if self._stop_event.is_set():
                return self._stop(index, ledger, RunStatus.FAILED, f"{stage}: interrupted (shutdown requested)", progress)
            self._emit(progress, index, f"{stage} (round {index.round}) started")
            try:
                output = self._run_stage(index, settings, paths, ledger, providers, review_note)
                index = self._persist(index, stage, output, ledger)
            except BudgetExceeded as exc:
                return self._stop(index, ledger, RunStatus.BUDGET_EXCEEDED, str(exc), progress)
            except HandoffInvalid as exc:
                return self._stop(index, ledger, RunStatus.FAILED, f"{stage}: {exc}", progress)
            except ProviderError as exc:
                # Reaching here means no retry applied (non-retryable, Claude Code, or retries used up): stop
                # now rather than let the crosscheck loop re-pay against a broken provider or sandbox.
                log.error("run %s: stage %s: provider error (retryable=%s): %s", run_id, stage, exc.retryable, exc)
                return self._stop(index, ledger, RunStatus.FAILED, f"{stage}: {_describe(exc)}", progress)
            except KeyboardInterrupt:
                self._stop(index, ledger, RunStatus.FAILED, f"{stage}: interrupted", progress)
                raise
            except Exception as exc:  # noqa: BLE001 - run() must never raise for stage errors
                log.exception("run %s: stage %s failed", run_id, stage)
                return self._stop(index, ledger, RunStatus.FAILED, f"{stage}: {_describe(exc)}", progress)

            self._emit(progress, index, self._done_message(stage, index))
            if index.status != RunStatus.RUNNING:
                return index

    def _run_stage(
        self,
        index: RunIndex,
        settings: Settings,
        paths: RunPaths,
        ledger: Ledger,
        providers: Providers,
        review_note: str | None,
    ) -> StageOutput:
        ctx = StageContext(
            index=index.model_copy(deep=True),
            settings=settings,
            vault=self.vault,
            paths=paths,
            ledger=ledger,
            providers=providers,
            stage=index.stage,
            now=self.clock(),
            review_note=review_note,
        )
        output = self.backends[index.stage].run_stage(ctx)
        if not isinstance(output, StageOutput):
            raise TypeError(f"{index.stage} backend returned {type(output).__name__}, expected StageOutput")
        return output

    def _persist(self, index: RunIndex, stage: StageName, output: StageOutput, ledger: Ledger) -> RunIndex:
        """Notes, then handoffs, then allowed index updates, then ledger totals, then run.md. The previous pass's
        ``criteria_unmet``/``unmet_criteria`` stay in the index (an extra round's execution can read them) until final
        runs again and replaces them."""
        for note in output.notes:
            self.vault.write_handoff(index.run_id, note.name, note.handoff)

        updated = index.model_copy(deep=True)
        for note in output.notes:
            if note.name not in updated.handoffs:
                updated.handoffs.append(note.name)
        updates = output.index_updates
        if stage == "final":  # final judges the criteria afresh: a key it leaves out means nothing is unmet
            updates = {"criteria_unmet": 0, "unmet_criteria": [], **updates}
        updated = self._apply_updates(updated, updates)
        self._mirror(updated, ledger)

        step = next_step(updated, stage, output, self.settings.max_crosscheck_loops)
        updated.completed_stages.append(stage)
        updated.stage = step.stage or stage
        updated.round = step.round
        updated.status = step.status
        updated.updated = self.clock()
        self.vault.write_index(updated)
        return updated

    @staticmethod
    def _apply_updates(index: RunIndex, updates: dict[str, Any]) -> RunIndex:
        rejected = sorted(set(updates) - ALLOWED_INDEX_UPDATES)
        if rejected:
            log.warning("run %s: ignoring disallowed index updates %s", index.run_id, rejected)
        allowed = {key: value for key, value in updates.items() if key in ALLOWED_INDEX_UPDATES}
        if not allowed:
            return index
        try:
            return RunIndex.model_validate({**index.model_dump(), **allowed})
        except ValidationError as exc:
            raise ValueError(f"invalid index update {allowed!r}: {exc.errors()[0]['msg']}") from exc

    @staticmethod
    def _mirror(index: RunIndex, ledger: Ledger) -> None:
        index.budget_usd = ledger.cap_usd
        index.spent_usd = round(ledger.spent_usd, 6)
        index.spend_by_agent = {agent: round(usd, 6) for agent, usd in ledger.by_agent().items()}
        index.spend_by_provider = {provider: round(usd, 6) for provider, usd in ledger.by_provider().items()}

    def _stop(
        self,
        index: RunIndex,
        ledger: Ledger,
        status: RunStatus,
        error: str,
        progress: ProgressFn | None,
    ) -> RunIndex:
        index.status = status
        index.error = error
        index.updated = self.clock()
        try:
            self._mirror(index, ledger)
        except Exception:  # noqa: BLE001 - never mask the original failure
            log.exception("run %s: mirroring ledger totals failed", index.run_id)
        self.vault.write_index(index)
        self._emit(progress, index, f"{index.stage} stopped: {status.value}: {error}")
        return index

    @staticmethod
    def _done_message(stage: StageName, index: RunIndex) -> str:
        spend = f"${index.spent_usd:.4f} of ${index.budget_usd:.2f}"
        if index.status == RunStatus.COMPLETED:
            return f"{stage} done; run completed ({spend})"
        if index.status == RunStatus.COMPLETED_WITH_ISSUES:
            return f"{stage} done; run completed with issues: {describe_issues(index)} ({spend})"
        if index.status == RunStatus.AWAITING_REVIEW:
            return f"{stage} done; awaiting review of 02-strategy ({spend})"
        return f"{stage} done; next {index.stage} round {index.round} ({spend})"

    @staticmethod
    def _emit(progress: ProgressFn | None, index: RunIndex, message: str) -> None:
        if progress is None:
            return
        try:
            progress(index, message)
        except Exception:  # noqa: BLE001 - an observer must not break a run
            log.exception("progress callback failed")
