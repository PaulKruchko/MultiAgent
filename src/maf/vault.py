"""The Obsidian vault: run folder layout, atomic writes, run.md index, wikilinks, asset/deliverable copies.

Owner: vault+handoff.

Layout::

    <vault>/runs/<YYYY-MM-DD>-<slug>/
        run.md                       run index (RunIndex frontmatter + rendered body)
        ledger.jsonl                 cost ledger (maf.ledger), not an Obsidian note
        01a-routing.md  01-ingestion.md  02-strategy.md
        03-execution.md              round 1; later rounds: 03-execution-r2.md, 03-execution-r3.md
        04a-critique-chatgpt.md  04a-critique-gemini.md  04a-critique-claude.md   (+ -r2 ...)
        04b-rebuttal.md  04c-adjudication.md  04-crosscheck.md                  (+ -r2 ...)
        05-final.md
        assets/                      images/plots, embedded with ![[name.png]]
        deliverables/                final artifacts: the workspace tree (code/mixed) or the listed artifacts (prose)
        source-audit.json            latest source-audit state per document (cross-check writes, final reads)
        cleanroom.json               last judged clean-room result and its digest (final)
    <workspaces>/<run_id>/           code trees, sim data, inputs/ (user files); outside the vault
    <workspaces>/.maf-cleanroom/<run_id>/   the final stage's clean room, beside the workspace

All writes go to a temp file in the same directory, then ``os.replace`` (atomic on POSIX). A workspace export
(``Vault.export_workspace``) is staged in a dot-folder of the run folder and swapped in as a whole.
Note names are unique within a run folder, so wikilinks use bare names: ``[[01-ingestion]]``.
Links to other runs use path form: ``[[runs/<run_id>/05-final|...]]``.
"""

from __future__ import annotations

import filecmp
import fnmatch
import os
import re
import secrets
import shutil
import stat
import tempfile
import unicodedata
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from maf import lint as _lint
from maf.config import check_workspaces_outside_vault
from maf.handoff import Handoff, HandoffKind, dump_frontmatter, load_frontmatter, parse_handoff, render_handoff
from maf.types import AgentName, ExecutionMode, ProviderName, RunStatus, StageName, Tier


class RunIndex(BaseModel):
    """run.md frontmatter; the pipeline's single source of truth for resume."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus = RunStatus.PENDING
    stage: StageName = "ingestion"
    """The stage to run next (or the one running/failed). ``final`` + ``completed``/``completed_with_issues`` means done."""
    completed_stages: list[StageName] = Field(default_factory=list)
    round: int = 1
    """Current execution/crosscheck pass (1-based)."""
    mode: ExecutionMode | None = None
    """Set by ingestion triage."""
    tier: Tier = "default"
    review: bool = False
    budget_usd: float
    spent_usd: float = 0.0
    spend_by_agent: dict[AgentName, float] = Field(default_factory=dict)
    spend_by_provider: dict[ProviderName, float] = Field(default_factory=dict)
    unresolved_critical: int = 0
    """Critical issues left unresolved by the latest cross-check; > 0 at final means ``completed_with_issues``."""
    criteria_unmet: int = Field(default=0, ge=0)
    """Hard acceptance criteria the final stage found not ``met`` (02-strategy's, plus maf's own gates: the clean-room
    reproduction of code/mixed runs, the source audit and the deliverable lint); > 0 at final means
    ``completed_with_issues``. Absent from run.md files written before it
    existed."""
    unmet_criteria: list[str] = Field(default_factory=list)
    """One line per criterion counted in ``criteria_unmet``: ``<id> [partial|unmet]: <criterion>``."""
    created: datetime
    updated: datetime
    workspace: str
    """Absolute path of ``workspaces/<run_id>``."""
    handoffs: list[str] = Field(default_factory=list)
    """Note names written so far, in order (no ``.md``), rendered as wikilinks in the body."""
    error: str | None = None
    brief: str
    """The raw user request (also rendered as a quoted block in the body)."""
    input_files: list[str] = Field(default_factory=list)
    """Workspace-relative paths of user-supplied files (under ``inputs/``)."""
    exported_at: datetime | None = None
    """When ``deliverables/`` was last written (by the final stage or ``maf export``)."""
    export_note: str | None = None
    """One line on that export, e.g. ``maf export: 57 file(s), 1.2 MB``."""
    tags: list[str] = Field(default_factory=lambda: ["maf", "maf/run"])

    @model_validator(mode="after")
    def _completed_with_open_criticals(self) -> RunIndex:
        """``completed`` with ``unresolved_critical > 0`` (or ``criteria_unmet > 0``) reads as
        ``completed_with_issues``. The pipeline never writes that pair, but runs finished before
        ``completed_with_issues`` existed still have it in run.md."""
        if self.status == RunStatus.COMPLETED and (self.unresolved_critical > 0 or self.criteria_unmet > 0):
            self.status = RunStatus.COMPLETED_WITH_ISSUES
        return self

    @property
    def has_issues(self) -> bool:
        """Final ran (or would end) with open issues: unresolved critical issues or unmet acceptance criteria."""
        return self.unresolved_critical > 0 or self.criteria_unmet > 0


class RunPaths(BaseModel):
    """Resolved filesystem locations for one run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    root: Path
    run_md: Path
    ledger: Path
    assets: Path
    deliverables: Path
    workspace: Path

    def note(self, name: str) -> Path:
        """``root / f"{name}.md"``."""
        return self.root / f"{name}.md"


_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ASSET_UNSAFE_RE = re.compile(r"[\[\]|#^:\\/<>\"*?\x00-\x1f]")
_SUFFIXED_KINDS = frozenset(
    {HandoffKind.EXECUTION, HandoffKind.CRITIQUE, HandoffKind.REBUTTAL, HandoffKind.ADJUDICATION, HandoffKind.CROSSCHECK}
)
_NOTE_BASE: dict[HandoffKind, str] = {
    HandoffKind.ROUTING: "01a-routing",
    HandoffKind.INGESTION: "01-ingestion",
    HandoffKind.STRATEGY: "02-strategy",
    HandoffKind.EXECUTION: "03-execution",
    HandoffKind.CRITIQUE: "04a-critique",
    HandoffKind.REBUTTAL: "04b-rebuttal",
    HandoffKind.ADJUDICATION: "04c-adjudication",
    HandoffKind.CROSSCHECK: "04-crosscheck",
    HandoffKind.FINAL: "05-final",
}
_TREE_IGNORE = frozenset({".git", ".hg", ".svn", "__pycache__", ".maf"})

PROTECTED_EXPORT_EXCLUDES: tuple[str, ...] = (
    # pipeline and tool state, the provisioned FreeRTOS kernel, the copied user inputs (root inputs/ only), VCS
    ".maf",
    ".claude",
    "FreeRTOS-Kernel",
    "inputs/*",
    ".git",
    ".hg",
    ".svn",
)
"""The part of ``DEFAULT_EXPORT_EXCLUDES`` nothing re-includes: ``maf.stages.base.export_excludes`` repeats these after
``Settings.export_include``, so pipeline notes, the user's own files and repository state never ship."""

DEFAULT_EXPORT_EXCLUDES: tuple[str, ...] = (
    *PROTECTED_EXPORT_EXCLUDES,
    # build output: directories named build at any depth (not a build script), in-source objects, libraries and
    # executables, CMake state
    "build/*",
    "*/build/*",
    "*.o",
    "*.obj",
    "*.a",
    "*.so",
    "*.dylib",
    "*.elf",
    "a.out",
    "CMakeFiles",
    "CMakeCache.txt",
    # caches and environments
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.egg-info",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    # the clean room's result files (maf.stages.final.REPRO_EXIT, REPRO_LOG): never carried into a fresh room
    "REPRO_EXIT",
    "REPRO_LOG",
)
"""What ``Vault.export_workspace`` leaves out, with the semantics of ``Settings.export_exclude``
(``maf.lint.excluded``): a case-sensitive ``fnmatch`` pattern matches any component of the workspace-relative POSIX
path, or a leading part of it. A pattern containing ``/`` can only match from the root, so ``inputs/*`` is the root
``inputs/`` folder alone, and ``build/*`` leaves a ``build`` directory's contents out (not a file named ``build``).
``maf.stages.base.export_excludes`` adds ``Settings.export_exclude``, then re-includes ``Settings.export_include``
(``!`` patterns, which can bring back anything here except ``PROTECTED_EXPORT_EXCLUDES``), for the export, the lint and
the source audit alike. Build output outside these patterns (an extensionless binary, ``results/``) still ships; the
clean room makes it look stale, so the reproduction command regenerates it."""

SANDBOX_PLACEHOLDER_FILES: tuple[str, ...] = (
    ".env",
    ".env.*",
    ".npmrc",
    ".yarnrc",
    ".yarnrc.yml",
    "bunfig.toml",
    "bun.lock",
    "bun.lockb",
    "package.json",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    ".mcp.json",
    ".bashrc",
    ".bash_profile",
    ".profile",
    ".zshrc",
    ".zprofile",
    ".gitconfig",
    ".gitmodules",
    ".ripgreprc",
    ".idea",  # editor config dirs, mounted as empty files (seen live 2026-09-29)
    ".vscode",
)
"""Names (``fnmatch``) of the zero-byte files Claude Code's sandbox may leave at the workspace root as mount points
for paths it protects. An export leaves them out only when they are empty and at the root: a real ``package.json``
is a deliverable."""

MAX_EXPORT_BYTES = 200 * 1024 * 1024
"""Size cap of one workspace export (200 MiB, as ``Settings.export_max_mb``). Bigger trees are refused
(``ExportTooLarge``) before anything is copied."""

MAX_EXPORT_FILES = 20_000
"""File-count cap of one workspace export."""

MAX_EXCLUDED_SHOWN = 5
"""``ExportResult.describe`` names this many excluded entries; the rest are counted."""

_EXPORT_STAGING = ".deliverables-"
"""Prefix of the dot-folders an export stages in (in the run folder, so the swap is a same-filesystem rename)."""


class ExportError(ValueError):
    """A workspace export could not be made; ``deliverables/`` was left as it was."""


class ExportTooLarge(ExportError):
    """The export would exceed its size or file-count cap."""


@dataclass(frozen=True)
class ExportResult:
    """What ``Vault.export_workspace`` put into ``deliverables/``."""

    deliverables: Path
    files: tuple[str, ...]
    """Exported paths, workspace-relative POSIX (the same path under ``deliverables/``), sorted."""
    total_bytes: int
    placeholders: tuple[str, ...] = ()
    """Empty sandbox placeholder files left out (``SANDBOX_PLACEHOLDER_FILES``)."""
    skipped: tuple[str, ...] = ()
    """Entries left out for safety, each with the reason: symlinked directories, symlinks that leave the workspace,
    dangle or point into an excluded path, and anything that is not a regular file."""
    excluded: tuple[str, ...] = ()
    """What the exclude patterns left out, apart from ``PROTECTED_EXPORT_EXCLUDES`` (pipeline state, the kernel,
    ``inputs/``, VCS): each entry is the highest directory with nothing exported below it (``build/``,
    ``src/__pycache__/``), else the file itself (``prog.o``); sorted. ``Settings.export_include`` brings one back."""

    def excluded_note(self) -> str:
        """``excluded: build/, prog.o and 2 more`` (at most ``MAX_EXCLUDED_SHOWN`` named, the rest counted);
        empty when nothing was excluded."""
        if not self.excluded:
            return ""
        more = len(self.excluded) - MAX_EXCLUDED_SHOWN
        shown = ", ".join(self.excluded[:MAX_EXCLUDED_SHOWN]) + (f" and {more} more" if more > 0 else "")
        return f"excluded: {shown}"

    def describe(self) -> str:
        text = f"{len(self.files)} file(s), {format_bytes(self.total_bytes)}"
        return f"{text}; {self.excluded_note()}" if self.excluded else text


def format_bytes(size: int) -> str:
    """``812 B``, ``3.4 kB``, ``1.2 MB``, ``1.5 GB`` (decimal units)."""
    value = float(size)
    for unit in ("B", "kB", "MB"):
        if value < 1000:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} GB"


def export_excluded(rel: str, patterns: Sequence[str]) -> bool:
    """Whether ``patterns`` exclude the workspace-relative POSIX path ``rel`` (``maf.lint.excluded``; see
    ``DEFAULT_EXPORT_EXCLUDES``)."""
    return _lint.excluded(rel, patterns)


def slugify(text: str, max_len: int = 48) -> str:
    """Lowercase ASCII, ``[a-z0-9-]`` only, collapsed dashes, trimmed at a word boundary to ``max_len``.
    Empty result becomes ``"run"``."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    if len(slug) > max_len:
        cut = slug[:max_len]
        if slug[max_len] != "-" and "-" in cut:
            cut = cut[: cut.rindex("-")]
        slug = cut.strip("-")
    return slug or "run"


def note_name(kind: HandoffKind, round: int = 1, agent: AgentName | None = None) -> str:
    """Canonical note name (no ``.md``) per the layout above. ``agent`` is required for CRITIQUE only.
    Round 1 has no suffix; round n>1 appends ``-r{n}``. Single-pass kinds (routing, ingestion, strategy,
    final) ignore ``round``."""
    if round < 1:
        raise ValueError(f"round must be >= 1, got {round}")
    base = _NOTE_BASE[kind]
    if kind == HandoffKind.CRITIQUE:
        if agent not in get_args(AgentName):
            raise ValueError(f"critique notes need an agent (chatgpt/gemini/claude), got {agent!r}")
        base = f"{base}-{agent}"
    elif agent is not None:
        raise ValueError(f"agent is only used for critique notes, not {kind.value}")
    if kind in _SUFFIXED_KINDS and round > 1:
        base = f"{base}-r{round}"
    return base


def wikilink(target: str, alias: str | None = None) -> str:
    """``[[target]]`` or ``[[target|alias]]``. ``target`` must not include ``.md``."""
    if not target or target.endswith(".md"):
        raise ValueError(f"wikilink target must be non-empty and without '.md': {target!r}")
    if any(bad in target for bad in ("[[", "]]", "|", "\n")):
        raise ValueError(f"invalid wikilink target: {target!r}")
    if alias is None:
        return f"[[{target}]]"
    if "]]" in alias or "\n" in alias:
        raise ValueError(f"invalid wikilink alias: {alias!r}")
    return f"[[{target}|{alias}]]"


def embed(target: str) -> str:
    """``![[target]]``: for assets, keep the extension (``![[plot.png]]``)."""
    if not target or any(bad in target for bad in ("[[", "]]", "|", "\n")):
        raise ValueError(f"invalid embed target: {target!r}")
    return f"![[{target}]]"


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _umask() -> int:
    umask = os.umask(0)
    os.umask(umask)
    return umask


def _default_mode() -> int:
    return 0o666 & ~_umask()


def _atomic_replace(dst: Path, fill: Callable[[BinaryIO, Path], None]) -> None:
    """Create a temp file next to ``dst``, let ``fill`` write it, fsync, then ``os.replace``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fill(fh, tmp)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(dst.parent)


def atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 via a temp file in ``path.parent``, fsync, then ``os.replace``. Creates parent dirs.
    An existing file's permission bits are kept; new files get the umask default."""
    data = text.encode("utf-8")
    try:
        mode = path.stat().st_mode & 0o7777
    except FileNotFoundError:
        mode = _default_mode()

    def fill(fh: BinaryIO, tmp: Path) -> None:
        fh.write(data)
        os.chmod(tmp, mode)

    _atomic_replace(path, fill)


def atomic_copy(src: Path, dst: Path) -> None:
    """Copy a file atomically (temp + ``os.replace``), preserving mtime (and permission bits)."""
    if not src.is_file():
        raise FileNotFoundError(f"not a file: {src}")

    def fill(fh: BinaryIO, tmp: Path) -> None:
        with src.open("rb") as source:
            shutil.copyfileobj(source, fh, length=1024 * 1024)
        fh.flush()
        shutil.copystat(src, tmp)

    _atomic_replace(dst, fill)


def _check_name(name: str, what: str) -> str:
    if not _SAFE_NAME_RE.match(name) or ".." in name:
        raise ValueError(f"invalid {what}: {name!r}")
    return name


def _unique_path(directory: Path, name: str) -> Iterator[Path]:
    """``name``, then ``stem-2.ext``, ``stem-3.ext``, ..."""
    stem, suffix = os.path.splitext(name)
    yield directory / name
    n = 2
    while True:
        yield directory / f"{stem}-{n}{suffix}"
        n += 1


class Vault:
    """Filesystem access to the vault and workspaces. Stateless apart from its root paths.
    Safe for concurrent use across different runs; within a run, the pipeline serializes writes."""

    def __init__(self, root: Path, workspaces_root: Path) -> None:
        """``ValueError`` if ``workspaces_root`` is the vault or inside it."""
        check_workspaces_outside_vault(root, workspaces_root)
        self.root = root
        self.workspaces_root = workspaces_root

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def new_run_id(self, brief: str, today: date) -> str:
        """``f"{today:%Y-%m-%d}-{slugify(brief)}"``, with ``-2``, ``-3`` appended until unused
        (in both the vault and the workspaces root)."""
        base = f"{today:%Y-%m-%d}-{slugify(brief)}"
        candidate, n = base, 1
        while (self.runs_dir / candidate).exists() or (self.workspaces_root / candidate).exists():
            n += 1
            candidate = f"{base}-{n}"
        return candidate

    def create_run(self, index: RunIndex, input_files: list[Path]) -> RunPaths:
        """Create the run folder, assets/, deliverables/, and the workspace with ``inputs/``;
        copy ``input_files`` into ``workspace/inputs/`` (recording them in ``index.input_files``);
        write run.md. Raises ``FileExistsError`` if the run folder exists.

        Inputs are checked before anything is created; on a later failure the new folders are removed.
        """
        paths = self.paths(index.run_id)
        for src in input_files:
            if not Path(src).is_file():
                raise FileNotFoundError(f"input file not found: {src}")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        paths.root.mkdir()  # FileExistsError if the run already exists
        workspace_existed = paths.workspace.exists()
        try:
            paths.assets.mkdir()
            paths.deliverables.mkdir()
            inputs_dir = paths.workspace / "inputs"
            inputs_dir.mkdir(parents=True, exist_ok=True)
            recorded = list(index.input_files)
            for src in input_files:
                src = Path(src)
                dst = next(p for p in _unique_path(inputs_dir, src.name) if not p.exists())
                atomic_copy(src, dst)
                recorded.append(dst.relative_to(paths.workspace).as_posix())
            index.input_files = recorded
            self.write_index(index)
        except BaseException:
            shutil.rmtree(paths.root, ignore_errors=True)
            if not workspace_existed:
                shutil.rmtree(paths.workspace, ignore_errors=True)
            raise
        return paths

    def paths(self, run_id: str) -> RunPaths:
        """Pure path resolution (no I/O). ``ValueError`` for ids that are not a single safe path segment."""
        _check_name(run_id, "run id")
        root = self.runs_dir / run_id
        return RunPaths(
            run_id=run_id,
            root=root,
            run_md=root / "run.md",
            ledger=root / "ledger.jsonl",
            assets=root / "assets",
            deliverables=root / "deliverables",
            workspace=self.workspaces_root / run_id,
        )

    def list_runs(self) -> list[RunIndex]:
        """Every parseable ``runs/*/run.md``, newest ``created`` first. Unparseable ones are skipped."""
        if not self.runs_dir.is_dir():
            return []
        runs: list[RunIndex] = []
        for run_md in self.runs_dir.glob("*/run.md"):
            try:
                runs.append(self._load_index(run_md))
            except (OSError, ValueError):
                continue
        runs.sort(key=lambda r: r.created.timestamp(), reverse=True)
        return runs

    def read_index(self, run_id: str) -> RunIndex:
        """Parse run.md frontmatter. ``FileNotFoundError`` if the run does not exist; ``ValueError`` if
        run.md is malformed."""
        run_md = self.paths(run_id).run_md
        if not run_md.is_file():
            raise FileNotFoundError(f"no such run: {run_id}")
        return self._load_index(run_md)

    @staticmethod
    def _load_index(run_md: Path) -> RunIndex:
        data, _ = load_frontmatter(run_md.read_text(encoding="utf-8"))
        if not data:
            raise ValueError(f"{run_md}: missing frontmatter")
        try:
            return RunIndex.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"{run_md}: invalid run index: {exc}") from None

    def write_index(self, index: RunIndex) -> None:
        """Atomically rewrite run.md: frontmatter plus ``render_run_body(index)``."""
        paths = self.paths(index.run_id)
        data = index.model_dump(mode="json")
        data["created"] = index.created
        data["updated"] = index.updated
        if index.exported_at is not None:
            data["exported_at"] = index.exported_at
        atomic_write_text(paths.run_md, dump_frontmatter(data) + "\n" + render_run_body(index))

    def write_handoff(self, run_id: str, name: str, handoff: Handoff) -> Path:
        """Atomically write ``<name>.md`` via ``maf.handoff.render_handoff``. Does not touch run.md."""
        if handoff.meta.run_id != run_id:
            raise ValueError(f"handoff belongs to run {handoff.meta.run_id!r}, not {run_id!r}")
        path = self.paths(run_id).note(_check_name(name, "note name"))
        atomic_write_text(path, render_handoff(handoff))
        return path

    def read_handoff(self, run_id: str, name: str) -> Handoff:
        """Parse ``<name>.md``. The file may have been hand-edited by the user (review gate).
        ``FileNotFoundError`` if absent; ``HandoffInvalid`` if the frontmatter is malformed."""
        path = self.paths(run_id).note(_check_name(name, "note name"))
        return parse_handoff(path.read_text(encoding="utf-8"))

    def has_note(self, run_id: str, name: str) -> bool:
        return self.paths(run_id).note(_check_name(name, "note name")).is_file()

    def copy_asset(self, run_id: str, src: Path) -> str:
        """Copy into ``assets/`` (name collisions get ``-2``, ``-3`` suffixes) and return the embed.

        Re-copying identical content reuses the existing file (idempotent stage re-runs). The embed uses
        the vault-relative path (``![[runs/<run_id>/assets/plot.png]]``) because asset names repeat
        across runs and a bare ``![[plot.png]]`` would be ambiguous in Obsidian.
        """
        paths = self.paths(run_id)
        src = Path(src)
        if not src.is_file():
            raise FileNotFoundError(f"asset not found: {src}")
        stem, suffix = os.path.splitext(src.name)
        clean = _ASSET_UNSAFE_RE.sub("-", stem).strip(" .-") or "asset"
        paths.assets.mkdir(parents=True, exist_ok=True)
        for dst in _unique_path(paths.assets, clean + suffix.lower()):
            if not dst.exists():
                atomic_copy(src, dst)
                break
            if dst.is_file() and filecmp.cmp(src, dst, shallow=False):
                break
        rel = dst.relative_to(self.root).as_posix()
        return embed(rel)

    def copy_deliverable(self, run_id: str, src: Path, rel_name: str | None = None) -> Path:
        """Copy a file or directory tree into ``deliverables/`` (``rel_name`` defaults to ``src.name``).
        ``src`` must resolve inside the run's workspace, else ``ValueError`` (path traversal guard).

        Trees skip VCS folders, ``__pycache__`` and ``.maf``; symlinked directories are skipped and
        symlinked files are copied only if their target is inside the workspace and not in
        ``PROTECTED_EXPORT_EXCLUDES``. An existing destination
        is replaced, so re-running a stage is idempotent.
        """
        paths = self.paths(run_id)
        workspace = paths.workspace.resolve()
        resolved = Path(src).resolve()
        if not resolved.is_relative_to(workspace):
            raise ValueError(f"deliverable {src} is outside the workspace {workspace}")
        if not resolved.exists():
            raise FileNotFoundError(f"deliverable not found: {src}")
        rel = Path(rel_name if rel_name is not None else resolved.name)
        paths.deliverables.mkdir(parents=True, exist_ok=True)
        deliverables = paths.deliverables.resolve()
        dst = deliverables / rel
        if rel.is_absolute() or ".." in rel.parts or not rel.parts or not dst.resolve().is_relative_to(deliverables):
            raise ValueError(f"invalid deliverable name: {rel_name!r}")
        if resolved.is_file():
            if dst.is_dir():
                shutil.rmtree(dst)
            atomic_copy(resolved, dst)
            return dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=dst.parent, prefix=f".{dst.name}.", suffix=".tmp"))
        try:
            self._copy_tree(resolved, staging, workspace)
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            elif dst.exists() or dst.is_symlink():
                dst.unlink()
            os.replace(staging, dst)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        _fsync_dir(dst.parent)
        return dst

    def export_workspace(
        self,
        run_id: str,
        *,
        excludes: Sequence[str] | None = None,
        placeholders: Sequence[str] | None = None,
        max_bytes: int | None = None,
        max_files: int | None = None,
    ) -> ExportResult:
        """Replace ``deliverables/`` with a copy of the run's workspace tree (code and mixed runs).

        Left out: paths ``excludes`` exclude (default ``DEFAULT_EXPORT_EXCLUDES``; ``ExportResult.excluded`` reports
        them, pipeline state aside), empty files at the workspace root whose names match ``placeholders`` (default
        ``SANDBOX_PLACEHOLDER_FILES``), symlinked directories (never followed), file symlinks that leave the workspace,
        dangle or point into an excluded path (``.maf/``, ``inputs/``: lint and the source audit never read those), and
        anything that is not a regular file. Any other file symlink inside the workspace is exported as a regular file
        with its target's content. Empty directories are not recreated.

        Nothing is copied when the tree exceeds ``max_bytes`` (default ``MAX_EXPORT_BYTES``) or ``max_files`` (default
        ``MAX_EXPORT_FILES``): ``ExportTooLarge`` names the largest top-level entries. The copy is staged in a
        dot-folder of the run folder, then swapped in by two renames, so ``deliverables/`` is never half-written and
        nothing from a previous export survives; staging folders left by a crashed export are removed first.
        ``ExportError`` when the workspace is missing or a file cannot be read; ``FileNotFoundError`` for an unknown
        run.
        """
        paths = self.paths(run_id)
        if not paths.root.is_dir():
            raise FileNotFoundError(f"no such run: {run_id}")
        if not paths.workspace.is_dir():
            raise ExportError(f"workspace {paths.workspace} does not exist")
        patterns = DEFAULT_EXPORT_EXCLUDES if excludes is None else tuple(excludes)
        names = SANDBOX_PLACEHOLDER_FILES if placeholders is None else tuple(placeholders)
        byte_cap = MAX_EXPORT_BYTES if max_bytes is None else max_bytes
        file_cap = MAX_EXPORT_FILES if max_files is None else max_files

        plan, left_out, skipped, excluded = _plan_export(paths.workspace.resolve(), patterns, names)
        total = sum(size for _, _, size in plan)
        if total > byte_cap or len(plan) > file_cap:
            raise ExportTooLarge(_too_large_message(run_id, plan, total, byte_cap, file_cap))

        _remove_export_leftovers(paths.root)
        staging = Path(tempfile.mkdtemp(dir=paths.root, prefix=_EXPORT_STAGING, suffix=".tmp"))
        try:
            os.chmod(staging, 0o777 & ~_umask())  # mkdtemp makes it 0700; it becomes deliverables/
            for rel, source, _ in plan:
                target = staging / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(source, target)
                except OSError as exc:
                    raise ExportError(f"cannot export {rel}: {exc}") from exc
            _swap_in(staging, paths.deliverables)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return ExportResult(
            deliverables=paths.deliverables,
            files=tuple(rel for rel, _, _ in plan),
            total_bytes=total,
            placeholders=tuple(left_out),
            skipped=tuple(skipped),
            excluded=tuple(excluded),
        )

    @staticmethod
    def _copy_tree(src_dir: Path, dst_dir: Path, workspace: Path) -> None:
        for dirpath, dirnames, filenames in os.walk(src_dir):
            here = Path(dirpath)
            dirnames[:] = sorted(d for d in dirnames if d not in _TREE_IGNORE and not (here / d).is_symlink())
            target_dir = dst_dir / here.relative_to(src_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            for filename in sorted(filenames):
                file = here / filename
                real = file.resolve()
                if not real.is_relative_to(workspace):
                    raise ValueError(f"{file} links outside the workspace")
                target = real.relative_to(workspace).as_posix()
                if file.is_symlink() and _lint.excluded(target, PROTECTED_EXPORT_EXCLUDES):
                    continue  # a link into pipeline state or the user's inputs: never shipped
                if real.is_file():
                    atomic_copy(real, target_dir / filename)


def _plan_export(
    root: Path, patterns: Sequence[str], placeholders: Sequence[str]
) -> tuple[list[tuple[str, Path, int]], list[str], list[str], list[str]]:
    """``(plan, placeholders_left_out, skipped, excluded)`` for ``export_workspace``: ``plan`` holds
    ``(rel, source, size)`` in walk order (sorted per directory), ``excluded`` is ``ExportResult.excluded``. ``root``
    is resolved; symlinked directories are never entered."""
    plan: list[tuple[str, Path, int]] = []
    left_out: list[str] = []
    skipped: list[str] = []
    dropped: list[tuple[str, bool]] = []  # (rel, is_dir) excluded by a pattern that export_include may undo
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        base = here.relative_to(root).as_posix()
        kept: list[str] = []
        for name in sorted(dirnames):
            rel = name if base == "." else f"{base}/{name}"
            if _lint.excluded_dir(rel, patterns):
                dropped.append((rel, True))
                continue
            if (here / name).is_symlink():
                skipped.append(f"{rel}/ (symlinked directory)")
                continue
            kept.append(name)
        dirnames[:] = kept
        for name in sorted(filenames):
            rel = name if base == "." else f"{base}/{name}"
            if export_excluded(rel, patterns):
                dropped.append((rel, False))
                continue
            source = here / name
            info = source.lstat()
            if stat.S_ISLNK(info.st_mode):
                real = source.resolve()
                if not real.is_relative_to(root):
                    skipped.append(f"{rel} (symlink out of the workspace)")
                    continue
                target = real.relative_to(root).as_posix()
                if target != "." and export_excluded(target, patterns):
                    skipped.append(f"{rel} (symlink into the excluded {target})")
                    continue
                try:
                    info = real.stat()
                except OSError:
                    skipped.append(f"{rel} (dangling symlink)")
                    continue
                source = real
            if not stat.S_ISREG(info.st_mode):
                skipped.append(f"{rel} (not a regular file)")
                continue
            if info.st_size == 0 and base == "." and any(fnmatch.fnmatchcase(name, p) for p in placeholders):
                left_out.append(rel)
                continue
            plan.append((rel, source, info.st_size))
    plan.sort(key=lambda item: item[0])
    return plan, left_out, skipped, _excluded_entries(dropped, [rel for rel, _, _ in plan])


def _excluded_entries(dropped: Sequence[tuple[str, bool]], exported: Sequence[str]) -> list[str]:
    """``ExportResult.excluded``: each dropped path (``(rel, is_dir)``) outside ``PROTECTED_EXPORT_EXCLUDES``, raised to
    its highest ancestor directory with nothing exported below it (``dir/``), else kept as the file itself."""
    busy = {str(parent) for rel in exported for parent in PurePosixPath(rel).parents}
    entries: set[str] = set()
    for rel, is_dir in dropped:
        if _lint.excluded(rel, PROTECTED_EXPORT_EXCLUDES):
            continue
        top = rel if is_dir else None
        for parent in PurePosixPath(rel).parents:
            if str(parent) == "." or str(parent) in busy:
                break
            top = str(parent)
        entries.add(f"{top}/" if top is not None else rel)
    return sorted(entries)


def _too_large_message(
    run_id: str, plan: list[tuple[str, Path, int]], total: int, max_bytes: int, max_files: int
) -> str:
    by_top: dict[str, int] = {}
    for rel, _, size in plan:
        top = rel.split("/", 1)[0] + ("/" if "/" in rel else "")
        by_top[top] = by_top.get(top, 0) + size
    largest = sorted(by_top.items(), key=lambda item: (-item[1], item[0]))[:3]
    return (
        f"the workspace export of {run_id} would be {format_bytes(total)} in {len(plan)} file(s), over the cap of "
        f"{format_bytes(max_bytes)} and {max_files} files; largest entries: "
        + ", ".join(f"{name} ({format_bytes(size)})" for name, size in largest)
        + ". Remove bulky outputs from the workspace or exclude them (Settings.export_exclude), then export again."
    )


def _remove_export_leftovers(run_root: Path) -> None:
    for leftover in run_root.glob(f"{_EXPORT_STAGING}*.tmp"):
        if leftover.is_dir() and not leftover.is_symlink():
            shutil.rmtree(leftover, ignore_errors=True)
        else:
            leftover.unlink(missing_ok=True)


def _swap_in(staging: Path, dst: Path) -> None:
    """Move ``staging`` to ``dst``, replacing whatever ``dst`` was (moved aside first, deleted last). If the second
    rename fails, the previous ``dst`` is put back."""
    old: Path | None = None
    if dst.exists() or dst.is_symlink():
        old = dst.with_name(f"{_EXPORT_STAGING}old-{secrets.token_hex(4)}.tmp")
        os.replace(dst, old)
    try:
        os.replace(staging, dst)
    except BaseException:
        if old is not None and not (dst.exists() or dst.is_symlink()):
            os.replace(old, dst)
        raise
    _fsync_dir(dst.parent)
    if old is not None:
        if old.is_dir() and not old.is_symlink():
            shutil.rmtree(old, ignore_errors=True)
        else:
            old.unlink(missing_ok=True)


def _money(value: float) -> str:
    return f"{value:.4f}"


_CROSSCHECK_NOTE_RE = re.compile(rf"^{re.escape(_NOTE_BASE[HandoffKind.CROSSCHECK])}(?:-r\d+)?$")


def latest_crosscheck(index: RunIndex) -> str | None:
    """Name of the last ``04-crosscheck[-rN]`` note recorded in ``index.handoffs`` (None if there is none)."""
    return next((name for name in reversed(index.handoffs) if _CROSSCHECK_NOTE_RE.match(name)), None)


def _criteria_count(n: int) -> str:
    return f"{n} acceptance criteri{'on' if n == 1 else 'a'}"


def describe_issues(index: RunIndex) -> str:
    """One line on what keeps a finished run from ``completed``: ``3 unresolved critical issue(s) after the
    cross-check loop cap; 2 acceptance criteria not met (AC-2, clean-room)``. Empty when nothing is open."""
    parts: list[str] = []
    if index.unresolved_critical > 0:
        parts.append(f"{index.unresolved_critical} unresolved critical issue(s) after the cross-check loop cap")
    if index.criteria_unmet > 0:
        ids = ", ".join(line.split(" ", 1)[0] for line in index.unmet_criteria if line.strip())
        parts.append(f"{_criteria_count(index.criteria_unmet)} not met" + (f" ({ids})" if ids else ""))
    return "; ".join(parts)


def _issues_callout(index: RunIndex) -> list[str]:
    """The ``completed_with_issues`` warning that opens ``## Status`` (empty for every other status): the unresolved
    critical count (linking the last cross-check) and the unmet acceptance criteria, one per line."""
    if index.status != RunStatus.COMPLETED_WITH_ISSUES:
        return []
    final = wikilink(_NOTE_BASE[HandoffKind.FINAL])
    critical, unmet = index.unresolved_critical, index.criteria_unmet
    if critical > 0 and unmet > 0:
        title = f"Completed with {critical} unresolved critical issue(s) and {_criteria_count(unmet)} not met"
    elif unmet > 0:
        title = f"Completed with {_criteria_count(unmet)} not met"
    else:
        title = f"Completed with {critical} unresolved critical issue(s)"
    lines = [f"> [!warning] {title}"]
    if critical > 0:
        crosscheck = latest_crosscheck(index)
        where = f"; see `## Unresolved Critical` in {wikilink(crosscheck)}" if crosscheck else ""
        lines.append(
            f"> The cross-check loop cap was reached with {critical} critical issue(s) still open{where}. "
            f"{final} lists them under Limitations."
        )
    if unmet > 0:
        lines.append(f"> {_criteria_count(unmet)} not met; see `## Acceptance` in {final}:")
        lines += [f"> - {' '.join(line.split())}" for line in index.unmet_criteria]
    lines += ["> Do not treat this run's deliverables as verified.", ""]
    return lines


def render_run_body(index: RunIndex) -> str:
    """Body of run.md: ``# <run_id>``; ``## Brief`` (quoted); ``## Status`` (opened by a warning callout when the
    status is ``completed_with_issues``, see ``_issues_callout``); ``## Handoffs``
    (wikilink bullets); ``## Cost`` (markdown table Agent | USD with a Total row, then Provider | USD);
    ``## Workspace`` (the path as inline code)."""
    brief = "\n".join(f"> {line}" if line.strip() else ">" for line in index.brief.strip().split("\n"))
    status = _issues_callout(index) + [
        f"- Status: **{index.status.value}**",
        f"- Stage: {index.stage}",
        f"- Round: {index.round}",
        f"- Mode: {index.mode or 'undecided'}",
        f"- Tier: {index.tier}",
        f"- Review gate: {'on' if index.review else 'off'}",
        f"- Completed stages: {', '.join(index.completed_stages) or 'none'}",
        f"- Unresolved critical issues: {index.unresolved_critical}",
        f"- Acceptance criteria not met: {index.criteria_unmet}",
        f"- Budget: ${_money(index.spent_usd)} of ${_money(index.budget_usd)}",
        f"- Created: {index.created.isoformat()}",
        f"- Updated: {index.updated.isoformat()}",
    ]
    if index.exported_at is not None:
        note = f" ({' '.join(index.export_note.split())})" if index.export_note else ""
        status.append(f"- Deliverables exported: {index.exported_at.isoformat()}{note}")
    if index.error:
        error = " ".join(index.error.split())
        status.append(f"- Error: {error}")
    handoffs = "\n".join(f"- {wikilink(name)}" for name in index.handoffs) or "None yet."

    agent_rows = [f"| {a} | {_money(index.spend_by_agent.get(a, 0.0))} |" for a in get_args(AgentName)]
    provider_rows = [
        f"| {p} | {_money(index.spend_by_provider[p])} |" for p in get_args(ProviderName) if p in index.spend_by_provider
    ]
    cost = "\n".join(
        ["| Agent | USD |", "|---|---:|", *agent_rows, f"| **Total** | **{_money(index.spent_usd)}** |", "",
         "| Provider | USD |", "|---|---:|", *(provider_rows or ["| (none) | 0.0000 |"])]
    )
    tick = "``" if "`" in index.workspace else "`"
    pad = " " if tick == "``" else ""
    workspace = f"{tick}{pad}{index.workspace}{pad}{tick}"
    blocks = [
        f"# {index.run_id}",
        f"## Brief\n\n{brief}",
        "## Status\n\n" + "\n".join(status),
        f"## Handoffs\n\n{handoffs}",
        f"## Cost\n\n{cost}",
        f"## Workspace\n\n{workspace}",
    ]
    if index.input_files:
        blocks.append("## Inputs\n\n" + "\n".join(f"- `{f}`" for f in index.input_files))
    return "\n\n".join(blocks) + "\n"
