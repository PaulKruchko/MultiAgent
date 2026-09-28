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
        deliverables/                final artifacts copied from the workspace
    <workspaces>/<run_id>/           code trees, sim data, inputs/ (user files); outside the vault

All writes go to a temp file in the same directory, then ``os.replace`` (atomic on POSIX).
Note names are unique within a run folder, so wikilinks use bare names: ``[[01-ingestion]]``.
Links to other runs use path form: ``[[runs/<run_id>/05-final|...]]``.
"""

from __future__ import annotations

import filecmp
import os
import re
import shutil
import tempfile
import unicodedata
from collections.abc import Callable, Iterator
from datetime import date, datetime
from pathlib import Path
from typing import BinaryIO, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from maf.config import check_workspaces_outside_vault
from maf.handoff import Handoff, HandoffKind, dump_frontmatter, load_frontmatter, parse_handoff, render_handoff
from maf.types import AgentName, ExecutionMode, ProviderName, RunStatus, StageName, Tier


class RunIndex(BaseModel):
    """run.md frontmatter; the pipeline's single source of truth for resume."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus = RunStatus.PENDING
    stage: StageName = "ingestion"
    """The stage to run next (or the one running/failed). ``final`` + ``completed`` means done."""
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
    tags: list[str] = Field(default_factory=lambda: ["maf", "maf/run"])


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


def _default_mode() -> int:
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask


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
        symlinked files are copied only if their target is inside the workspace. An existing destination
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
                if real.is_file():
                    atomic_copy(real, target_dir / filename)


def _money(value: float) -> str:
    return f"{value:.4f}"


def render_run_body(index: RunIndex) -> str:
    """Body of run.md: ``# <run_id>``; ``## Brief`` (quoted); ``## Status``; ``## Handoffs``
    (wikilink bullets); ``## Cost`` (markdown table Agent | USD with a Total row, then Provider | USD);
    ``## Workspace`` (the path as inline code)."""
    brief = "\n".join(f"> {line}" if line.strip() else ">" for line in index.brief.strip().split("\n"))
    status = [
        f"- Status: **{index.status.value}**",
        f"- Stage: {index.stage}",
        f"- Round: {index.round}",
        f"- Mode: {index.mode or 'undecided'}",
        f"- Tier: {index.tier}",
        f"- Review gate: {'on' if index.review else 'off'}",
        f"- Completed stages: {', '.join(index.completed_stages) or 'none'}",
        f"- Unresolved critical issues: {index.unresolved_critical}",
        f"- Budget: ${_money(index.spent_usd)} of ${_money(index.budget_usd)}",
        f"- Created: {index.created.isoformat()}",
        f"- Updated: {index.updated.isoformat()}",
    ]
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
