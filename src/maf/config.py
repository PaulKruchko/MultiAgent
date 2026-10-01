"""Settings, model pins, tiers and the dated price table.

Owner: core (config + ledger).

Settings resolve in this order (later wins): built-in defaults, then
``~/.config/maf/config.yaml`` (or ``$MAF_CONFIG``), then environment variables
(``MAF_VAULT``, ``MAF_WORKSPACES``, ``MAF_BUDGET_USD``), then explicit keyword overrides (CLI flags).

Settings are checked when they load, before a run spends anything: unknown keys (nested blocks included), model IDs
without a price today, a ``claude_code_tmp_base`` too long for the sandbox's sockets and unscoped file-tool rules are
all errors. A missing ``python_executable`` is only a warning (``python_executable_warning``).

The defaults follow the installation: ``python_executable`` is the interpreter running maf, and ``workspaces/`` and
``inbox/`` live in the source checkout maf runs from (``source_checkout``; a clone at ``~/MultiAgent`` keeps
``~/MultiAgent/workspaces``), or under ``$XDG_DATA_HOME/maf`` (``~/.local/share/maf``) for an installed copy.
"""

from __future__ import annotations

import difflib
import logging
import math
import os
import re
import sys
import tomllib
from collections.abc import Iterable, Iterator, Mapping
from datetime import date
from pathlib import Path
from typing import Any, Literal, get_args

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

from maf.sandbox import DEFAULT_TMP_BASE, check_scoped_tools, check_tmp_base
from maf.types import STAGE_ORDER, AgentName, StageName, Tier, Usage

log = logging.getLogger(__name__)

ModelRole = Literal["chatgpt", "gemini", "claude", "claude_code"]
"""The four model slots. ``claude`` = Messages API, ``claude_code`` = headless ``claude -p``."""

MODEL_ROLES: tuple[ModelRole, ...] = get_args(ModelRole)


class StrictModel(BaseModel):
    """A config block that refuses unknown keys, naming them and the allowed ones: a typo such as
    ``output_limits: {claude_code_budget: 20}`` would otherwise be dropped and its default used without a word."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _known_keys(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            unknown = sorted(str(key) for key in data if key not in cls.model_fields)
            if unknown:
                raise ValueError(
                    f"unknown key(s): {', '.join(unknown)} (allowed keys: {', '.join(cls.model_fields)})"
                )
        return data


class ModelPins(StrictModel):
    """One model ID per role for a tier."""

    chatgpt: str
    gemini: str
    claude: str
    claude_code: str

    def for_role(self, role: ModelRole) -> str:
        return getattr(self, role)


TIER_MODELS: dict[Tier, ModelPins] = {
    "default": ModelPins(
        chatgpt="gpt-6-sol",
        gemini="gemini-3.8-flash",
        claude="claude-opus-5-5",
        claude_code="claude-opus-5-5",
    ),
    "max": ModelPins(
        chatgpt="gpt-6-astra",
        gemini="gemini-3.8-flash",
        claude="claude-fable-5-1",
        claude_code="claude-fable-5-1",
    ),
}


class ModelPrice(StrictModel):
    """USD per million tokens for one model, valid from ``effective_from`` until superseded.

    ``verified=False`` marks placeholder numbers that must be checked against the
    provider's official pricing page before any live run. Placeholders are
    deliberately conservative (high), so the budget check errs toward stopping early.
    """

    model_config = ConfigDict(frozen=True)

    model: str
    effective_from: date
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float
    cache_write_per_mtok: float | None = None
    """None means cache writes bill at the normal input rate."""
    search_query_usd: float = 0.0
    """Per billable grounding/search query (Gemini Google Search grounding)."""
    verified: bool = False
    source: str = ""

    def cost(self, usage: Usage) -> float:
        """Exact USD cost of ``usage`` under this price. See ``maf.types.Usage`` for token semantics.

        If an adapter reports cached + write tokens exceeding the input total, the uncached part
        is floored at zero rather than going negative.
        """
        cached = usage.cached_input_tokens
        written = usage.cache_write_tokens
        uncached = max(0, usage.input_tokens - cached - written)
        write_rate = self.input_per_mtok if self.cache_write_per_mtok is None else self.cache_write_per_mtok
        token_usd = math.fsum(
            (
                uncached * self.input_per_mtok,
                cached * self.cached_input_per_mtok,
                written * write_rate,
                usage.output_tokens * self.output_per_mtok,
            )
        )
        return token_usd / 1e6 + usage.search_queries * self.search_query_usd


# Grouped by model, chronological within a model. A lookup picks the latest entry whose
# effective_from <= the call date (order does not affect correctness).
PRICE_TABLE: tuple[ModelPrice, ...] = (
    # Anthropic prices come from the claude-api skill model table (cached 2026-06-24).
    ModelPrice(
        model="claude-opus-5-5",
        effective_from=date(2026, 1, 1),
        input_per_mtok=4.00,
        output_per_mtok=20.00,
        cached_input_per_mtok=0.20,
        cache_write_per_mtok=5.00,  # 1.25x input (5-minute TTL); verify
        verified=True,
        source="claude-api skill model table, 2026-06-24",
    ),
    ModelPrice(
        model="claude-fable-5-1",
        effective_from=date(2026, 1, 1),
        input_per_mtok=10.00,
        output_per_mtok=50.00,
        cached_input_per_mtok=0.25,
        cache_write_per_mtok=12.50,  # 1.25x input (5-minute TTL); verify
        verified=True,
        source="claude-api skill model table, 2026-06-24",
    ),
    # OpenAI/Gemini prices verified against official pricing pages on 2026-09-28 (research run).
    # Long-context rates (OpenAI >272K input) are not modeled; stages stay well below that.
    ModelPrice(
        model="gpt-6-sol",
        effective_from=date(2026, 1, 1),
        input_per_mtok=2.00,
        output_per_mtok=10.00,
        cached_input_per_mtok=0.20,
        cache_write_per_mtok=2.50,
        verified=True,
        source="https://openai.com/api/pricing, 2026-09-28",
    ),
    ModelPrice(
        model="gpt-6-astra",
        effective_from=date(2026, 1, 1),
        input_per_mtok=10.00,
        output_per_mtok=50.00,
        cached_input_per_mtok=1.00,
        cache_write_per_mtok=12.50,
        verified=True,
        source="https://openai.com/api/pricing, 2026-09-28",
    ),
    ModelPrice(
        model="gemini-3.8-flash",
        effective_from=date(2026, 1, 1),
        input_per_mtok=0.75,
        output_per_mtok=3.75,
        cached_input_per_mtok=0.075,
        search_query_usd=0.035,  # conservative; first 5,000 searches/month are free
        verified=True,
        source="https://ai.google.dev/gemini-api/docs/pricing, 2026-09-28",
    ),
    ModelPrice(
        model="gemini-3.8-flash",
        effective_from=date(2027, 1, 1),  # prices double on 2027-01-01 (DESIGN.md)
        input_per_mtok=1.50,
        output_per_mtok=7.50,
        cached_input_per_mtok=0.15,
        search_query_usd=0.035,
        verified=True,
        source="https://ai.google.dev/gemini-api/docs/pricing, 2026-09-28",
    ),
)


class UnknownModelPrice(KeyError):
    """No price entry covers (model, date). Calls to unpriced models are refused."""


def price_for(model: str, on: date, table: tuple[ModelPrice, ...] = PRICE_TABLE) -> ModelPrice:
    """Return the entry for ``model`` with the latest ``effective_from <= on``.

    Raises ``UnknownModelPrice`` if the model is missing or every entry starts after ``on``.
    """
    candidates = [p for p in table if p.model == model and p.effective_from <= on]
    if not candidates:
        known = any(p.model == model for p in table)
        reason = f"no price for {model!r} effective on {on.isoformat()}" if known else f"unknown model {model!r}"
        raise UnknownModelPrice(reason)
    return max(candidates, key=lambda p: p.effective_from)


class OutputLimits(StrictModel):
    """Default ``max_output_tokens`` per role. Worst-case budget checks use these. Unknown keys are refused."""

    chatgpt: int = 16_000
    gemini: int = 32_000
    claude: int = 64_000
    claude_code_budget_usd: float = 8.0
    """Per-invocation cap passed as ``--max-budget-usd`` (further clamped to the remaining run budget)."""


DEFAULT_CLAUDE_CODE_TOOLS: tuple[str, ...] = (
    "Read(./**)",
    "Edit(./**)",  # Edit rules cover every file-editing tool, Write included; Write(...) rules are ignored
    "Glob",
    "Grep",
    # Bash runs inside Claude Code's OS sandbox (bubblewrap): writes limited to the workspace, no network,
    # sensitive paths unreadable. Sandboxed commands are auto-allowed, so per-command patterns add nothing.
    "Bash",
)
"""Allowed tools for Claude Code in the workspace: edit, build, test, QEMU. No web tools. File tools are
scoped to the workspace (``./**`` is relative to Claude Code's cwd); a bare ``Read``/``Edit``/``Write`` rule
would match every path and is rejected when settings load (``maf.sandbox.check_scoped_tools``)."""


_ORIGIN_RE = re.compile(r"https?://(\[[0-9a-f:.]+\]|[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*)(:[0-9]{1,5})?")
"""A serialized web origin as browsers send it: lowercase scheme and host, optional port, nothing else."""

BASH_TIMEOUT_SHARE = 0.75
"""Default ``Settings.bash_timeout_s`` as a share of ``claude_code_timeout_s``."""

DEFAULT_MCP_MAX_BUDGET_USD = 5.0
"""Default ``Settings.mcp_max_budget_usd``: an MCP run gets at most this unless the config raises it."""
DEFAULT_MCP_DAILY_BUDGET_USD = 25.0
"""Default ``Settings.mcp_daily_budget_usd`` (rolling 24 hours, MCP runs only)."""

PACKAGE_DIR = Path(__file__).resolve().parent
"""This package's directory: ``<checkout>/src/maf`` when maf runs from a source checkout (an editable install
included), ``.../site-packages/maf`` when it is installed."""
PROJECT_NAME = "maf"


def source_checkout(package_dir: Path | None = None) -> Path | None:
    """The source checkout maf runs from: ``<root>`` when the package directory (default ``PACKAGE_DIR``) is
    ``<root>/src/maf`` and ``<root>/pyproject.toml`` names the project ``maf``; None for an installed copy."""
    package = PACKAGE_DIR if package_dir is None else package_dir
    if package.name != PROJECT_NAME or package.parent.name != "src":
        return None
    root = package.parent.parent
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8")).get("project")
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None
    name = project.get("name") if isinstance(project, dict) else None
    return root if isinstance(name, str) and re.sub(r"[-_.]+", "-", name).lower() == PROJECT_NAME else None


def xdg_data_home(environ: Mapping[str, str] | None = None) -> Path | None:
    """``$XDG_DATA_HOME`` when it is an absolute path (the XDG spec ignores a relative one), else None."""
    environ = os.environ if environ is None else environ
    xdg = environ.get("XDG_DATA_HOME", "").strip()
    return Path(xdg) if xdg and Path(xdg).is_absolute() else None


def data_home(environ: Mapping[str, str] | None = None) -> Path:
    """``$XDG_DATA_HOME/maf`` (``xdg_data_home``), else ``~/.local/share/maf``. ``maf chatgpt setup`` copies an
    absolute ``XDG_DATA_HOME`` into the maf-mcp unit when maf has no source checkout
    (``maf.chatgpt.settings_sources``), since the systemd user manager does not have the shell's value."""
    return (xdg_data_home(environ) or Path.home() / ".local" / "share") / PROJECT_NAME


def default_data_root() -> Path:
    """Parent of the default ``workspaces/`` and ``inbox/``: the source checkout maf runs from (``source_checkout``),
    else ``data_home()``. A clone at ``~/MultiAgent`` run from its venv keeps the original ``~/MultiAgent/workspaces``
    and ``~/MultiAgent/inbox``."""
    return source_checkout() or data_home()


def default_python_executable() -> Path:
    """The interpreter running maf (``sys.executable``, not resolved, so a venv's ``bin/`` stays its parent)."""
    return Path(sys.executable) if sys.executable else Path("/usr/bin/python3")


class Settings(BaseModel):
    """Fully resolved run-independent configuration."""

    model_config = ConfigDict(extra="forbid")

    vault_path: Path = Field(default_factory=lambda: Path.home() / "Obsidian" / "MultiAgent")
    workspaces_path: Path = Field(default_factory=lambda: default_data_root() / "workspaces")
    """Build trees, one per run. Default ``<checkout>/workspaces`` when maf runs from a source checkout, else
    ``$XDG_DATA_HOME/maf/workspaces`` (``default_data_root``). A run's workspace is ``<workspaces_path>/<run_id>``, so
    moving this setting moves where existing runs are looked for."""
    python_executable: Path = Field(default_factory=default_python_executable)
    """Interpreter Claude Code should use for simulations (numpy/scipy/matplotlib live here): its ``bin/`` goes first on
    Claude Code's ``PATH``. Default: the interpreter running maf, so the venv maf is installed in. A missing one is
    logged as a warning (``python_executable_warning``), not an error."""
    claude_executable: Path = Field(default_factory=lambda: Path.home() / ".local" / "bin" / "claude")

    budget_usd: float = Field(default=25.0, gt=0, allow_inf_nan=False)
    tier: Tier = "default"
    review: bool = False
    max_crosscheck_loops: int = Field(default=2, ge=0)

    stage_model_overrides: dict[StageName, dict[ModelRole, str]] = Field(default_factory=dict)
    """e.g. ``{"crosscheck": {"chatgpt": "gpt-6-astra"}}``. Beats the tier pin."""

    output_limits: OutputLimits = Field(default_factory=OutputLimits)
    claude_code_tools: tuple[str, ...] = DEFAULT_CLAUDE_CODE_TOOLS
    claude_code_timeout_s: float = 5400.0
    """Wall-clock limit of one Claude Code session; the CLI is killed after it and the session is charged its worst
    case. An execution or fix session that times out gets one continuation session in the same workspace
    (``maf.stages.base.work_session``); a second timeout fails the run. 90 minutes: the thesis execution of 2026-09-29
    was killed at 60 minutes with its work nearly done."""
    claude_code_min_session_usd: float = Field(default=3.0, ge=0, allow_inf_nan=False)
    """Smallest useful ``--max-budget-usd`` of a Claude Code work session (execution, fix pass, continuation). When
    the run's remaining budget (minus one turn of headroom) would clamp a session below it (or below the session's own
    budget, or ``claude_code_min_session_share`` of the run's budget, if either is smaller), the session is not started:
    an execution stops the run ``budget_exceeded``, naming the budget to resume with, and a fix pass is skipped. The
    sandbox preflight and the clean room keep their own minima."""
    claude_code_min_session_share: float = Field(default=0.25, gt=0, le=1, allow_inf_nan=False)
    """The minimum above never exceeds this share of the run's budget (``maf.stages.base.session_floor``), so a small
    run still gets its sessions: the $5 MCP default leaves an Opus execution session about $2.50 after ingestion,
    strategy, the preflight and one turn of headroom, against a minimum of $1.25. The minimum reaches the full $3 from
    a $12 run on."""
    claude_code_bash_timeout_s: float | None = Field(default=None, gt=0)
    """Longest timeout of one Bash command in a Claude Code session (``BASH_MAX_TIMEOUT_MS``; the CLI's own cap is 10
    minutes), so a long simulation, QEMU battery or clean-room reproduction is not killed before it finishes. None
    means 75 % of ``claude_code_timeout_s`` (``bash_timeout_s``), leaving the session time to read the log and report;
    a value must stay below ``claude_code_timeout_s``."""
    claude_code_turn_context_tokens: int = Field(default=200_000, gt=0)
    claude_code_turn_output_tokens: int = Field(default=64_000, gt=0)
    """Size of one Claude Code model turn. The CLI checks ``--max-budget-usd`` between turns, so the price of
    one such turn is held back from the clamp to keep the run cap hard."""
    claude_code_tmp_base: Path = DEFAULT_TMP_BASE
    """Parent of Claude Code's private ``TMPDIR`` (``<base>/maf-<12 random hex>``, one per provider instance). Sockets
    live under ``TMPDIR``, and Claude Code gives sandboxed commands ``TMPDIR=<TMPDIR>/claude-<uid>``, which must fit in
    44 bytes. So the base may be at most 15 bytes for a 4-digit uid (``maf.sandbox.check_tmp_base`` when settings load,
    ``max_tmpdir_bytes`` again before each call)."""
    claude_code_preflight_budget_usd: float | None = Field(default=None, gt=0)
    """``--max-budget-usd`` of the sandbox preflight that execution runs before the first code/mixed-mode call. None
    scales it with the model's price (``maf.providers.claude_code.preflight_budget_usd``: about $0.44 on
    claude-opus-5-5, $1.09 on claude-fable-5-1); a flat $0.15 is less than Fable's first turn."""
    provider_timeout_s: float = 600.0

    freertos_path: Path | None = Field(
        default_factory=lambda: Path.home() / ".local" / "share" / "maf" / "FreeRTOS-Kernel"
    )
    """Local FreeRTOS-Kernel clone, copied into ``<workspace>/FreeRTOS-Kernel`` before code-mode execution
    (Claude Code has no network). None or a missing directory means no kernel is provisioned."""

    export_exclude: tuple[str, ...] = ()
    """Extra case-sensitive ``fnmatch`` patterns of workspace paths that are neither exported to ``deliverables/`` nor
    linted nor audited. They add to the built-in list, ``maf.vault.DEFAULT_EXPORT_EXCLUDES`` (pipeline state, version
    control, the kernel, ``inputs/``, build output, caches, virtualenvs), which always applies: setting this never
    re-exports anything (``export_include`` does that). A pattern matches any component of the workspace-relative
    POSIX path or a leading part of it (``maf.lint.excluded``): ``.git`` and ``*.pyc`` match at any depth, ``build/tmp``
    only at the workspace root. Empty, absolute and ``!`` patterns are refused."""
    export_include: tuple[str, ...] = ()
    """Patterns (same form) of workspace paths to ship although an exclude pattern matches them, such as a hand-written
    ``build/toolchain.cmake`` (``build/*.cmake``) or ``build/package/*``. Applied after ``export_exclude`` and the
    defaults, like a gitignore's ``!`` lines, but never to pipeline state, the kernel, ``inputs/`` or version control
    (``maf.vault.PROTECTED_EXPORT_EXCLUDES``). As in a gitignore, nothing below a directory excluded as a whole
    (``node_modules``) comes back unless the pattern names a path in it."""
    export_max_mb: float = Field(default=200.0, gt=0)
    """Cap on the exported deliverable tree, in MiB (``export_max_bytes``)."""
    cleanroom_budget_usd: float = Field(default=1.5, gt=0)
    """``--max-budget-usd`` of the clean-room check, which rebuilds the exported deliverables in a fresh copy with the
    README's reproduction command."""
    source_audit: bool = True
    """Have Gemini (web search) verify the references cited by document deliverables. Strategy then requires a hard
    acceptance criterion that every reference passes the audit."""
    source_audit_max_refs: int = Field(default=60, gt=0)
    """Most references one source audit checks; the rest are reported as unaudited."""

    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8765
    """``maf serve`` listens on ``http://<mcp_host>:<mcp_port>/mcp``. Under ``maf serve --uds`` (the systemd unit) it
    listens on the Unix socket instead, and these only name the Host header clients must send."""
    mcp_inbox: Path | None = Field(default_factory=lambda: default_data_root() / "inbox")
    """``start_run`` over MCP only accepts input files inside this directory. None means no files over MCP. The default
    is ``<checkout>/inbox`` when maf runs from a source checkout, where ``.gitignore`` keeps it out of commits, else
    ``$XDG_DATA_HOME/maf/inbox`` (``default_data_root``); ``maf chatgpt setup`` creates it 0700."""
    mcp_max_budget_usd: float | None = Field(default=DEFAULT_MCP_MAX_BUDGET_USD, gt=0, allow_inf_nan=False)
    """Highest ``budget_usd`` an MCP client may request, and the budget of a ``start_run`` that names none (unless
    ``budget_usd`` is lower). ``null`` means ``budget_usd``. Only the CLI can go higher."""
    mcp_max_pending_runs: int = Field(default=2, ge=1)
    """Most MCP runs one ``maf serve`` holds at once, queued or running; ``start_run`` refuses more. Bounds what a
    remembered ChatGPT approval (or a prompt injection riding on it) can queue."""
    mcp_daily_budget_usd: float = Field(default=DEFAULT_MCP_DAILY_BUDGET_USD, gt=0, allow_inf_nan=False)
    """Cap on MCP spend over any rolling 24 hours: ``start_run`` refuses a run whose budget, added to what the MCP
    runs created in the last 24 h committed (the budget of a queued or running run, the actual spend of a finished
    or stopped one; ``maf.mcp_server.mcp_committed_usd``), would exceed it. CLI runs do not count."""
    mcp_allowed_origins: tuple[str, ...] = ()
    """Origin header values ``maf serve`` accepts (exact ``scheme://host[:port]``, no wildcards). Empty by default:
    tunnel-client and other non-browser clients send no Origin, and any browser page is refused. If the tunnel ever
    forwards one (the journal shows ``Invalid Origin header: <origin>``), add exactly that origin, such as
    ``https://chatgpt.com``; DNS-rebinding protection stays on either way."""

    @model_validator(mode="after")
    def _workspaces_outside_vault(self) -> Settings:
        check_workspaces_outside_vault(self.vault_path, self.workspaces_path)
        return self

    @model_validator(mode="after")
    def _models_priced(self) -> Settings:
        """Every model these settings can call has a price today (``price_for``), or nothing runs: an unpriced model
        would otherwise fail its stage after earlier stages were paid for."""
        today = date.today()
        problems: list[str] = []
        for where, model in self.configured_models():
            try:
                price_for(model, today)
            except UnknownModelPrice as exc:
                problems.append(f"{where}: {exc.args[0]}")
        if problems:
            known = sorted({p.model for p in PRICE_TABLE if p.effective_from <= today})
            raise ValueError(f"{'; '.join(problems)}. Known model IDs on {today.isoformat()}: {', '.join(known)}")
        return self

    @field_validator("claude_code_tmp_base")
    @classmethod
    def _short_absolute_tmp_base(cls, value: Path) -> Path:
        check_tmp_base(value)
        return value

    @field_validator("claude_code_tools")
    @classmethod
    def _scoped_file_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        check_scoped_tools(value)
        return value

    @field_validator("mcp_allowed_origins")
    @classmethod
    def _exact_origins(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for origin in value:
            if not _ORIGIN_RE.fullmatch(origin):
                raise ValueError(
                    f"mcp_allowed_origins entries must be exact origins like 'https://chatgpt.com' (scheme://host[:port], "
                    f"lowercase, no path or wildcard), got {origin!r}"
                )
        return value

    @field_validator("export_exclude", "export_include")
    @classmethod
    def _relative_patterns(cls, value: tuple[str, ...], info: ValidationInfo) -> tuple[str, ...]:
        patterns = tuple(p.strip() for p in value)
        for pattern in patterns:
            if not pattern or pattern.startswith(("/", "!")):
                raise ValueError(
                    f"{info.field_name} patterns must be non-empty, workspace-relative and without '!', got {pattern!r}"
                )
        return patterns

    @model_validator(mode="after")
    def _bash_timeout_below_session_timeout(self) -> Settings:
        bash = self.claude_code_bash_timeout_s
        if bash is not None and bash >= self.claude_code_timeout_s:
            raise ValueError(
                f"claude_code_bash_timeout_s ({self.claude_code_bash_timeout_s:g}) must be below claude_code_timeout_s "
                f"({self.claude_code_timeout_s:g}), or the session is killed before its command times out"
            )
        return self

    @property
    def bash_timeout_s(self) -> float:
        """``claude_code_bash_timeout_s``, or 75 % of ``claude_code_timeout_s`` when unset."""
        if self.claude_code_bash_timeout_s is not None:
            return self.claude_code_bash_timeout_s
        return BASH_TIMEOUT_SHARE * self.claude_code_timeout_s

    @property
    def export_max_bytes(self) -> int:
        return int(self.export_max_mb * 1024 * 1024)

    @property
    def mcp_budget_ceiling_usd(self) -> float:
        return self.budget_usd if self.mcp_max_budget_usd is None else self.mcp_max_budget_usd

    def model_for(self, role: ModelRole, stage: StageName | None = None) -> str:
        """Model ID for ``role``: per-stage override, else the tier pin."""
        if stage is not None:
            override = self.stage_model_overrides.get(stage, {}).get(role)
            if override:
                return override
        return TIER_MODELS[self.tier].for_role(role)

    def configured_models(self) -> Iterator[tuple[str, str]]:
        """``(where, model)`` for every model ID ``model_for`` can return: the selected tier's pins
        (``tier default: claude``), then each non-empty ``stage_model_overrides`` entry
        (``stage_model_overrides.final.claude``)."""
        pins = TIER_MODELS[self.tier]
        for role in MODEL_ROLES:
            yield f"tier {self.tier}: {role}", pins.for_role(role)
        for stage in STAGE_ORDER:
            for role, model in self.stage_model_overrides.get(stage, {}).items():
                if model:
                    yield f"stage_model_overrides.{stage}.{role}", model


def python_executable_warning(settings: Settings) -> str | None:
    """Why Claude Code would run without the configured interpreter, or None when ``python_executable`` exists.
    ``load_settings`` logs it and the CLI prints it (``maf: warning: ...``, ``maf run`` included): a missing venv is not
    an error (help, tests and prose runs need none), but code runs would silently get the system python and its
    packages."""
    python = settings.python_executable
    if python.exists():
        return None
    return (
        f"python_executable {python} does not exist, so Claude Code runs without that venv first on its PATH "
        "(simulations get the system python3 and its packages); set python_executable in the config to the venv's "
        "python, or recreate the venv"
    )


def check_workspaces_outside_vault(vault_path: Path, workspaces_path: Path) -> None:
    """``ValueError`` if the workspaces root is the vault or inside it: build trees and simulation data
    must stay out of Obsidian's index (DESIGN.md)."""
    vault = vault_path.expanduser().resolve()
    workspaces = workspaces_path.expanduser().resolve()
    if workspaces == vault or workspaces.is_relative_to(vault):
        raise ValueError(f"workspaces_path {workspaces} must be outside the vault {vault}")


def agent_for_role(role: ModelRole) -> AgentName:
    """Map a model slot to the agent its spend is attributed to (``claude_code`` -> ``claude``)."""
    return "claude" if role == "claude_code" else role


def default_config_path() -> Path:
    """``$MAF_CONFIG`` if set, else ``~/.config/maf/config.yaml``."""
    env = os.environ.get("MAF_CONFIG", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".config" / "maf" / "config.yaml"


_PATH_FIELDS = (
    "vault_path",
    "workspaces_path",
    "python_executable",
    "claude_executable",
    "claude_code_tmp_base",
    "freertos_path",
    "mcp_inbox",
)


def describe_unknown_keys(unknown: list[str], known: Iterable[str]) -> str:
    """``unknown`` joined by ``, ``, each with the closest ``known`` key when one is close enough
    (``claude_code_timout_s (did you mean claude_code_timeout_s?)``): the top level has too many keys to list them all,
    as the nested blocks' message does (``StrictModel``)."""
    known = list(known)
    described = []
    for key in unknown:
        close = difflib.get_close_matches(key, known, n=1)
        described.append(f"{key} (did you mean {close[0]}?)" if close else key)
    return ", ".join(described)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Parse the config file. Missing -> ``{}``; unreadable, malformed or non-mapping -> ``ValueError``, and so are
    unknown top-level keys (``describe_unknown_keys`` suggests the closest known key for each)."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read config file {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"malformed YAML in config file {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a mapping, got {type(data).__name__}")
    unknown = sorted(str(k) for k in data if k not in Settings.model_fields)
    if unknown:
        raise ValueError(
            f"unknown key(s) in config file {path}: {describe_unknown_keys(unknown, Settings.model_fields)}"
        )
    return data


def _env_layer() -> dict[str, Any]:
    layer: dict[str, Any] = {}
    if vault := os.environ.get("MAF_VAULT", "").strip():
        layer["vault_path"] = vault
    if workspaces := os.environ.get("MAF_WORKSPACES", "").strip():
        layer["workspaces_path"] = workspaces
    if budget := os.environ.get("MAF_BUDGET_USD", "").strip():
        try:
            layer["budget_usd"] = float(budget)
        except ValueError as exc:
            raise ValueError(f"MAF_BUDGET_USD must be a number, got {budget!r}") from exc
    return layer


def load_settings(config_path: Path | None = None, **overrides: object) -> Settings:
    """Build ``Settings`` from defaults, the optional YAML file, env vars, then ``overrides``.

    ``overrides`` with value ``None`` are ignored, so CLI code can pass unset flags straight through.
    A missing config file is not an error; a malformed or invalid one raises ``ValueError`` (unknown keys, nested ones
    included; unpriced models; a too-long ``claude_code_tmp_base``; unscoped file-tool rules). A missing
    ``python_executable`` is logged as a warning (``python_executable_warning``).
    Unknown override names raise ``TypeError``. Path values have ``~`` expanded.
    """
    unknown = sorted(k for k in overrides if k not in Settings.model_fields)
    if unknown:
        raise TypeError(f"unknown settings override(s): {', '.join(unknown)}")

    path = config_path if config_path is not None else default_config_path()
    data = _read_yaml(Path(path).expanduser())
    data.update(_env_layer())
    data.update({k: v for k, v in overrides.items() if v is not None})
    for key in _PATH_FIELDS:
        if isinstance(data.get(key), (str, Path)):
            data[key] = Path(data[key]).expanduser()
    try:
        settings = Settings.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"invalid settings (from {path}, MAF_* env vars and overrides): {exc}") from exc
    if (warning := python_executable_warning(settings)) is not None:
        log.warning("%s", warning)
    return settings
