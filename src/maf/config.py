"""Settings, model pins, tiers and the dated price table.

Owner: core (config + ledger).

Settings resolve in this order (later wins): built-in defaults, then
``~/.config/maf/config.yaml`` (or ``$MAF_CONFIG``), then environment variables
(``MAF_VAULT``, ``MAF_WORKSPACES``, ``MAF_BUDGET_USD``), then explicit keyword overrides (CLI flags).
"""

from __future__ import annotations

import math
import os
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from maf.types import AgentName, StageName, Tier, Usage

ModelRole = Literal["chatgpt", "gemini", "claude", "claude_code"]
"""The four model slots. ``claude`` = Messages API, ``claude_code`` = headless ``claude -p``."""


class ModelPins(BaseModel):
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


class ModelPrice(BaseModel):
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


class OutputLimits(BaseModel):
    """Default ``max_output_tokens`` per role. Worst-case budget checks use these."""

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
would match every path and is rejected by ``ClaudeCodeProvider``."""


class Settings(BaseModel):
    """Fully resolved run-independent configuration."""

    vault_path: Path = Field(default_factory=lambda: Path.home() / "Obsidian" / "MultiAgent")
    workspaces_path: Path = Field(default_factory=lambda: Path.home() / "MultiAgent" / "workspaces")
    python_executable: Path = Field(default_factory=lambda: Path.home() / "MultiAgent" / ".venv" / "bin" / "python")
    """Interpreter Claude Code should use for simulations (numpy/scipy/matplotlib live here)."""
    claude_executable: Path = Field(default_factory=lambda: Path.home() / ".local" / "bin" / "claude")

    budget_usd: float = Field(default=25.0, gt=0)
    tier: Tier = "default"
    review: bool = False
    max_crosscheck_loops: int = Field(default=2, ge=0)

    stage_model_overrides: dict[StageName, dict[ModelRole, str]] = Field(default_factory=dict)
    """e.g. ``{"crosscheck": {"chatgpt": "gpt-6-astra"}}``. Beats the tier pin."""

    output_limits: OutputLimits = Field(default_factory=OutputLimits)
    claude_code_tools: tuple[str, ...] = DEFAULT_CLAUDE_CODE_TOOLS
    claude_code_timeout_s: float = 3600.0
    claude_code_turn_context_tokens: int = Field(default=200_000, gt=0)
    claude_code_turn_output_tokens: int = Field(default=64_000, gt=0)
    """Size of one Claude Code model turn. The CLI checks ``--max-budget-usd`` between turns, so the price of
    one such turn is held back from the clamp to keep the run cap hard."""
    provider_timeout_s: float = 600.0

    freertos_path: Path | None = Field(
        default_factory=lambda: Path.home() / ".local" / "share" / "maf" / "FreeRTOS-Kernel"
    )
    """Local FreeRTOS-Kernel clone, copied into ``<workspace>/FreeRTOS-Kernel`` before code-mode execution
    (Claude Code has no network). None or a missing directory means no kernel is provisioned."""

    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8765
    mcp_inbox: Path | None = Field(default_factory=lambda: Path.home() / "MultiAgent" / "inbox")
    """``start_run`` over MCP only accepts input files inside this directory. None means no files over MCP."""
    mcp_max_budget_usd: float | None = Field(default=None, gt=0)
    """Highest ``budget_usd`` an MCP client may request; None means ``budget_usd``. Only the CLI can go higher."""

    @model_validator(mode="after")
    def _workspaces_outside_vault(self) -> Settings:
        check_workspaces_outside_vault(self.vault_path, self.workspaces_path)
        return self

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


_PATH_FIELDS = ("vault_path", "workspaces_path", "python_executable", "claude_executable", "freertos_path", "mcp_inbox")


def _read_yaml(path: Path) -> dict[str, Any]:
    """Parse the config file. Missing -> ``{}``; unreadable, malformed or non-mapping -> ``ValueError``."""
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
        raise ValueError(f"unknown key(s) in config file {path}: {', '.join(unknown)}")
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
    A missing config file is not an error; a malformed one raises ``ValueError``.
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
        return Settings.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"invalid settings (from {path}, MAF_* env vars and overrides): {exc}") from exc
