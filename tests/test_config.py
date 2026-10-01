"""Tests for maf.config: pricing, price lookup, model selection and settings layering."""

from __future__ import annotations

import logging
import os
import sys
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from maf import config, lint, sandbox
from maf.config import (
    PRICE_TABLE,
    TIER_MODELS,
    ModelPins,
    ModelPrice,
    OutputLimits,
    Settings,
    UnknownModelPrice,
    agent_for_role,
    data_home,
    default_config_path,
    load_settings,
    price_for,
    python_executable_warning,
    source_checkout,
)
from maf.stages.base import export_excludes
from maf.types import Usage
from maf.vault import DEFAULT_EXPORT_EXCLUDES, PROTECTED_EXPORT_EXCLUDES, Vault

FIXTURES = Path(__file__).parent / "fixtures" / "config"


def _price(**kw: object) -> ModelPrice:
    base: dict[str, object] = dict(
        model="m",
        effective_from=date(2026, 1, 1),
        input_per_mtok=2.0,
        output_per_mtok=10.0,
        cached_input_per_mtok=0.5,
    )
    base.update(kw)
    return ModelPrice(**base)  # type: ignore[arg-type]


# -- ModelPrice.cost -------------------------------------------------------------------------


def test_cost_plain_tokens() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=500_000)
    assert _price().cost(usage) == pytest.approx(2.0 + 5.0)


def test_cost_zero_usage_is_zero() -> None:
    assert _price(search_query_usd=0.1).cost(Usage()) == 0.0


def test_cost_cached_and_write_are_subsets_of_input() -> None:
    usage = Usage(input_tokens=1_000_000, cached_input_tokens=400_000, cache_write_tokens=100_000, output_tokens=0)
    price = _price(cache_write_per_mtok=2.5)
    expected = (500_000 * 2.0 + 400_000 * 0.5 + 100_000 * 2.5) / 1e6
    assert price.cost(usage) == pytest.approx(expected)


def test_cost_write_defaults_to_input_rate() -> None:
    usage = Usage(input_tokens=100_000, cache_write_tokens=100_000)
    assert _price(cache_write_per_mtok=None).cost(usage) == pytest.approx(0.2)


def test_cost_reasoning_tokens_are_not_double_billed() -> None:
    plain = Usage(output_tokens=1000)
    with_reasoning = Usage(output_tokens=1000, reasoning_tokens=800)
    assert _price().cost(plain) == _price().cost(with_reasoning)


def test_cost_search_queries() -> None:
    usage = Usage(input_tokens=0, output_tokens=0, search_queries=6)
    assert _price(search_query_usd=0.035).cost(usage) == pytest.approx(0.21)


def test_cost_inconsistent_usage_never_negative() -> None:
    usage = Usage(input_tokens=10, cached_input_tokens=50)
    assert _price().cost(usage) == pytest.approx(50 * 0.5 / 1e6)


def test_cost_anthropic_opus_realistic() -> None:
    price = price_for("claude-opus-5-5", date(2026, 9, 28))
    usage = Usage(input_tokens=30_000, cached_input_tokens=20_000, cache_write_tokens=5_000, output_tokens=4_000)
    expected = (5_000 * 4.0 + 20_000 * 0.20 + 5_000 * 5.0 + 4_000 * 20.0) / 1e6
    assert price.cost(usage) == pytest.approx(expected)


# -- price_for -------------------------------------------------------------------------------


def test_price_for_gemini_doubles_on_2027_01_01() -> None:
    before = price_for("gemini-3.8-flash", date(2026, 12, 31))
    on = price_for("gemini-3.8-flash", date(2027, 1, 1))
    later = price_for("gemini-3.8-flash", date(2028, 6, 1))
    assert on.input_per_mtok == pytest.approx(2 * before.input_per_mtok)
    assert on.output_per_mtok == pytest.approx(2 * before.output_per_mtok)
    assert on.cached_input_per_mtok == pytest.approx(2 * before.cached_input_per_mtok)
    assert later == on
    usage = Usage(input_tokens=100_000, output_tokens=10_000)
    assert on.cost(usage) == pytest.approx(2 * before.cost(usage))


def test_price_for_unknown_model() -> None:
    with pytest.raises(UnknownModelPrice, match="unknown model"):
        price_for("gpt-2", date(2026, 9, 28))


def test_price_for_before_first_entry() -> None:
    with pytest.raises(UnknownModelPrice, match="effective on 2025-12-31"):
        price_for("claude-opus-5-5", date(2025, 12, 31))


def test_unknown_model_price_is_key_error() -> None:
    assert issubclass(UnknownModelPrice, KeyError)


def test_price_for_uses_latest_effective_entry_regardless_of_order() -> None:
    table = (
        _price(effective_from=date(2027, 1, 1), input_per_mtok=3.0),
        _price(effective_from=date(2026, 1, 1), input_per_mtok=1.0),
        _price(effective_from=date(2026, 6, 1), input_per_mtok=2.0),
    )
    assert price_for("m", date(2026, 5, 31), table).input_per_mtok == 1.0
    assert price_for("m", date(2026, 6, 1), table).input_per_mtok == 2.0
    assert price_for("m", date(2030, 1, 1), table).input_per_mtok == 3.0


def test_every_pinned_model_is_priced() -> None:
    for pins in TIER_MODELS.values():
        for role in ("chatgpt", "gemini", "claude", "claude_code"):
            price_for(pins.for_role(role), date(2026, 9, 28))  # type: ignore[arg-type]


def test_price_table_chronological_per_model_and_unique() -> None:
    keys = [(p.model, p.effective_from) for p in PRICE_TABLE]
    assert len(keys) == len(set(keys))
    for model in {m for m, _ in keys}:
        dates = [d for m, d in keys if m == model]
        assert dates == sorted(dates)


def test_all_prices_are_verified_with_a_source() -> None:
    assert all(p.verified and p.source for p in PRICE_TABLE)


# -- model selection -------------------------------------------------------------------------


def test_model_for_tier_pins() -> None:
    assert Settings().model_for("chatgpt") == "gpt-6-sol"
    assert Settings().model_for("claude_code", "execution") == "claude-opus-5-5"
    max_tier = Settings(tier="max")
    assert max_tier.model_for("chatgpt") == "gpt-6-astra"
    assert max_tier.model_for("claude") == "claude-fable-5-1"
    assert max_tier.model_for("gemini") == "gemini-3.8-flash"


def test_model_for_stage_override_beats_tier() -> None:
    s = Settings(stage_model_overrides={"crosscheck": {"chatgpt": "gpt-6-astra"}})
    assert s.model_for("chatgpt", "crosscheck") == "gpt-6-astra"
    assert s.model_for("chatgpt", "strategy") == "gpt-6-sol"
    assert s.model_for("chatgpt") == "gpt-6-sol"
    assert s.model_for("claude", "crosscheck") == "claude-opus-5-5"


def test_agent_for_role() -> None:
    assert agent_for_role("claude_code") == "claude"
    assert agent_for_role("gemini") == "gemini"


# -- default_config_path ---------------------------------------------------------------------


def test_default_config_path_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_config_path() == tmp_path / ".config" / "maf" / "config.yaml"


def test_default_config_path_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MAF_CONFIG", str(tmp_path / "c.yaml"))
    assert default_config_path() == tmp_path / "c.yaml"


def test_default_config_path_env_expands_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MAF_CONFIG", "~/x.yaml")
    assert default_config_path() == tmp_path / "x.yaml"


def test_default_config_path_blank_env_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MAF_CONFIG", "  ")
    assert default_config_path() == tmp_path / ".config" / "maf" / "config.yaml"


# -- load_settings ---------------------------------------------------------------------------


def test_load_missing_file_gives_defaults(tmp_path: Path) -> None:
    s = load_settings(tmp_path / "nope.yaml")
    assert s == Settings()


def test_load_uses_default_path_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("budget_usd: 7\n", encoding="utf-8")
    monkeypatch.setenv("MAF_CONFIG", str(cfg))
    assert load_settings().budget_usd == 7


def test_load_full_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    s = load_settings(FIXTURES / "full.yaml")
    assert s.vault_path == tmp_path / "Vaults" / "MAF"
    assert s.workspaces_path == Path("/srv/maf/workspaces")
    assert s.budget_usd == 40
    assert s.tier == "max"
    assert s.review is True
    assert s.max_crosscheck_loops == 1
    assert s.output_limits.chatgpt == 8000
    assert s.output_limits.claude == 64_000  # unspecified nested fields keep defaults
    assert s.output_limits.claude_code_budget_usd == 5.0
    assert s.provider_timeout_s == 120
    assert s.mcp_port == 9000
    assert s.model_for("chatgpt", "crosscheck") == "gpt-6-sol"
    assert s.model_for("chatgpt", "strategy") == "gpt-6-astra"
    assert s.model_for("claude", "final") == "claude-opus-5-5"
    assert s.model_for("claude", "execution") == "claude-fable-5-1"


def test_load_empty_file_gives_defaults(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("# nothing yet\n", encoding="utf-8")
    assert load_settings(cfg) == Settings()


def test_env_beats_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MAF_VAULT", str(tmp_path / "envvault"))
    monkeypatch.setenv("MAF_WORKSPACES", str(tmp_path / "envws"))
    monkeypatch.setenv("MAF_BUDGET_USD", "12.5")
    s = load_settings(FIXTURES / "full.yaml")
    assert s.vault_path == tmp_path / "envvault"
    assert s.workspaces_path == tmp_path / "envws"
    assert s.budget_usd == 12.5
    assert s.tier == "max"  # untouched YAML value survives


def test_overrides_beat_env_and_none_is_ignored(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MAF_BUDGET_USD", "12.5")
    monkeypatch.setenv("MAF_VAULT", str(tmp_path / "envvault"))
    s = load_settings(
        FIXTURES / "full.yaml",
        budget_usd=3.0,
        vault_path=None,
        tier=None,
        workspaces_path=str(tmp_path / "cli"),
    )
    assert s.budget_usd == 3.0
    assert s.vault_path == tmp_path / "envvault"
    assert s.tier == "max"
    assert s.workspaces_path == tmp_path / "cli"


def test_override_false_is_not_ignored(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("review: true\n", encoding="utf-8")
    assert load_settings(cfg, review=False).review is False


def test_unknown_override_rejected(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="budget"):
        load_settings(tmp_path / "none.yaml", budget=5)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("budget_usd: [1, 2\n", "malformed YAML"),
        ("- a\n- b\n", "must contain a mapping"),
        ("budgt_usd: 5\n", "unknown key"),
        ("budget_usd: -1\n", "invalid settings"),
        ("tier: ultra\n", "invalid settings"),
        ("stage_model_overrides:\n  nosuchstage:\n    chatgpt: x\n", "invalid settings"),
        ("claude_code_tmp_base: tmp\n", "must be an absolute path"),
        ("claude_code_preflight_budget_usd: 0\n", "invalid settings"),
        ("export_exclude: [\"\"]\n", "workspace-relative"),
        ("export_exclude: [/etc]\n", "workspace-relative"),
        ("export_exclude: .git\n", "invalid settings"),
        ("export_exclude: ['!build/x']\n", "without '!'"),
        ("export_include: [/abs]\n", "export_include patterns must be"),
        ("export_max_mb: 0\n", "invalid settings"),
        ("cleanroom_budget_usd: -1\n", "invalid settings"),
        ("source_audit_max_refs: 0\n", "invalid settings"),
        ("claude_code_bash_timeout_s: 5400\n", "must be below claude_code_timeout_s"),
    ],
)
def test_bad_config_raises_value_error(tmp_path: Path, text: str, match: str) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_settings(cfg)


def test_claude_code_sandbox_defaults() -> None:
    s = Settings()
    assert s.claude_code_tmp_base == Path("/tmp")
    assert s.claude_code_preflight_budget_usd is None  # scaled with the model's price by the provider


def test_claude_code_tmp_base_from_yaml_expands_user(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", "/h")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("claude_code_tmp_base: ~/t\nclaude_code_preflight_budget_usd: 0.4\n", encoding="utf-8")
    s = load_settings(cfg)
    assert s.claude_code_tmp_base == Path("/h/t")
    assert s.claude_code_preflight_budget_usd == 0.4
    assert load_settings(cfg, claude_code_tmp_base="/var/tmp").claude_code_tmp_base == Path("/var/tmp")


def test_config_path_is_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read"):
        load_settings(tmp_path)


def test_bad_env_budget_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MAF_BUDGET_USD", "lots")
    with pytest.raises(ValueError, match="MAF_BUDGET_USD"):
        load_settings(tmp_path / "none.yaml")


def test_invalid_override_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="budget_usd"):
        load_settings(tmp_path / "none.yaml", budget_usd=0)


@pytest.mark.parametrize("field", ["budget_usd", "mcp_max_budget_usd"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_budgets_must_be_finite(tmp_path: Path, field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        load_settings(tmp_path / "none.yaml", **{field: value})


def test_deliverable_and_verification_defaults() -> None:
    s = Settings()
    assert s.export_exclude == ()  # extra patterns only: the built-in list always applies
    assert export_excludes(s) == DEFAULT_EXPORT_EXCLUDES
    assert {".maf", ".git", "FreeRTOS-Kernel", "__pycache__", "*.pyc", "inputs/*"} <= set(export_excludes(s))
    assert s.export_max_mb == 200 and s.export_max_bytes == 200 * 1024 * 1024
    assert s.cleanroom_budget_usd == 1.5
    assert s.source_audit is True and s.source_audit_max_refs == 60


def test_deliverable_settings_from_yaml_and_overrides(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "export_exclude: [' build ', '*.o']\nexport_max_mb: 0.5\ncleanroom_budget_usd: 3\n"
        "source_audit: false\nsource_audit_max_refs: 10\n",
        encoding="utf-8",
    )
    s = load_settings(cfg)
    assert s.export_exclude == ("build", "*.o")
    assert s.export_max_bytes == 524_288
    assert (s.cleanroom_budget_usd, s.source_audit, s.source_audit_max_refs) == (3.0, False, 10)
    overridden = load_settings(cfg, source_audit=True, export_exclude=())
    assert overridden.source_audit is True and overridden.export_exclude == ()
    assert load_settings(cfg, export_include=[" build/*.cmake "]).export_include == ("build/*.cmake",)
    assert Settings().export_include == ()


def test_bash_timeout_defaults_to_a_share_of_the_session_timeout() -> None:
    assert Settings().bash_timeout_s == 4050.0  # 75 % of the 90-minute session timeout
    assert Settings(claude_code_timeout_s=1200).bash_timeout_s == 900.0
    assert Settings(claude_code_bash_timeout_s=1500).bash_timeout_s == 1500.0


def test_mcp_allowed_origins_default_empty_and_exact() -> None:
    assert Settings().mcp_allowed_origins == ()
    ok = ("https://chatgpt.com", "http://localhost:6274", "http://[::1]:8080")
    assert Settings(mcp_allowed_origins=ok).mcp_allowed_origins == ok


@pytest.mark.parametrize(
    "origin", ["*", "https://chatgpt.com/", "https://*.chatgpt.com", "http://localhost:*", "HTTPS://chatgpt.com",
               "null", "chatgpt.com", "https://chatgpt.com/path", ""],
)
def test_mcp_allowed_origins_rejects_wildcards_paths_and_non_origins(origin: str) -> None:
    with pytest.raises(ValueError, match="mcp_allowed_origins"):
        Settings(mcp_allowed_origins=(origin,))


def test_mcp_spend_limits_default_low_and_validate() -> None:
    """An MCP run gets $5 at most unless the config raises it (not budget_usd's $25); at most two MCP runs wait or run
    at once; MCP runs may commit $25 per rolling 24 hours."""
    s = Settings()
    assert (s.mcp_max_budget_usd, s.mcp_budget_ceiling_usd) == (5.0, 5.0)
    assert (s.mcp_max_pending_runs, s.mcp_daily_budget_usd) == (2, 25.0)
    assert Settings(mcp_max_budget_usd=None).mcp_budget_ceiling_usd == s.budget_usd  # explicit null: budget_usd
    for bad in ({"mcp_max_pending_runs": 0}, {"mcp_daily_budget_usd": 0}, {"mcp_daily_budget_usd": float("inf")}):
        with pytest.raises(ValueError):
            Settings(**bad)


# -- early validation: nested keys, model prices, sandbox rules -------------------------------


def _load(tmp_path: Path, text: str) -> Settings:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(text, encoding="utf-8")
    return load_settings(cfg)


def test_nested_unknown_key_is_refused_naming_it_and_the_allowed_keys(tmp_path: Path) -> None:
    """A typo in a nested block used to be dropped silently, leaving the default budget in force."""
    with pytest.raises(ValueError, match="output_limits") as info:
        _load(tmp_path, "output_limits: {claude_code_budget: 20}\n")
    message = str(info.value)
    assert "unknown key(s): claude_code_budget" in message
    assert "allowed keys: chatgpt, gemini, claude, claude_code_budget_usd" in message
    assert _load(tmp_path, "output_limits: {claude_code_budget_usd: 20}\n").output_limits.claude_code_budget_usd == 20


def test_top_level_unknown_keys_are_refused_with_the_closest_known_key(tmp_path: Path) -> None:
    """The top level has too many keys to list (about 45), so each typo gets the nearest real key instead; a key like
    nothing known is still named."""
    with pytest.raises(ValueError) as info:
        _load(tmp_path, "claude_code_timout_s: 60\nfoo: 1\nworkspace_path: /x\n")
    message = str(info.value)
    assert message == (
        f"unknown key(s) in config file {tmp_path / 'c.yaml'}: claude_code_timout_s (did you mean "
        "claude_code_timeout_s?), foo, workspace_path (did you mean workspaces_path?)"
    )
    assert config.describe_unknown_keys(["reveiw", "budget"], Settings.model_fields) == (
        "reveiw (did you mean review?), budget (did you mean budget_usd?)"
    )
    assert config.describe_unknown_keys(["zzz"], ["a", "b"]) == "zzz"


@pytest.mark.parametrize(
    ("model", "data"),
    [
        (OutputLimits, {"chatgpt": 1, "claud": 2}),
        (ModelPins, {"chatgpt": "a", "gemini": "b", "claude": "c", "claude_code": "d", "codex": "e"}),
        (ModelPrice, {"model": "m", "effective_from": "2026-01-01", "input_per_mtok": 1, "output_per_mtok": 1,
                      "cached_input_per_mtok": 1, "cache_read_per_mtok": 1}),
    ],
)
def test_every_nested_config_model_forbids_unknown_keys(model: type[OutputLimits], data: dict[str, object]) -> None:
    unknown = (set(data) - set(model.model_fields)).pop()
    with pytest.raises(ValidationError, match=f"unknown key\\(s\\): {unknown} \\(allowed keys: "):
        model.model_validate(data)


def test_settings_refuse_unknown_fields_when_built_directly() -> None:
    with pytest.raises(ValidationError, match="budget"):
        Settings(budget=5)  # type: ignore[call-arg]


def test_unpriced_stage_override_fails_at_load_naming_the_model_and_the_known_ids(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stage_model_overrides.final.claude: unknown model 'claude-sonnet-9'") as info:
        _load(tmp_path, "stage_model_overrides: {final: {claude: claude-sonnet-9}}\n")
    known = sorted({p.model for p in PRICE_TABLE})
    assert f"Known model IDs on {date.today().isoformat()}: {', '.join(known)}" in str(info.value)


def test_every_unpriced_model_is_named_at_once(tmp_path: Path) -> None:
    with pytest.raises(ValueError) as info:
        _load(tmp_path, "stage_model_overrides:\n  crosscheck: {chatgpt: gpt-5}\n  final: {claude: claude-sonnet-9}\n")
    assert "crosscheck.chatgpt: unknown model 'gpt-5'" in str(info.value)
    assert "final.claude: unknown model 'claude-sonnet-9'" in str(info.value)


def test_priced_overrides_and_empty_entries_load(tmp_path: Path) -> None:
    s = _load(tmp_path, "stage_model_overrides:\n  crosscheck: {chatgpt: gpt-6-astra}\n  final: {claude: ''}\n")
    assert s.model_for("chatgpt", "crosscheck") == "gpt-6-astra"
    assert s.model_for("claude", "final") == "claude-opus-5-5"  # an empty override falls back to the tier pin
    assert ("stage_model_overrides.crosscheck.chatgpt", "gpt-6-astra") in list(s.configured_models())


def test_the_selected_tiers_pins_must_be_priced(monkeypatch: pytest.MonkeyPatch) -> None:
    pins = TIER_MODELS["max"].model_copy(update={"chatgpt": "gpt-9-unpriced"})
    monkeypatch.setitem(TIER_MODELS, "max", pins)
    assert Settings().tier == "default"  # only the selected tier is checked
    with pytest.raises(ValidationError, match="tier max: chatgpt: unknown model 'gpt-9-unpriced'"):
        Settings(tier="max")


def test_models_must_be_priced_on_todays_date(monkeypatch: pytest.MonkeyPatch) -> None:
    class Before2026(date):
        @classmethod
        def today(cls) -> Before2026:
            return cls(2025, 6, 1)

    monkeypatch.setattr(config, "date", Before2026)
    with pytest.raises(ValidationError, match="tier default: chatgpt: no price for 'gpt-6-sol' effective on 2025-06-01"):
        Settings()


def test_tmp_base_too_long_for_the_sandbox_sockets_fails_at_load(tmp_path: Path) -> None:
    longest = max_tmp_base_bytes()
    ok = "/" + "b" * (longest - 1)
    assert _load(tmp_path, f"claude_code_tmp_base: {ok}\n").claude_code_tmp_base == Path(ok)
    with pytest.raises(ValueError, match="claude_code_tmp_base .* is too long") as info:
        _load(tmp_path, f"claude_code_tmp_base: {ok}b\n")
    assert f"at most {longest} bytes" in str(info.value) and f"claude-{os.getuid()}" in str(info.value)
    with pytest.raises(ValueError, match="too long"):
        Settings(claude_code_tmp_base=Path("/" + "é" * (longest // 2 + 1)))  # few characters, too many bytes


def max_tmp_base_bytes() -> int:
    """Longest ``claude_code_tmp_base`` for this uid: ``max_tmpdir_bytes()`` minus ``/maf-<12 hex>``."""
    return sandbox.max_tmpdir_bytes() - len("/maf-") - 2 * sandbox.TMPDIR_RANDOM_BYTES


@pytest.mark.parametrize("extra", [-2, -1, 0, 1, 2])
def test_load_time_tmp_base_rule_matches_the_providers_check(extra: int) -> None:
    """One rule, two checkpoints: a base passes at load exactly when its TMPDIR passes ``check_tmpdir`` per call."""
    from maf.providers.base import ProviderError
    from maf.providers.claude_code import check_tmpdir, scratch_tmpdir

    base = Path("/" + "c" * (max_tmp_base_bytes() - 1 + extra))
    try:
        check_tmpdir(scratch_tmpdir(base))
        provider_ok = True
    except ProviderError:
        provider_ok = False
    try:
        Settings(claude_code_tmp_base=base)
        load_ok = True
    except ValidationError:
        load_ok = False
    assert load_ok == provider_ok == (extra <= 0)


def test_unscoped_file_tool_rules_fail_at_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="(?s)claude_code_tools.*need a path scope such as 'Edit\\(\\./\\*\\*\\)': Read"):
        _load(tmp_path, "claude_code_tools: [Read, Glob]\n")
    for bare in ("Edit", "Write", " Write "):
        with pytest.raises(ValidationError, match="path scope"):
            Settings(claude_code_tools=("Read(./**)", bare))
    assert _load(tmp_path, "claude_code_tools: ['Read(./**)', Bash]\n").claude_code_tools == ("Read(./**)", "Bash")


def test_config_and_provider_share_the_sandbox_rules() -> None:
    from maf.providers import claude_code

    assert claude_code.check_scoped_tools is sandbox.check_scoped_tools
    assert claude_code.max_tmpdir_bytes is sandbox.max_tmpdir_bytes
    assert claude_code.CLI_CHILD_TMPDIR_MAX_BYTES == sandbox.CLI_CHILD_TMPDIR_MAX_BYTES == 44


# -- python_executable --------------------------------------------------------------------------


def test_python_executable_defaults_to_the_running_interpreter() -> None:
    assert Settings().python_executable == Path(sys.executable)  # not resolved: a venv's bin/ stays its parent
    assert python_executable_warning(Settings()) is None


def test_missing_python_executable_warns_but_loads(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    missing = tmp_path / "gone" / "bin" / "python"
    with caplog.at_level(logging.WARNING, logger="maf.config"):
        s = _load(tmp_path, f"python_executable: {missing}\n")
    assert s.python_executable == missing
    warning = python_executable_warning(s)
    assert warning is not None and str(missing) in warning and "does not exist" in warning
    assert [r.getMessage() for r in caplog.records if r.name == "maf.config"] == [warning]


def test_existing_python_executable_does_not_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    with caplog.at_level(logging.WARNING, logger="maf.config"):
        _load(tmp_path, f"python_executable: {python}\n")
    assert not [r for r in caplog.records if r.name == "maf.config"]


# -- checkout-independent defaults ------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fake_checkout(root: Path, name: str = "maf") -> Path:
    package = root / "src" / "maf"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "0"\n', encoding="utf-8")
    return package


def test_defaults_follow_the_real_checkout() -> None:
    """Run from this checkout, workspaces/ and inbox/ stay in it: on the development machine that is
    ~/MultiAgent/workspaces and ~/MultiAgent/inbox, where existing runs' workspaces live."""
    assert config.PACKAGE_DIR == REPO_ROOT / "src" / "maf"
    assert source_checkout() == REPO_ROOT
    s = Settings()
    assert (s.workspaces_path, s.mcp_inbox) == (REPO_ROOT / "workspaces", REPO_ROOT / "inbox")
    if REPO_ROOT == Path.home() / "MultiAgent":  # the layout the earlier, hard-coded defaults assumed
        assert s.workspaces_path == Path.home() / "MultiAgent" / "workspaces"
        assert s.mcp_inbox == Path.home() / "MultiAgent" / "inbox"


def test_a_checkout_at_home_multiagent_keeps_the_old_defaults_and_run_workspaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Backward compatible: ``~/MultiAgent`` run from its venv gives the paths the old defaults hard-coded, so a run's
    workspace (``<workspaces_path>/<run_id>``, the path its run.md records) still resolves."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(config, "PACKAGE_DIR", _fake_checkout(tmp_path / "MultiAgent"))
    s = Settings()
    assert s.workspaces_path == Path.home() / "MultiAgent" / "workspaces"
    assert s.mcp_inbox == Path.home() / "MultiAgent" / "inbox"
    run_id = "2026-09-29-research-a-thesis"
    recorded = str(Path.home() / "MultiAgent" / "workspaces" / run_id)  # RunIndex.workspace of an existing run
    assert str(Vault(tmp_path / "vault", s.workspaces_path).paths(run_id).workspace) == recorded


def test_defaults_follow_a_checkout_anywhere(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "src-trees" / "MultiAgent"
    monkeypatch.setattr(config, "PACKAGE_DIR", _fake_checkout(root))
    s = load_settings(tmp_path / "none.yaml")
    assert (s.workspaces_path, s.mcp_inbox) == (root / "workspaces", root / "inbox")


def test_installed_copy_defaults_to_xdg_data_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package = tmp_path / "lib" / "python3.12" / "site-packages" / "maf"
    package.mkdir(parents=True)
    monkeypatch.setattr(config, "PACKAGE_DIR", package)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert source_checkout() is None
    s = Settings()
    assert (s.workspaces_path, s.mcp_inbox) == (tmp_path / "xdg" / "maf" / "workspaces", tmp_path / "xdg" / "maf" / "inbox")


@pytest.mark.parametrize("xdg", [None, "", "relative/data"])
def test_installed_copy_falls_back_to_local_share(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, xdg: str | None) -> None:
    """The XDG spec ignores an unset, empty or relative ``XDG_DATA_HOME``."""
    monkeypatch.setattr(config, "PACKAGE_DIR", tmp_path / "site-packages" / "maf")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    if xdg is None:
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    else:
        monkeypatch.setenv("XDG_DATA_HOME", xdg)
    assert data_home() == tmp_path / "home" / ".local" / "share" / "maf"
    assert Settings().workspaces_path == tmp_path / "home" / ".local" / "share" / "maf" / "workspaces"


def test_data_home_takes_an_explicit_environment(tmp_path: Path) -> None:
    assert data_home({"XDG_DATA_HOME": str(tmp_path)}) == tmp_path / "maf"


@pytest.mark.parametrize(
    "setup",
    ["other-project", "no-pyproject", "bad-toml", "not-src"],
)
def test_source_checkout_needs_src_maf_and_a_maf_pyproject(tmp_path: Path, setup: str) -> None:
    root = tmp_path / "repo"
    package = _fake_checkout(root, name="other" if setup == "other-project" else "maf")
    if setup == "no-pyproject":
        (root / "pyproject.toml").unlink()
    elif setup == "bad-toml":
        (root / "pyproject.toml").write_text("[project\nname = maf", encoding="utf-8")
    elif setup == "not-src":
        package = root / "lib" / "maf"
        package.mkdir(parents=True)
    assert source_checkout(package) is None
    assert source_checkout(_fake_checkout(tmp_path / "ok", name="MAF")) == tmp_path / "ok"


def test_env_and_config_still_beat_the_checkout_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MAF_WORKSPACES", str(tmp_path / "ws"))
    s = _load(tmp_path, f"mcp_inbox: {tmp_path / 'in'}\npython_executable: {sys.executable}\n")
    assert (s.workspaces_path, s.mcp_inbox) == (tmp_path / "ws", tmp_path / "in")


# -- export_exclude adds to the built-in list ---------------------------------------------------


def test_export_exclude_adds_to_the_defaults_instead_of_replacing_them(tmp_path: Path) -> None:
    s = _load(tmp_path, "export_exclude: [results, '*.log']\n")
    patterns = export_excludes(s)
    assert patterns == (*DEFAULT_EXPORT_EXCLUDES, "results", "*.log")
    for rel in ("results/run.csv", "logs/sim.log"):
        assert lint.excluded(rel, patterns), rel
    for rel in ("build/a.o", "src/__pycache__/m.pyc", ".git/HEAD", ".maf/run.log", "inputs/brief.md", "x/node_modules/y"):
        assert lint.excluded(rel, patterns), rel  # the built-in list still applies
    assert not lint.excluded("src/main.c", patterns)


def test_export_exclude_repeating_a_default_is_harmless(tmp_path: Path) -> None:
    s = _load(tmp_path, "export_exclude: [.git, build/*, extra]\n")
    assert export_excludes(s) == (*DEFAULT_EXPORT_EXCLUDES, "extra")


def test_export_include_still_cannot_reach_protected_paths(tmp_path: Path) -> None:
    s = _load(tmp_path, "export_exclude: [results]\nexport_include: ['build/*.cmake', '.maf/*', 'results/keep.csv']\n")
    patterns = export_excludes(s)
    assert patterns[-len(PROTECTED_EXPORT_EXCLUDES):] == PROTECTED_EXPORT_EXCLUDES
    assert not lint.excluded("build/toolchain.cmake", patterns)
    assert not lint.excluded("results/keep.csv", patterns) and lint.excluded("results/other.csv", patterns)
    assert lint.excluded(".maf/run.log", patterns)
