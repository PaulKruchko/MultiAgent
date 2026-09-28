"""Tests for maf.config: pricing, price lookup, model selection and settings layering."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from maf.config import (
    PRICE_TABLE,
    TIER_MODELS,
    ModelPrice,
    Settings,
    UnknownModelPrice,
    agent_for_role,
    default_config_path,
    load_settings,
    price_for,
)
from maf.types import Usage

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
