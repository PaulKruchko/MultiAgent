"""Tests for the strategy stage."""

from __future__ import annotations

import pytest

from maf import handoff as hf
from maf.handoff import HandoffInvalid, HandoffKind
from maf.stages.strategy import (
    CRITERION_RE,
    Criterion,
    StrategyBackend,
    acceptance_criteria_errors,
    criteria_guidance,
    normalize_acceptance_criteria,
    parse_acceptance_criteria,
)
from test_stages_base import StageEnv, prompt_of, stage_env  # noqa: F401


@pytest.fixture
def with_ingestion(stage_env: StageEnv, sample_bodies: dict[str, str]) -> StageEnv:
    stage_env.put("01a-routing", HandoffKind.ROUTING, sample_bodies["routing"], from_="chatgpt")
    stage_env.put("01-ingestion", HandoffKind.INGESTION, sample_bodies["ingestion"], from_="gemini")
    return stage_env


def test_strategy_consumes_routing_and_ingestion(with_ingestion: StageEnv, sample_bodies: dict[str, str]) -> None:
    env = with_ingestion
    env.fakes.chatgpt.script(sample_bodies["strategy"])
    output = StrategyBackend().run_stage(env.ctx("strategy"))

    assert [n.name for n in output.notes] == ["02-strategy"]
    assert output.index_updates == {} and not output.loop_back
    note = output.notes[0].handoff
    assert (note.meta.from_, note.meta.to, note.meta.stage) == ("chatgpt", "claude", HandoffKind.STRATEGY)
    assert note.meta.inputs == ["[[01a-routing]]", "[[01-ingestion]]"]

    (request,) = env.fakes.chatgpt.calls
    prompt = prompt_of(request)
    assert '<note name="01a-routing">' in prompt and '<note name="01-ingestion">' in prompt
    assert "Masmano" in prompt and "Survey O(1) allocator designs" in prompt
    assert env.index.brief in prompt
    assert "review gate" not in prompt
    assert prompt.count(hf.format_spec(HandoffKind.STRATEGY)) == 1
    assert "ChatGPT" in request.system


def test_review_note_is_included(with_ingestion: StageEnv, sample_bodies: dict[str, str]) -> None:
    with_ingestion.fakes.chatgpt.script(sample_bodies["strategy"])
    StrategyBackend().run_stage(with_ingestion.ctx("strategy", review_note="Prefer a buddy allocator."))
    prompt = prompt_of(with_ingestion.fakes.chatgpt.calls[0])
    assert "review gate" in prompt and "Prefer a buddy allocator." in prompt


def test_strategy_repair_then_failure(with_ingestion: StageEnv) -> None:
    with_ingestion.fakes.chatgpt.script("## Summary\n\nx", "## Summary\n\ny")
    with pytest.raises(HandoffInvalid):
        StrategyBackend().run_stage(with_ingestion.ctx("strategy"))
    assert len(with_ingestion.fakes.chatgpt.calls) == 2


def test_missing_ingestion_note_raises(stage_env: StageEnv) -> None:
    with pytest.raises(FileNotFoundError):
        StrategyBackend().run_stage(stage_env.ctx("strategy"))
    assert stage_env.fakes.chatgpt.calls == []


# ---------------------------------------------------------------------- acceptance criteria


GRAMMAR_CRITERIA = """- AC-1 [hard]: `make all` passes on POSIX, FreeRTOS and QEMU, 3 runs in a row.
- AC-2 [hard]: `.text` < 2048 bytes on Cortex-M3 at `-Os`.
  Measured with `arm-none-eabi-size`,
  - on the release build only.
- AC-3 [soft]: README explains the lock hooks."""


def _strategy_with(body: str, criteria: str) -> str:
    head, rest = body.split("## Acceptance Criteria\n", 1)
    return f"{head}## Acceptance Criteria\n\n{criteria}\n\n## Risks\n" + rest.split("## Risks\n", 1)[1]


def test_grammar_criteria_are_kept_verbatim(with_ingestion: StageEnv, sample_bodies: dict[str, str]) -> None:
    with_ingestion.fakes.chatgpt.script(_strategy_with(sample_bodies["strategy"], GRAMMAR_CRITERIA))
    note = StrategyBackend().run_stage(with_ingestion.ctx("strategy", mode="code")).notes[0].handoff
    assert note.section("Acceptance Criteria") == GRAMMAR_CRITERIA
    assert parse_acceptance_criteria(note) == [
        Criterion("AC-1", True, "`make all` passes on POSIX, FreeRTOS and QEMU, 3 runs in a row."),
        Criterion("AC-2", True, "`.text` < 2048 bytes on Cortex-M3 at `-Os`. Measured with `arm-none-eabi-size`, "
                                "- on the release build only."),
        Criterion("AC-3", False, "README explains the lock hooks."),
    ]


def test_fixture_criteria_follow_the_grammar(sample_bodies: dict[str, str]) -> None:
    section = hf.split_sections(sample_bodies["strategy"])[2]["Acceptance Criteria"]
    assert acceptance_criteria_errors(section) == []


LOOSE_CRITERIA = "- All tests pass on POSIX, FreeRTOS and QEMU.\n- `.text` < 2048 bytes on Cortex-M3 at `-Os`."


def test_loose_criteria_are_rewritten_into_the_grammar(with_ingestion: StageEnv, sample_bodies: dict[str, str]) -> None:
    """Plain bullets, as ChatGPT wrote them before the grammar: they become hard criteria."""
    with_ingestion.fakes.chatgpt.script(_strategy_with(sample_bodies["strategy"], LOOSE_CRITERIA))
    output = StrategyBackend().run_stage(with_ingestion.ctx("strategy", mode="code"))
    note = output.notes[0].handoff
    assert note.section("Acceptance Criteria") == (
        "- AC-1 [hard]: All tests pass on POSIX, FreeRTOS and QEMU.\n"
        "- AC-2 [hard]: `.text` < 2048 bytes on Cortex-M3 at `-Os`."
    )
    assert hf.validate_handoff(note) == []
    assert len(with_ingestion.fakes.chatgpt.calls) == 1  # normalized by Python, not sent back for repair


@pytest.mark.parametrize(
    ("mode", "present", "absent"),
    [
        ("code", ["hard clean-room criterion", "at least 3 consecutive runs", "model mismatch"], ["include code"]),
        ("mixed", ["hard clean-room criterion", "model mismatch", "verified by the source audit"], ["include code"]),
        ("prose", ["verified by the source audit"], ["clean-room", "model mismatch"]),
        (None, ["If the deliverables include code: a hard clean-room criterion", "verified by the source audit"], []),
    ],
)
def test_prompt_asks_for_numbered_hard_soft_criteria_per_mode(
    with_ingestion: StageEnv, sample_bodies: dict[str, str], mode: str | None, present: list[str], absent: list[str]
) -> None:
    with_ingestion.fakes.chatgpt.script(sample_bodies["strategy"])
    StrategyBackend().run_stage(with_ingestion.ctx("strategy", mode=mode))  # type: ignore[arg-type]
    prompt = prompt_of(with_ingestion.fakes.chatgpt.calls[0])
    assert "`- AC-<n> [hard]: <criterion>`" in prompt and "testable" in prompt
    for text in present:
        assert text in prompt
    for text in absent:
        assert text not in prompt


def test_without_the_source_audit_references_must_come_from_the_ingestion() -> None:
    assert "verified by the source audit" in criteria_guidance("prose", source_audit=True)
    guidance = criteria_guidance("prose", source_audit=False)
    assert "source audit" not in guidance and "ingestion report's `## Sources`" in guidance


def test_prompt_follows_the_source_audit_setting(with_ingestion: StageEnv, sample_bodies: dict[str, str]) -> None:
    with_ingestion.settings = with_ingestion.settings.model_copy(update={"source_audit": False})
    with_ingestion.fakes.chatgpt.script(sample_bodies["strategy"])
    StrategyBackend().run_stage(with_ingestion.ctx("strategy", mode="prose"))
    assert "source audit" not in prompt_of(with_ingestion.fakes.chatgpt.calls[0])


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ("- AC-1 [hard]: a\n- AC-2 [soft]: b", [("AC-1", True, "a"), ("AC-2", False, "b")]),
        ("1. **AC-2 (soft):** docs\n2. AC-5 [HARD] - size", [("AC-2", False, "docs"), ("AC-5", True, "size")]),
        ("- [soft] nice to have\n- AC3: no dash", [("AC-1", False, "nice to have"), ("AC-3", True, "no dash")]),
        ("- **Build**: `make` passes", [("AC-1", True, "**Build**: `make` passes")]),
        ("Intro line.\n\n- one\n- two\ncontinued", [("AC-1", True, "one"), ("AC-2", True, "two continued")]),
        ("- AC-2 [hard]: x\n- AC-2 [soft]: dup\n- y", [("AC-2", True, "x"), ("AC-1", False, "dup"), ("AC-3", True, "y")]),
        ("  - AC-4 [soft]: indented but strict", [("AC-4", False, "indented but strict")]),
        ("- AC-1 [hard]:\n  text on the next line", [("AC-1", True, "text on the next line")]),
        ("- AC-1 [hard]:\n- AC-2 [soft]: b", [("AC-2", False, "b")]),
        ("All tests pass and the code fits in 2 KB.", [("AC-1", True, "All tests pass and the code fits in 2 KB.")]),
        # layouts seen in the demo strategies: a task-list checkbox, the level after a closing bold marker
        ("- [ ] AC-2 [soft]: figures have captions\n- [x] AC-1 [hard]: builds", [
            ("AC-2", False, "figures have captions"), ("AC-1", True, "builds"),
        ]),
        ("2. **AC-2** (soft) — captions\n3. **AC-3** [hard]: size", [("AC-2", False, "captions"), ("AC-3", True, "size")]),
        ("- [ ] figures have captions", [("AC-1", True, "figures have captions")]),
        ("None.", []),
        ("", []),
    ],
)
def test_parse_acceptance_criteria(section: str, expected: list[tuple[str, bool, str]]) -> None:
    assert [(c.id, c.hard, c.text) for c in parse_acceptance_criteria(section)] == expected


def test_criterion_line_round_trips() -> None:
    criteria = parse_acceptance_criteria("1. AC-7 (soft): fast\n- slow")
    assert [c.line for c in criteria] == ["- AC-7 [soft]: fast", "- AC-1 [hard]: slow"]
    assert parse_acceptance_criteria("\n".join(c.line for c in criteria)) == criteria
    assert all(CRITERION_RE.match(c.line) for c in criteria)


@pytest.mark.parametrize(
    ("section", "fragment"),
    [
        ("- All tests pass.", "does not match"),
        ("Intro.\n- AC-1 [hard]: x", "'Intro.' does not match"),
        ("- AC-1 [must]: x", "does not match"),
        ("- AC-1 [hard]: x\n- AC-1 [soft]: y", "duplicate id AC-1"),
        ("- AC-1 [hard]:", "criterion AC-1 has no text"),
        ("- AC-1 [hard]: x\n  - AC-2 hard: y", "does not match"),
        ("  indented first line", "does not match"),
        ("", "at least one criterion"),
    ],
)
def test_acceptance_criteria_errors(section: str, fragment: str) -> None:
    errors = acceptance_criteria_errors(section)
    assert errors and any(fragment in e for e in errors)


def test_strict_sections_are_valid_and_unchanged_by_normalization() -> None:
    assert acceptance_criteria_errors(GRAMMAR_CRITERIA) == []
    assert normalize_acceptance_criteria(GRAMMAR_CRITERIA) == GRAMMAR_CRITERIA
    loose = "- All tests pass.\n1. **AC-3 (soft):** docs"
    assert normalize_acceptance_criteria(loose) == "- AC-1 [hard]: All tests pass.\n- AC-3 [soft]: docs"
    assert acceptance_criteria_errors(normalize_acceptance_criteria(loose)) == []
