"""Tests for the strategy stage."""

from __future__ import annotations

import pytest

from maf import handoff as hf
from maf.handoff import HandoffInvalid, HandoffKind
from maf.stages.strategy import StrategyBackend
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
