"""Architect-owned smoke tests: the package imports, contracts line up, and the shared fixtures are sane."""

from __future__ import annotations

import importlib

import pytest

from maf.handoff import REQUIRED_SECTIONS, HandoffKind
from maf.prompts import load_prompt, placeholders, render_prompt
from maf.providers.base import CompletionRequest, Provider
from maf.stages import default_backends
from maf.stages.base import StageBackend
from maf.types import STAGE_ORDER

MODULES = [
    "maf",
    "maf.types",
    "maf.config",
    "maf.ledger",
    "maf.vault",
    "maf.handoff",
    "maf.providers",
    "maf.providers.base",
    "maf.providers.openai_provider",
    "maf.providers.gemini_provider",
    "maf.providers.claude_provider",
    "maf.providers.claude_code",
    "maf.stages",
    "maf.stages.base",
    "maf.stages.ingestion",
    "maf.stages.strategy",
    "maf.stages.execution",
    "maf.stages.crosscheck",
    "maf.stages.final",
    "maf.prompts",
    "maf.pipeline",
    "maf.cli",
    "maf.mcp_server",
]

PROMPTS = [
    "roles/chatgpt",
    "roles/gemini",
    "roles/claude",
    "roles/claude_code",
    "triage",
    "ingestion",
    "strategy",
    "execution_code",
    "execution_prose",
    "critique",
    "rebuttal",
    "adjudicate",
    "apply_fixes",
    "final",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_every_stage_has_a_backend() -> None:
    backends = default_backends()
    assert tuple(backends) == STAGE_ORDER
    for stage, backend in backends.items():
        assert isinstance(backend, StageBackend)
        assert backend.name == stage


def test_every_kind_has_sections_and_sample(sample_bodies: dict[str, str]) -> None:
    for kind in HandoffKind:
        assert REQUIRED_SECTIONS[kind]
        assert kind.value in sample_bodies, f"missing tests/fixtures/handoffs/{kind.value}.md"
        body = sample_bodies[kind.value]
        for title in REQUIRED_SECTIONS[kind]:
            assert f"## {title}\n" in body


@pytest.mark.parametrize("name", PROMPTS)
def test_prompts_render(name: str) -> None:
    assert load_prompt(name).strip()
    rendered = render_prompt(name, **{p: f"<{p}>" for p in placeholders(name)})
    assert "{{" not in rendered


def test_render_prompt_rejects_missing_and_extra() -> None:
    with pytest.raises(KeyError):
        render_prompt("triage", brief="x")
    with pytest.raises(TypeError):
        render_prompt("roles/chatgpt", surprise="x")


def test_fake_providers_satisfy_protocol(fake_providers) -> None:  # type: ignore[no-untyped-def]
    for fake in fake_providers.all():
        assert isinstance(fake, Provider)
    fake_providers.chatgpt.script({"ok": True}, "plain")
    req = CompletionRequest.simple("gpt-6-sol", "hi", max_output_tokens=10)
    assert fake_providers.chatgpt.complete(req).parsed == {"ok": True}
    assert fake_providers.chatgpt.complete(req).text == "plain"
    with pytest.raises(AssertionError):
        fake_providers.chatgpt.complete(req)
