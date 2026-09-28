"""Tests for the execution stage (Claude Code for code/mixed, Claude Messages for prose)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from maf import handoff as hf
from maf.handoff import HandoffInvalid, HandoffKind
from maf.stages.execution import (
    MODE_GUIDANCE,
    ExecutionBackend,
    demote_headings,
    extract_document,
    parse_artifact_paths,
    promote_headings,
    resolve_in_workspace,
)
from test_stages_base import StageEnv, prompt_of, stage_env  # noqa: F401

CODE_BODY = """## Summary

Built and tested.

## Artifacts

- `src/alloc.c` - allocator
- `plots/latency.png` - latency histogram
- `logs/missing.log` - never written

## Implementation Notes

TLSF.

## Verification

`make test` passes.

## Known Limitations

None.
"""

PROSE_BODY = """## Summary

A short report.

## Artifacts

### Burn Control of a DT Plasma

#### Introduction

The fusion power $P_f$ scales as $n^2 \\langle\\sigma v\\rangle$.

```python
# not a heading
```

#### Model

$$
\\frac{dW}{dt} = P_\\alpha + P_{aux} - W/\\tau_E
$$

## Implementation Notes

Written from the ingestion report.

## Verification

Equations checked against the sources.

## Known Limitations

None.
"""


@pytest.fixture
def ready(stage_env: StageEnv, sample_bodies: dict[str, str]) -> StageEnv:
    stage_env.put("01-ingestion", HandoffKind.INGESTION, sample_bodies["ingestion"], from_="gemini")
    stage_env.put("02-strategy", HandoffKind.STRATEGY, sample_bodies["strategy"], from_="chatgpt")
    return stage_env


def test_code_mode_runs_claude_code_and_embeds_images(ready: StageEnv) -> None:
    def write_files_then_answer(_request: object) -> str:
        ready.workspace_file("src/alloc.c", "int x;\n")
        ready.workspace_file("plots/latency.png", b"\x89PNG\r\n\x1a\nfake")
        return CODE_BODY

    ready.fakes.claude_code.script(write_files_then_answer)
    output = ExecutionBackend().run_stage(ready.ctx("execution", mode="code"))

    assert [n.name for n in output.notes] == ["03-execution"]
    note = output.notes[0].handoff
    assert (note.meta.from_, note.meta.to) == ("claude", "crosscheck")
    assert note.meta.inputs == ["[[01-ingestion]]", "[[02-strategy]]"]

    (request,) = ready.fakes.claude_code.calls
    assert request.max_budget_usd == pytest.approx(ready.settings.output_limits.claude_code_budget_usd)
    assert request.effort == "high"
    assert str(ready.settings.python_executable) in request.system
    prompt = prompt_of(request)
    assert MODE_GUIDANCE["code"] in prompt
    assert prompt.count(hf.format_spec(HandoffKind.EXECUTION)) == 1
    saved = ready.paths.workspace / ".maf" / "execution-r1.md"
    assert saved.read_text(encoding="utf-8") == prompt

    artifacts = note.section("Artifacts")
    assert f"- `plots/latency.png` - latency histogram ![[runs/{ready.index.run_id}/assets/latency.png]]" in artifacts
    assert (ready.paths.assets / "latency.png").is_file()
    assert "`src/alloc.c` - allocator\n" in artifacts
    assert note.section("Known Limitations") == "- maf: listed artifact `logs/missing.log` was not found in the workspace."


def test_code_mode_rerun_is_idempotent(ready: StageEnv) -> None:
    ready.workspace_file("plots/latency.png", b"\x89PNG same")
    ready.fakes.claude_code.script(CODE_BODY, CODE_BODY)
    first = ExecutionBackend().run_stage(ready.ctx("execution", mode="mixed")).notes[0].handoff
    second = ExecutionBackend().run_stage(ready.ctx("execution", mode="mixed")).notes[0].handoff
    assert first.section("Artifacts") == second.section("Artifacts")
    assert sorted(p.name for p in ready.paths.assets.iterdir()) == ["latency.png"]
    assert MODE_GUIDANCE["mixed"] in prompt_of(ready.fakes.claude_code.calls[0])


def test_code_mode_artifact_grammar_goes_through_repair(ready: StageEnv) -> None:
    escaping = CODE_BODY.replace("`src/alloc.c`", "`../../outside.c`")
    no_bullets = CODE_BODY.replace("- `src/alloc.c` - allocator\n- `plots/latency.png` - latency histogram\n- `logs/missing.log` - never written", "Everything is in src.")
    ready.fakes.claude_code.script(escaping, no_bullets)
    with pytest.raises(HandoffInvalid) as excinfo:
        ExecutionBackend().run_stage(ready.ctx("execution", mode="code"))
    assert any("Artifacts" in e for e in excinfo.value.errors)
    repair_prompt = prompt_of(ready.fakes.claude_code.calls[1])
    assert "escapes the workspace" in repair_prompt


def test_round_two_consumes_previous_crosscheck(ready: StageEnv, sample_bodies: dict[str, str]) -> None:
    crosscheck = sample_bodies["crosscheck"].replace(
        "## Unresolved Critical\n\nNone.", "## Unresolved Critical\n\n- [critical] GEM-4: stress test crashes"
    ).replace("PASS", "LOOP")
    ready.put("04-crosscheck", HandoffKind.CROSSCHECK, crosscheck, from_="maf")
    ready.workspace_file("src/alloc.c")
    ready.fakes.claude_code.script(CODE_BODY)
    output = ExecutionBackend().run_stage(ready.ctx("execution", mode="code", round=2, review_note="Keep it small."))

    assert output.notes[0].name == "03-execution-r2"
    note = output.notes[0].handoff
    assert note.meta.round == 2
    assert note.meta.inputs[-1] == "[[04-crosscheck]]"
    prompt = prompt_of(ready.fakes.claude_code.calls[0])
    assert "GEM-4: stress test crashes" in prompt
    assert "## Rulings" in prompt and "## Applied Fixes" not in prompt
    assert "Keep it small." in prompt
    assert (ready.paths.workspace / ".maf" / "execution-r2.md").is_file()


def test_prose_mode_moves_document_to_workspace(ready: StageEnv) -> None:
    ready.fakes.claude.script(PROSE_BODY)
    output = ExecutionBackend().run_stage(ready.ctx("execution", mode="prose"))
    note = output.notes[0].handoff
    assert note.section("Artifacts") == "- `document.md` - Burn Control of a DT Plasma"
    assert ready.fakes.claude_code.calls == []
    request = ready.fakes.claude.calls[0]
    assert request.max_output_tokens == ready.settings.output_limits.claude
    assert request.effort == "high"

    document = (ready.paths.workspace / "document.md").read_text(encoding="utf-8")
    assert document.startswith("# Burn Control of a DT Plasma\n")
    assert "\n## Introduction\n" in document and "\n## Model\n" in document
    assert "# not a heading" in document and "\\frac{dW}{dt}" in document


def test_prose_round_two_revises_current_document(ready: StageEnv, sample_bodies: dict[str, str]) -> None:
    ready.put("04-crosscheck", HandoffKind.CROSSCHECK, sample_bodies["crosscheck"], from_="maf")
    ready.workspace_file("document.md", "# Old Title\n\n## Old Section\n\nold text\n")
    ready.fakes.claude.script(PROSE_BODY)
    ExecutionBackend().run_stage(ready.ctx("execution", mode="prose", round=2))
    prompt = prompt_of(ready.fakes.claude.calls[0])
    assert '<document path="document.md">\n### Old Title\n\n#### Old Section' in prompt
    assert "Burn Control" in (ready.paths.workspace / "document.md").read_text()


def test_prose_without_h3_is_repaired(ready: StageEnv) -> None:
    flat = PROSE_BODY.replace("### Burn Control of a DT Plasma\n\n", "")
    flat = flat.replace("#### Introduction", "Introduction").replace("#### Model", "Model")
    ready.fakes.claude.script(flat, PROSE_BODY)
    output = ExecutionBackend().run_stage(ready.ctx("execution", mode="prose"))
    assert "H3 title" in prompt_of(ready.fakes.claude.calls[1])
    assert output.notes[0].handoff.meta.cost_usd == pytest.approx(0.02)


def test_mode_is_required(ready: StageEnv) -> None:
    with pytest.raises(ValueError, match="mode"):
        ExecutionBackend().run_stage(ready.ctx("execution"))


def test_parse_artifact_paths() -> None:
    section = "- `a/b.c` - one\n* `dir/` - two\n  - `a/b.c` - dup\n- no path here\n+ `x y.txt`: three\ntext `z`"
    assert parse_artifact_paths(section) == ["a/b.c", "dir/", "x y.txt"]


def test_resolve_in_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    (ws / "sub").mkdir(parents=True)
    assert resolve_in_workspace(ws, "sub/file.c") == (ws / "sub" / "file.c").resolve()
    assert resolve_in_workspace(ws, "sub/../file.c") == (ws / "file.c").resolve()
    for bad in ("", "  ", "/etc/passwd", "../x", "sub/../../x", "~/x"):
        with pytest.raises(ValueError):
            resolve_in_workspace(ws, bad)
    os.symlink(tmp_path, ws / "link")
    with pytest.raises(ValueError, match="escapes"):
        resolve_in_workspace(ws, "link/secret")


def test_document_helpers() -> None:
    title, doc = extract_document("preface\n```\n### fenced\n```\n### Real Title ###\n\n#### A\n\ntext")
    assert title == "Real Title"
    assert doc.startswith("### Real Title")
    promoted = promote_headings("### T\n\n#### A\n\n```\n#### code\n```\n###### deep\n")
    assert promoted == "# T\n\n## A\n\n```\n#### code\n```\n#### deep\n"
    assert demote_headings(promoted) == "### T\n\n#### A\n\n```\n#### code\n```\n###### deep\n"
    with pytest.raises(ValueError):
        extract_document("no heading")
