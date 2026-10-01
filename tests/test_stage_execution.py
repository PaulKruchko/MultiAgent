"""Tests for the execution stage (Claude Code for code/mixed, Claude Messages for prose)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest

from maf import handoff as hf
from maf.handoff import HandoffInvalid, HandoffKind
from maf.providers.base import SandboxUnavailable
from maf.providers.claude_code import (
    PREFLIGHT_COMMAND,
    PREFLIGHT_FILE,
    PREFLIGHT_OUTPUT,
    PREFLIGHT_SCHEMA,
    ClaudeCodeProvider,
    CompletedProcess,
    format_budget,
    preflight_budget_usd,
)
from maf.lint import LintIssue
from maf.stages import execution
from maf.stages.execution import (
    DELIVERABLE_RULES,
    LINT_SECTION,
    MODE_GUIDANCE,
    ExecutionBackend,
    demote_headings,
    ensure_sandbox,
    extract_document,
    parse_artifact_paths,
    promote_headings,
    resolve_in_workspace,
    with_lint,
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


class FakeClaudeCli:
    """Stands in for the ``claude`` binary behind a real ``ClaudeCodeProvider``: answers the sandbox preflight by
    hashing the probe file in its cwd and writing the digest to ``PREFLIGHT_OUTPUT`` (or wrongly, with
    ``preflight_ok=False``) and every other prompt with ``body``."""

    def __init__(self, body: str, *, preflight_ok: bool = True, preflight_cost: float = 0.04, cost: float = 1.5):
        self.body = body
        self.preflight_ok = preflight_ok
        self.preflight_cost = preflight_cost
        self.cost = cost
        self.calls: list[tuple[list[str], str]] = []

    def __call__(
        self, argv: Sequence[str], stdin: str, cwd: Path, env: dict[str, str], timeout: float
    ) -> CompletedProcess:
        self.calls.append((list(argv), stdin))
        payload: dict[str, object] = {"type": "result", "subtype": "success", "is_error": False, "num_turns": 2}
        if PREFLIGHT_COMMAND in stdin:
            data = (cwd / PREFLIGHT_FILE).read_bytes()
            digest = hashlib.sha256(data if self.preflight_ok else b"other").hexdigest()
            (cwd / PREFLIGHT_OUTPUT).write_text(f"{digest}  {env['TMPDIR']}/claude-1000/maf-preflight\n")
            payload |= {"result": "", "structured_output": {"digest": digest}, "total_cost_usd": self.preflight_cost}
        else:
            payload |= {"result": self.body, "total_cost_usd": self.cost}
        return CompletedProcess(returncode=0, stdout=json.dumps(payload))

    @property
    def prompts(self) -> list[str]:
        return [stdin for _argv, stdin in self.calls]


@pytest.fixture
def real_code(
    ready: StageEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., FakeClaudeCli]]:
    """Install a real ``ClaudeCodeProvider`` (fake CLI runner, short private TMPDIR base) as the claude_code role."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-real")
    base = Path(tempfile.mkdtemp(prefix="m", dir="/tmp"))  # 14 bytes: TMPDIR fits max_tmpdir_bytes()

    def install(body: str, **kw: object) -> FakeClaudeCli:
        cli = FakeClaudeCli(body, **kw)  # type: ignore[arg-type]
        ready.fakes.claude_code = ClaudeCodeProvider(  # type: ignore[assignment]
            ready.paths.workspace,
            executable=Path("/opt/claude"),
            allowed_tools=ready.settings.claude_code_tools,
            runner=cli,
            secrets_dir=tmp_path / "secrets",
            tmp_base=base,
            turn_context_tokens=10_000,
            turn_output_tokens=1_000,
        )
        return cli

    yield install
    shutil.rmtree(base, ignore_errors=True)


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _ledger(env: StageEnv) -> list[tuple[str, str, float]]:
    return [(e.stage, e.purpose, e.cost_usd) for e in env.ledger.entries]


@pytest.mark.parametrize("mode", ["code", "mixed"])
def test_code_modes_run_the_sandbox_preflight_first(
    ready: StageEnv, real_code: Callable[..., FakeClaudeCli], mode: str
) -> None:
    ready.workspace_file("src/alloc.c")
    cli = real_code(CODE_BODY)
    output = ExecutionBackend().run_stage(ready.ctx("execution", mode=mode))  # type: ignore[arg-type]

    preflight, execution = cli.calls
    assert PREFLIGHT_COMMAND in preflight[1]
    budget = format_budget(preflight_budget_usd(ready.settings.model_for("claude_code", "execution")))
    assert (_flag(preflight[0], "--effort"), _flag(preflight[0], "--max-budget-usd")) == ("low", budget)
    assert json.loads(_flag(preflight[0], "--json-schema")) == PREFLIGHT_SCHEMA
    assert MODE_GUIDANCE[mode] in execution[1]  # type: ignore[index]
    assert _ledger(ready) == [("execution", "preflight", 0.04), ("execution", "execution", 1.5)]
    assert {(e.provider, e.agent) for e in ready.ledger.entries} == {("claude_code", "claude")}
    assert output.notes[0].handoff.meta.cost_usd == pytest.approx(1.5)  # the note's own calls only
    assert ready.fakes.claude_code.sandbox_verified is True  # type: ignore[attr-defined]
    assert not (ready.paths.workspace / PREFLIGHT_FILE).exists()


def test_preflight_budget_comes_from_settings(ready: StageEnv, real_code: Callable[..., FakeClaudeCli]) -> None:
    ready.settings = ready.settings.model_copy(update={"claude_code_preflight_budget_usd": 0.3})
    cli = real_code(CODE_BODY)
    ExecutionBackend().run_stage(ready.ctx("execution", mode="code"))
    assert _flag(cli.calls[0][0], "--max-budget-usd") == "0.3000"


def test_max_tier_preflight_budget_covers_a_fable_turn(ready: StageEnv, real_code: Callable[..., FakeClaudeCli]) -> None:
    """A flat $0.15 cap was below the price of one Fable first turn (about 13k tokens at $12.50/M), so every max-tier
    code run would have stopped at the preflight. Unset, the cap scales with the model."""
    ready.settings = ready.settings.model_copy(update={"tier": "max"})
    assert ready.settings.claude_code_preflight_budget_usd is None
    cli = real_code(CODE_BODY)
    ExecutionBackend().run_stage(ready.ctx("execution", mode="code"))
    argv = cli.calls[0][0]
    assert _flag(argv, "--model") == "claude-fable-5-1"
    assert _flag(argv, "--max-budget-usd") == format_budget(preflight_budget_usd("claude-fable-5-1")) == "1.0911"
    assert _ledger(ready)[0][:2] == ("execution", "preflight")


def test_preflight_runs_once_per_run(
    ready: StageEnv, real_code: Callable[..., FakeClaudeCli], sample_bodies: dict[str, str]
) -> None:
    cli = real_code(CODE_BODY)
    ExecutionBackend().run_stage(ready.ctx("execution", mode="code"))
    ready.put("04-crosscheck", HandoffKind.CROSSCHECK, sample_bodies["crosscheck"], from_="maf")
    ExecutionBackend().run_stage(ready.ctx("execution", mode="code", round=2))
    assert [PREFLIGHT_COMMAND in p for p in cli.prompts] == [True, False, False]
    assert [purpose for _stage, purpose, _cost in _ledger(ready)] == ["preflight", "execution", "execution"]


def test_prose_mode_never_preflights(ready: StageEnv, real_code: Callable[..., FakeClaudeCli]) -> None:
    cli = real_code(CODE_BODY)
    ready.fakes.claude.script(PROSE_BODY)
    ExecutionBackend().run_stage(ready.ctx("execution", mode="prose"))
    assert cli.calls == []
    assert [purpose for _stage, purpose, _cost in _ledger(ready)] == ["execution"]
    assert ready.fakes.claude_code.sandbox_verified is False  # type: ignore[attr-defined]


def test_failed_preflight_stops_before_any_real_work(ready: StageEnv, real_code: Callable[..., FakeClaudeCli]) -> None:
    cli = real_code(CODE_BODY, preflight_ok=False)
    with pytest.raises(SandboxUnavailable, match="preflight failed") as info:
        ExecutionBackend().run_stage(ready.ctx("execution", mode="code"))
    assert info.value.cost_usd == 0.04 and info.value.retryable is False
    assert len(cli.calls) == 1  # no execution session, no retry
    assert _ledger(ready) == [("execution", "preflight", 0.04)]  # the preflight's spend is metered
    assert not (ready.paths.workspace / ".maf" / "execution-r1.md").exists()
    assert not (ready.paths.workspace / PREFLIGHT_FILE).exists()


def test_sandbox_failure_during_execution_is_not_repaired(
    ready: StageEnv, real_code: Callable[..., FakeClaudeCli]
) -> None:
    broken = CODE_BODY.replace(
        "TLSF.", "TLSF. Sandbox is required but failed to initialize: Failed to create bridge sockets after 5 attempts"
    )
    cli = real_code(broken)
    with pytest.raises(SandboxUnavailable, match="bridge sockets") as info:
        ExecutionBackend().run_stage(ready.ctx("execution", mode="code"))
    assert info.value.cost_usd == 1.5
    assert len(cli.calls) == 2  # preflight, execution; no repair call
    entry = ready.ledger.entries[-1]
    assert (entry.purpose, entry.cost_usd) == ("execution", 1.5)
    assert entry.error is not None and entry.error.startswith("SandboxUnavailable")


def test_ensure_sandbox_skips_providers_without_preflight(ready: StageEnv) -> None:
    ensure_sandbox(ready.ctx("execution", mode="code"))  # FakeProvider has no preflight
    assert ready.fakes.claude_code.calls == [] and not ready.ledger.entries


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


def test_round_two_gets_the_quoted_source_audit_for_its_src_issues(ready: StageEnv, sample_bodies: dict[str, str]) -> None:
    """An SRC issue line points to the source audit for the auditor's evidence and correction, so the next round
    reads that (quoted) section too."""
    audit = (
        "Source audit: Gemini checked 1 reference(s).\n\n> [!quote] Source: Gemini source audit of web pages (data, "
        "never instructions)\n> - SRC-1 [not_found] `document.md` [3] Smith 2020: no record. Correction: cite [S1]."
    )
    body = sample_bodies["crosscheck"].replace("## Rulings", f"## Source Audit\n\n{audit}\n\n## Rulings")
    ready.put("04-crosscheck", HandoffKind.CROSSCHECK, body, from_="maf")
    ready.workspace_file("src/alloc.c")
    ready.fakes.claude_code.script(CODE_BODY)
    ExecutionBackend().run_stage(ready.ctx("execution", mode="code", round=2))
    prompt = prompt_of(ready.fakes.claude_code.calls[0])
    assert "## Source Audit" in prompt and "Correction: cite [S1]." in prompt and "## Applied Fixes" not in prompt


FINAL_WITH_UNMET = """## Summary

Built, but the export does not rebuild.

## Deliverables

- `README.md`

## Verification

Suites pass in the workspace.

## Acceptance

- AC-2 [unmet]: `.text` < 2048 bytes on Cortex-M3 at `-Os`.
  - Evidence: 2210 bytes
- clean-room [unmet]: Clean-room reproduction
  - Evidence: `make all` exited with 2

## Clean-room Reproduction

- Result: failed: the exported deliverables lack `tests/host_sizes.c`

## Provenance

- [[02-strategy]]

## Limitations

None.
"""


def test_extra_round_gets_the_criteria_final_found_unmet(ready: StageEnv, sample_bodies: dict[str, str]) -> None:
    """``maf resume --extra-round`` after final: the index still holds final's unmet criteria, and the execution pass
    gets them with 05-final's verdicts and clean-room record, besides the cross-check's (empty) unresolved issues."""
    ready.put("04-crosscheck", HandoffKind.CROSSCHECK, sample_bodies["crosscheck"], from_="maf")
    ready.put("05-final", HandoffKind.FINAL, FINAL_WITH_UNMET)
    ready.workspace_file("src/alloc.c")
    ready.fakes.claude_code.script(CODE_BODY, CODE_BODY)
    ctx = ready.ctx("execution", mode="code", round=2)
    unmet = ["AC-2 [unmet]: `.text` < 2048 bytes on Cortex-M3 at `-Os`.", "clean-room [unmet]: Clean-room reproduction"]
    ctx.index = ctx.index.model_copy(update={"criteria_unmet": 2, "unmet_criteria": unmet})
    note = ExecutionBackend().run_stage(ctx).notes[0].handoff

    prompt = prompt_of(ready.fakes.claude_code.calls[0])
    block = prompt.split("## Acceptance criteria not met at the last final report ([[05-final]])", 1)[1]
    assert "- AC-2 [unmet]: `.text` < 2048 bytes" in block and "- clean-room [unmet]:" in block
    assert '<note name="05-final">' in block and "Evidence: 2210 bytes" in block
    assert "lack `tests/host_sizes.c`" in block and "## Provenance" not in block
    assert "## Unresolved Critical\n\nNone." in prompt  # the cross-check part stays
    assert note.meta.inputs[-2:] == ["[[04-crosscheck]]", "[[05-final]]"]

    # A normal loop round (final has not run) has no such block, even with an old 05-final on disk.
    ExecutionBackend().run_stage(ready.ctx("execution", mode="code", round=2))
    assert "not met at the last final report" not in prompt_of(ready.fakes.claude_code.calls[1])


def test_final_verdict_sections_match_the_final_stage() -> None:
    from maf.stages import final

    assert execution.FINAL_VERDICT_SECTIONS == (final.ACCEPTANCE_SECTION, final.CLEANROOM_SECTION)


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


# ---------------------------------------------------------------------- deliverable rules and lint


def test_both_prompts_carry_the_deliverable_rules(ready: StageEnv) -> None:
    ready.workspace_file("src/alloc.c")
    ready.fakes.claude_code.script(CODE_BODY)
    ExecutionBackend().run_stage(ready.ctx("execution", mode="code"))
    ready.fakes.claude.script(PROSE_BODY)
    ExecutionBackend().run_stage(ready.ctx("execution", mode="prose"))

    code_request, prose_request = ready.fakes.claude_code.calls[0], ready.fakes.claude.calls[0]
    for request in (code_request, prose_request):
        prompt = prompt_of(request)
        assert prompt.count(DELIVERABLE_RULES) == 1
        assert "{{" not in prompt  # the rules are inserted verbatim, never re-rendered
    for needle in ("## Sources", "[[01-ingestion]]", "previous revision", "`n/a`", "\\lvert x \\rvert"):
        assert needle in DELIVERABLE_RULES
    system = code_request.system
    assert "at least 3 times" in system and "ingestion report lists" in system and "clean copy" in system


def test_mode_guidance_asks_for_reproduction_repetition_and_sensitivity() -> None:
    for mode in ("code", "mixed"):
        guidance = MODE_GUIDANCE[mode]  # type: ignore[index]
        assert "`README.md`" in guidance and "one reproduction command" in guidance
        assert "at least 3 times" in guidance and "deterministic" in guidance
        assert "files you add while fixing" in guidance
        # round 1 knows what ships: the whole tree minus pipeline state, inputs, the kernel and build output
        assert execution.EXPORT_RULES in guidance and "The whole workspace tree is exported" in guidance
        assert "`inputs/`" in guidance and "`*.o`" in guidance and "absolute path" in guidance
    assert "model-mismatch" in MODE_GUIDANCE["mixed"] and "which conclusions survive" in MODE_GUIDANCE["mixed"]


def test_a_lint_failure_never_discards_the_paid_execution_pass(
    ready: StageEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A linter crash after the Claude Code session used to fail the run and throw the note away (and a resume paid
    for the session again); now it is logged and noted in one line."""

    def broken(*_args: object, **_kw: object) -> list[LintIssue]:
        raise ValueError("embedded null byte")

    monkeypatch.setattr(execution._lint, "lint_deliverables", broken)
    ready.fakes.claude_code.script(CODE_BODY)
    note = ExecutionBackend().run_stage(ready.ctx("execution", mode="code")).notes[0].handoff
    assert note.section(LINT_SECTION) == (
        "maf lint failed after this pass (ValueError: embedded null byte); the cross-check lints again."
    )
    assert hf.validate_handoff(note) == []


LEAKY_DOC = """# Burn control

The scaling [[01-ingestion]] holds. As proposed in review, the gain is scheduled.

| Case | Q |
|---|---|
| A | 10 |
| B | n/a |
"""


def test_lint_findings_are_listed_in_the_execution_note(ready: StageEnv, caplog: pytest.LogCaptureFixture) -> None:
    def write_then_answer(_request: object) -> str:
        ready.workspace_file("src/alloc.c")
        ready.workspace_file("document.md", LEAKY_DOC)
        return CODE_BODY

    ready.fakes.claude_code.script(write_then_answer)
    with caplog.at_level(logging.INFO, logger="maf.stages.execution"):
        note = ExecutionBackend().run_stage(ready.ctx("execution", mode="mixed")).notes[0].handoff

    assert list(note.sections)[-1] == LINT_SECTION
    assert hf.validate_handoff(note) == []
    lint = note.section(LINT_SECTION)
    assert "`LINT` issues" in lint
    assert "- [critical] pipeline-wikilink document.md:3: link to the internal pipeline note `[[01-ingestion]]`" in lint
    assert "- [major] meta-commentary document.md:3:" in lint
    assert "null-rendering" not in lint  # minor findings are only logged
    assert "2 critical/major and 1 minor" in caplog.text
    assert "lint: [minor] null-rendering document.md:8" in caplog.text
    assert "`logs/missing.log`" in note.section("Known Limitations")  # the missing-artifact flags are kept


def test_a_clean_workspace_adds_no_lint_section(ready: StageEnv) -> None:
    ready.workspace_file("src/alloc.c")
    ready.workspace_file("README.md", "# Allocator\n\nRun `make all`; it builds and runs every test.\n")
    ready.fakes.claude_code.script(CODE_BODY)
    note = ExecutionBackend().run_stage(ready.ctx("execution", mode="code")).notes[0].handoff
    assert LINT_SECTION not in note.sections


def test_lint_skips_pipeline_metadata_inputs_the_kernel_and_export_exclusions(ready: StageEnv) -> None:
    ready.settings = ready.settings.model_copy(update={"export_exclude": (*ready.settings.export_exclude, "scratch")})
    skipped = (
        "FreeRTOS-Kernel/README.md", "inputs/brief.md", ".maf/notes.md", "scratch/draft.md", "build/scratch/x.md",
        ".claude/notes.md", "src/build/gen.md",
    )
    for rel in skipped:
        ready.workspace_file(rel, "TODO [[05-final]]\n")
    ready.fakes.claude_code.script(CODE_BODY)
    note = ExecutionBackend().run_stage(ready.ctx("execution", mode="code")).notes[0].handoff
    assert LINT_SECTION not in note.sections  # the prompt copy in .maf/ mentions pipeline notes too


def test_lint_checks_what_ships_so_links_into_inputs_are_broken(ready: StageEnv) -> None:
    """Execution lints the same tree as the export and the cross-check (``export_excludes``): the user's ``inputs/``
    is not shipped, so a deliverable linking into it is broken there."""
    ready.workspace_file("src/alloc.c")
    ready.workspace_file("inputs/spec.md", "# spec\n")
    ready.workspace_file("README.md", "# Allocator\n\nSee [the spec](inputs/spec.md).\n")
    ready.fakes.claude_code.script(CODE_BODY)
    note = ExecutionBackend().run_stage(ready.ctx("execution", mode="code")).notes[0].handoff
    assert "- [major] broken-link README.md:3: link `[the spec](inputs/spec.md)`" in note.section(LINT_SECTION)


def test_the_prose_document_is_linted(ready: StageEnv) -> None:
    ready.fakes.claude.script(PROSE_BODY.replace("#### Model", "#### Model\n\nThe previous revision had a sign error."))
    note = ExecutionBackend().run_stage(ready.ctx("execution", mode="prose")).notes[0].handoff
    assert "- [major] meta-commentary document.md:" in note.section(LINT_SECTION)
    assert note.section("Artifacts") == "- `document.md` - Burn Control of a DT Plasma"


def test_with_lint_caps_the_list_and_leaves_clean_notes_alone(
    ready: StageEnv, sample_bodies: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    note = ready.put("03-execution", HandoffKind.EXECUTION, sample_bodies["execution"])
    minor = LintIssue("null-rendering", "minor", "a.md", 1, "x")
    assert with_lint(note, []) is note and with_lint(note, [minor]) is note

    monkeypatch.setattr(execution, "MAX_LINT_ITEMS", 2)
    issues = [LintIssue("placeholder", "major", "a.md", n, "unfilled placeholder `TODO`") for n in (1, 2, 3)]
    lint = with_lint(note, [*issues, minor]).section(LINT_SECTION)
    assert lint.splitlines()[-3:] == [
        "- [major] placeholder a.md:1: unfilled placeholder `TODO`",
        "- [major] placeholder a.md:2: unfilled placeholder `TODO`",
        "- ... and 1 more",
    ]
