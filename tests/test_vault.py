"""Tests for maf.vault: naming, atomic writes, run lifecycle, run.md rendering and copy guards."""

from __future__ import annotations

import os
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from maf.handoff import HandoffInvalid, HandoffKind, HandoffMeta, build_handoff, dump_frontmatter
from maf.types import RunStatus
from maf.vault import (
    DEFAULT_EXPORT_EXCLUDES,
    PROTECTED_EXPORT_EXCLUDES,
    ExportError,
    ExportTooLarge,
    RunIndex,
    Vault,
    atomic_copy,
    atomic_write_text,
    describe_issues,
    embed,
    export_excluded,
    format_bytes,
    latest_crosscheck,
    note_name,
    render_run_body,
    slugify,
    wikilink,
)

RUN_ID = "2026-09-28-portable-allocator"


@pytest.fixture
def vault(tmp_path: Path) -> Vault:
    return Vault(tmp_path / "vault", tmp_path / "workspaces")


def make_index(vault: Vault, run_id: str = RUN_ID, created: datetime | None = None, **kw: object) -> RunIndex:
    created = created or datetime(2026, 9, 28, 12, 0, 0)
    data: dict[str, object] = {
        "run_id": run_id,
        "budget_usd": 25.0,
        "created": created,
        "updated": created,
        "workspace": str(vault.workspaces_root / run_id),
        "brief": "Design a portable O(1) allocator.\n\nTarget: Cortex-M3.",
    }
    data.update(kw)
    return RunIndex.model_validate(data)


def make_handoff(kind: HandoffKind, body: str, run_id: str = RUN_ID):  # type: ignore[no-untyped-def]
    meta = HandoffMeta.model_validate(
        {
            "run_id": run_id,
            "stage": kind,
            "from": "claude",
            "to": "crosscheck",
            "created": datetime(2026, 9, 28, 12, 0, 0),
            "model": "claude-opus-5-5",
            "cost_usd": 0.25,
        }
    )
    return build_handoff(body, meta)


# ---------------------------------------------------------------------------
# pure helpers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Design a portable O(1) allocator!", "design-a-portable-o-1-allocator"),
        ("  Café -- Résumé  ", "cafe-resume"),
        ("日本語", "run"),
        ("", "run"),
        ("---", "run"),
        ("UPPER_case and_Under", "upper-case-and-under"),
    ],
)
def test_slugify(text: str, expected: str) -> None:
    assert slugify(text) == expected


def test_slugify_trims_at_word_boundary() -> None:
    assert slugify("alpha beta gamma delta", max_len=13) == "alpha-beta"
    assert slugify("alpha beta gamma", max_len=10) == "alpha-beta"
    assert slugify("abcdefghijklmnop", max_len=5) == "abcde"
    long = slugify("word " * 40)
    assert len(long) <= 48 and not long.endswith("-")


@pytest.mark.parametrize(
    ("kind", "round_", "agent", "expected"),
    [
        (HandoffKind.ROUTING, 1, None, "01a-routing"),
        (HandoffKind.INGESTION, 1, None, "01-ingestion"),
        (HandoffKind.STRATEGY, 1, None, "02-strategy"),
        (HandoffKind.EXECUTION, 1, None, "03-execution"),
        (HandoffKind.EXECUTION, 3, None, "03-execution-r3"),
        (HandoffKind.CRITIQUE, 1, "gemini", "04a-critique-gemini"),
        (HandoffKind.CRITIQUE, 2, "chatgpt", "04a-critique-chatgpt-r2"),
        (HandoffKind.REBUTTAL, 2, None, "04b-rebuttal-r2"),
        (HandoffKind.ADJUDICATION, 1, None, "04c-adjudication"),
        (HandoffKind.CROSSCHECK, 2, None, "04-crosscheck-r2"),
        (HandoffKind.FINAL, 1, None, "05-final"),
        (HandoffKind.FINAL, 3, None, "05-final"),
        (HandoffKind.STRATEGY, 2, None, "02-strategy"),
    ],
)
def test_note_name(kind: HandoffKind, round_: int, agent: str | None, expected: str) -> None:
    assert note_name(kind, round_, agent) == expected  # type: ignore[arg-type]


def test_note_name_errors() -> None:
    with pytest.raises(ValueError):
        note_name(HandoffKind.CRITIQUE)
    with pytest.raises(ValueError):
        note_name(HandoffKind.CRITIQUE, 1, "llama")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        note_name(HandoffKind.EXECUTION, 1, "claude")
    with pytest.raises(ValueError):
        note_name(HandoffKind.EXECUTION, 0)


def test_wikilink_and_embed() -> None:
    assert wikilink("01-ingestion") == "[[01-ingestion]]"
    assert wikilink("runs/x/05-final", "Final") == "[[runs/x/05-final|Final]]"
    assert embed("plot.png") == "![[plot.png]]"
    for bad in ("note.md", "", "a|b", "a]]", "a\nb"):
        with pytest.raises(ValueError):
            wikilink(bad)
    with pytest.raises(ValueError):
        wikilink("a", "b]]")
    with pytest.raises(ValueError):
        embed("")


# ---------------------------------------------------------------------------
# atomic writes


def test_atomic_write_text_creates_parents_and_replaces(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "note.md"
    atomic_write_text(target, "héllo\n")
    assert target.read_bytes() == "héllo\n".encode()
    os.chmod(target, 0o640)
    atomic_write_text(target, "second\r\n")
    assert target.read_bytes() == b"second\r\n"
    assert target.stat().st_mode & 0o777 == 0o640
    assert [p.name for p in target.parent.iterdir()] == ["note.md"]


def test_atomic_write_text_cleans_temp_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "note.md"
    target.write_text("original")

    def boom(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(target, "new")
    assert target.read_text() == "original"
    assert [p.name for p in tmp_path.iterdir()] == ["note.md"]


def test_atomic_copy_preserves_mtime(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"\x00\x01" * 1000)
    old = datetime(2020, 1, 1).timestamp()
    os.utime(src, (old, old))
    dst = tmp_path / "out" / "dst.bin"
    atomic_copy(src, dst)
    assert dst.read_bytes() == src.read_bytes()
    assert dst.stat().st_mtime == pytest.approx(old)
    with pytest.raises(FileNotFoundError):
        atomic_copy(tmp_path / "missing", dst)


# ---------------------------------------------------------------------------
# runs


def test_paths_is_pure_and_guards_ids(vault: Vault) -> None:
    p = vault.paths(RUN_ID)
    assert p.root == vault.root / "runs" / RUN_ID
    assert p.run_md.name == "run.md" and p.ledger.name == "ledger.jsonl"
    assert p.assets.name == "assets" and p.deliverables.name == "deliverables"
    assert p.workspace == vault.workspaces_root / RUN_ID
    assert p.note("01-ingestion") == p.root / "01-ingestion.md"
    assert not vault.root.exists()
    for bad in ("../etc", "a/b", "", ".hidden", "a..b", "/abs"):
        with pytest.raises(ValueError):
            vault.paths(bad)


def test_new_run_id_suffixes(vault: Vault) -> None:
    today = date(2026, 9, 28)
    assert vault.new_run_id("Portable allocator", today) == RUN_ID
    (vault.runs_dir / "2026-09-28-portable-allocator").mkdir(parents=True)
    assert vault.new_run_id("Portable allocator", today) == "2026-09-28-portable-allocator-2"
    (vault.workspaces_root / "2026-09-28-portable-allocator-2").mkdir(parents=True)
    assert vault.new_run_id("Portable allocator", today) == "2026-09-28-portable-allocator-3"


def test_create_run(vault: Vault, tmp_path: Path) -> None:
    a = tmp_path / "in1" / "spec.pdf"
    b = tmp_path / "in2" / "spec.pdf"
    for f, data in ((a, b"A"), (b, b"B")):
        f.parent.mkdir()
        f.write_bytes(data)
    index = make_index(vault)
    paths = vault.create_run(index, [a, b])
    assert paths.assets.is_dir() and paths.deliverables.is_dir()
    assert index.input_files == ["inputs/spec.pdf", "inputs/spec-2.pdf"]
    assert (paths.workspace / "inputs" / "spec.pdf").read_bytes() == b"A"
    assert (paths.workspace / "inputs" / "spec-2.pdf").read_bytes() == b"B"
    assert vault.read_index(RUN_ID) == index
    with pytest.raises(FileExistsError):
        vault.create_run(make_index(vault), [])


def test_create_run_missing_input_creates_nothing(vault: Vault, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        vault.create_run(make_index(vault), [tmp_path / "missing.txt"])
    assert not (vault.runs_dir / RUN_ID).exists()
    assert not (vault.workspaces_root / RUN_ID).exists()


def test_create_run_rolls_back_on_failure(vault: Vault, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "x.txt"
    f.write_text("x")

    def fail(self: Vault, index: RunIndex) -> None:
        raise OSError("boom")

    monkeypatch.setattr(Vault, "write_index", fail)
    with pytest.raises(OSError):
        vault.create_run(make_index(vault), [f])
    assert not (vault.runs_dir / RUN_ID).exists()
    assert not (vault.workspaces_root / RUN_ID).exists()


def test_index_round_trip_with_all_fields(vault: Vault) -> None:
    index = make_index(
        vault,
        status=RunStatus.AWAITING_REVIEW,
        stage="execution",
        completed_stages=["ingestion", "strategy"],
        round=2,
        mode="mixed",
        tier="max",
        review=True,
        spent_usd=3.14159,
        spend_by_agent={"chatgpt": 1.0, "gemini": 0.14159, "claude": 2.0},
        spend_by_provider={"openai": 1.0, "gemini": 0.14159, "anthropic": 0.5, "claude_code": 1.5},
        unresolved_critical=1,
        handoffs=["01a-routing", "01-ingestion", "02-strategy"],
        error="line one\nline two: with colon",
        brief="Brief with 'quotes', \"double\", #hash, and: colon\n  indented line\n",
    )
    vault.create_run(index, [])
    assert vault.read_index(RUN_ID) == index
    index.updated = index.updated + timedelta(minutes=5)
    index.status = RunStatus.COMPLETED_WITH_ISSUES  # unresolved_critical=1: plain completed would be normalized
    vault.write_index(index)
    assert vault.read_index(RUN_ID) == index


def test_read_index_errors(vault: Vault) -> None:
    with pytest.raises(FileNotFoundError):
        vault.read_index(RUN_ID)
    run_md = vault.paths(RUN_ID).run_md
    run_md.parent.mkdir(parents=True)
    run_md.write_text("no frontmatter")
    with pytest.raises(ValueError):
        vault.read_index(RUN_ID)
    run_md.write_text("---\nrun_id: x\n---\n")
    with pytest.raises(ValueError):
        vault.read_index(RUN_ID)


def test_list_runs_newest_first_skips_bad(vault: Vault) -> None:
    assert vault.list_runs() == []
    base = datetime(2026, 9, 28, 12, 0, 0)
    for i, run_id in enumerate(["2026-09-28-a", "2026-09-28-b", "2026-09-28-c"]):
        vault.create_run(make_index(vault, run_id, created=base + timedelta(hours=i)), [])
    bad = vault.runs_dir / "2026-09-28-bad"
    bad.mkdir()
    (bad / "run.md").write_text("---\nrun_id: [broken\n---\n")
    (vault.runs_dir / "no-index").mkdir()
    assert [r.run_id for r in vault.list_runs()] == ["2026-09-28-c", "2026-09-28-b", "2026-09-28-a"]


# ---------------------------------------------------------------------------
# handoffs


def test_write_read_handoff(vault: Vault, sample_bodies: dict[str, str]) -> None:
    vault.create_run(make_index(vault), [])
    h = make_handoff(HandoffKind.EXECUTION, sample_bodies["execution"])
    path = vault.write_handoff(RUN_ID, "03-execution", h)
    assert path == vault.paths(RUN_ID).root / "03-execution.md"
    assert vault.has_note(RUN_ID, "03-execution")
    assert not vault.has_note(RUN_ID, "05-final")
    assert vault.read_handoff(RUN_ID, "03-execution") == h
    with pytest.raises(FileNotFoundError):
        vault.read_handoff(RUN_ID, "05-final")
    with pytest.raises(ValueError):
        vault.write_handoff(RUN_ID, "../escape", h)
    with pytest.raises(ValueError):
        vault.write_handoff("2026-09-28-other", "03-execution", h)


def test_read_handoff_hand_edited_broken(vault: Vault) -> None:
    vault.create_run(make_index(vault), [])
    vault.paths(RUN_ID).note("02-strategy").write_text("---\nstage: strategy\n---\n\n## Summary\n")
    with pytest.raises(HandoffInvalid) as info:
        vault.read_handoff(RUN_ID, "02-strategy")
    assert info.value.kind == HandoffKind.STRATEGY


# ---------------------------------------------------------------------------
# assets and deliverables


def test_copy_asset_collisions_and_idempotence(vault: Vault, tmp_path: Path) -> None:
    vault.create_run(make_index(vault), [])
    one = tmp_path / "a" / "plot.PNG"
    two = tmp_path / "b" / "plot.PNG"
    one.parent.mkdir()
    two.parent.mkdir()
    one.write_bytes(b"one")
    two.write_bytes(b"two")
    assert vault.copy_asset(RUN_ID, one) == f"![[runs/{RUN_ID}/assets/plot.png]]"
    assert vault.copy_asset(RUN_ID, one) == f"![[runs/{RUN_ID}/assets/plot.png]]"
    assert vault.copy_asset(RUN_ID, two) == f"![[runs/{RUN_ID}/assets/plot-2.png]]"
    assert vault.copy_asset(RUN_ID, two) == f"![[runs/{RUN_ID}/assets/plot-2.png]]"
    assert sorted(p.name for p in vault.paths(RUN_ID).assets.iterdir()) == ["plot-2.png", "plot.png"]


def test_copy_asset_sanitizes_names(vault: Vault, tmp_path: Path) -> None:
    vault.create_run(make_index(vault), [])
    src = tmp_path / "fig [1]|#a^.svg"
    src.write_text("<svg/>")
    assert vault.copy_asset(RUN_ID, src) == f"![[runs/{RUN_ID}/assets/fig -1---a.svg]]"
    with pytest.raises(FileNotFoundError):
        vault.copy_asset(RUN_ID, tmp_path / "missing.png")


def test_copy_deliverable_file_and_tree(vault: Vault) -> None:
    paths = vault.create_run(make_index(vault), [])
    ws = paths.workspace
    (ws / "alloc" / "src").mkdir(parents=True)
    (ws / "alloc" / "src" / "alloc.c").write_text("int x;")
    (ws / "alloc" / "README.md").write_text("# alloc")
    (ws / "alloc" / ".git").mkdir()
    (ws / "alloc" / ".git" / "HEAD").write_text("ref")
    (ws / "alloc" / "__pycache__").mkdir()
    (ws / "alloc" / "link.c").symlink_to(ws / "alloc" / "src" / "alloc.c")
    (ws / "report.pdf").write_bytes(b"%PDF")

    dst = vault.copy_deliverable(RUN_ID, ws / "report.pdf")
    assert dst == paths.deliverables.resolve() / "report.pdf" and dst.read_bytes() == b"%PDF"

    tree = vault.copy_deliverable(RUN_ID, ws / "alloc")
    files = sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file())
    assert files == ["README.md", "link.c", "src/alloc.c"]
    assert not (tree / "link.c").is_symlink()

    # Re-copy replaces the old tree (stale files disappear).
    (ws / "alloc" / "README.md").unlink()
    vault.copy_deliverable(RUN_ID, ws / "alloc")
    assert not (tree / "README.md").exists()

    nested = vault.copy_deliverable(RUN_ID, ws / "report.pdf", "docs/thesis.pdf")
    assert nested == paths.deliverables.resolve() / "docs" / "thesis.pdf"
    assert [p.name for p in paths.deliverables.iterdir() if p.name.startswith(".")] == []


def test_copy_deliverable_guards(vault: Vault, tmp_path: Path) -> None:
    paths = vault.create_run(make_index(vault), [])
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    with pytest.raises(ValueError, match="outside the workspace"):
        vault.copy_deliverable(RUN_ID, outside)
    with pytest.raises(ValueError, match="outside the workspace"):
        vault.copy_deliverable(RUN_ID, paths.workspace / ".." / ".." / "secret.txt")
    (paths.workspace / "escape.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="outside the workspace"):
        vault.copy_deliverable(RUN_ID, paths.workspace / "escape.txt")
    (paths.workspace / "tree").mkdir()
    (paths.workspace / "tree" / "leak").symlink_to(outside)
    with pytest.raises(ValueError, match="links outside"):
        vault.copy_deliverable(RUN_ID, paths.workspace / "tree")
    assert not (paths.deliverables / "tree").exists()
    ok = paths.workspace / "ok.txt"
    ok.write_text("ok")
    for bad in ("../x.txt", "/abs.txt", ""):
        with pytest.raises(ValueError):
            vault.copy_deliverable(RUN_ID, ok, bad)
    with pytest.raises(FileNotFoundError):
        vault.copy_deliverable(RUN_ID, paths.workspace / "missing.txt")


# ---------------------------------------------------------------------------
# run.md body


def test_render_run_body(vault: Vault) -> None:
    index = make_index(
        vault,
        status=RunStatus.FAILED,
        spent_usd=3.5,
        spend_by_agent={"chatgpt": 1.0, "claude": 2.5},
        spend_by_provider={"claude_code": 2.0, "openai": 1.0, "anthropic": 0.5},
        handoffs=["01a-routing", "01-ingestion"],
        error="HandoffInvalid:\nmissing section",
        input_files=["inputs/spec.pdf"],
    )
    body = render_run_body(index)
    assert body.startswith(f"# {RUN_ID}\n\n## Brief\n\n> Design a portable O(1) allocator.\n>\n> Target: Cortex-M3.\n")
    order = [body.index(f"## {s}\n") for s in ("Brief", "Status", "Handoffs", "Cost", "Workspace")]
    assert order == sorted(order)
    assert "- Status: **failed**" in body
    assert "- Error: HandoffInvalid: missing section" in body
    assert "- [[01a-routing]]\n- [[01-ingestion]]" in body
    assert "| chatgpt | 1.0000 |\n| gemini | 0.0000 |\n| claude | 2.5000 |\n| **Total** | **3.5000** |" in body
    assert "| openai | 1.0000 |\n| anthropic | 0.5000 |\n| claude_code | 2.0000 |" in body
    assert f"## Workspace\n\n`{vault.workspaces_root / RUN_ID}`" in body
    assert "- `inputs/spec.pdf`" in body


def test_render_run_body_completed_with_issues_callout(vault: Vault) -> None:
    index = make_index(
        vault,
        status=RunStatus.COMPLETED_WITH_ISSUES,
        stage="final",
        round=3,
        unresolved_critical=20,
        handoffs=["03-execution", "04-crosscheck", "03-execution-r2", "04-crosscheck-r2", "04a-critique-gemini-r3",
                  "04-crosscheck-r3", "05-final"],
    )
    body = render_run_body(index)
    status = body.split("## Status\n\n", 1)[1]
    callout, rest = status.split("\n\n", 1)
    assert callout.splitlines()[0] == "> [!warning] Completed with 20 unresolved critical issue(s)"
    assert "[[04-crosscheck-r3]]" in callout and "[[05-final]]" in callout
    assert all(line.startswith("> ") for line in callout.splitlines())
    assert rest.startswith("- Status: **completed_with_issues**\n")
    assert "- Unresolved critical issues: 20" in rest


@pytest.mark.parametrize("status", [s for s in RunStatus if s != RunStatus.COMPLETED_WITH_ISSUES])
def test_render_run_body_no_callout_for_other_statuses(vault: Vault, status: RunStatus) -> None:
    unresolved = 0 if status == RunStatus.COMPLETED else 2  # completed with open criticals reads as with_issues
    body = render_run_body(make_index(vault, status=status, unresolved_critical=unresolved, handoffs=["04-crosscheck"]))
    assert "[!warning]" not in body
    assert body.split("## Status\n\n", 1)[1].startswith("- Status: ")


def test_latest_crosscheck(vault: Vault) -> None:
    assert latest_crosscheck(make_index(vault)) is None
    names = ["04-crosscheck", "04a-critique-chatgpt-r2", "04-crosscheck-r2", "05-final"]
    assert latest_crosscheck(make_index(vault, handoffs=names)) == "04-crosscheck-r2"
    assert latest_crosscheck(make_index(vault, handoffs=["04-crosscheck", "04-crosscheck-extra"])) == "04-crosscheck"


def test_run_md_frontmatter_round_trips_completed_with_issues(vault: Vault) -> None:
    index = make_index(vault, status=RunStatus.COMPLETED_WITH_ISSUES, unresolved_critical=9)
    vault.create_run(index, [])
    text = vault.paths(RUN_ID).run_md.read_text()
    assert "\nstatus: completed_with_issues\n" in text
    assert vault.read_index(RUN_ID).status == RunStatus.COMPLETED_WITH_ISSUES



def test_legacy_completed_run_with_open_criticals_reads_as_completed_with_issues(vault: Vault) -> None:
    """The 2026-09-28 demo runs finished before completed_with_issues existed: their run.md says ``completed`` with
    open criticals. Reading normalizes it, so every reader (list, status, MCP, resume --extra-round) sees the truth."""
    vault.create_run(make_index(vault, status=RunStatus.COMPLETED, stage="final", round=3), [])
    run_md = vault.paths(RUN_ID).run_md
    data = make_index(vault, stage="final", round=3).model_dump(mode="json") | {
        "status": "completed", "unresolved_critical": 9
    }
    run_md.write_text(dump_frontmatter(data) + "\n# legacy\n", encoding="utf-8")
    assert vault.read_index(RUN_ID).status == RunStatus.COMPLETED_WITH_ISSUES
    assert [r.status for r in vault.list_runs()] == [RunStatus.COMPLETED_WITH_ISSUES]
    assert "\nstatus: completed\n" in run_md.read_text(encoding="utf-8")  # reading never rewrites run.md
    assert make_index(vault, status=RunStatus.COMPLETED, unresolved_critical=0).status == RunStatus.COMPLETED

def test_render_run_body_empty_run(vault: Vault) -> None:
    body = render_run_body(make_index(vault, workspace="/tmp/odd`path"))
    assert "None yet." in body
    assert "| (none) | 0.0000 |" in body
    assert "`` /tmp/odd`path ``" in body
    assert "- Error" not in body


def test_run_md_on_disk_has_body(vault: Vault) -> None:
    paths = vault.create_run(make_index(vault), [])
    text = paths.run_md.read_text()
    assert text.startswith("---\nrun_id: 2026-09-28-portable-allocator\nstatus: pending\n")
    assert "created: 2026-09-28T12:00:00\n" in text
    assert "brief: |-\n  Design a portable O(1) allocator.\n" in text
    assert "\n---\n\n# 2026-09-28-portable-allocator\n" in text


# ---------------------------------------------------------------------------
# acceptance criteria and exports in run.md


def test_run_md_without_the_new_keys_still_loads(vault: Vault) -> None:
    """run.md files written before criteria_unmet / exported_at existed read with the defaults."""
    vault.create_run(make_index(vault), [])
    run_md = vault.paths(RUN_ID).run_md
    data = make_index(vault, status=RunStatus.COMPLETED, stage="final").model_dump(mode="json")
    for key in ("criteria_unmet", "unmet_criteria", "exported_at", "export_note"):
        data.pop(key)
    run_md.write_text(dump_frontmatter(data) + "\n# legacy\n", encoding="utf-8")
    index = vault.read_index(RUN_ID)
    assert (index.status, index.criteria_unmet, index.unmet_criteria, index.exported_at) == (RunStatus.COMPLETED, 0, [], None)


def test_completed_with_unmet_criteria_reads_as_completed_with_issues(vault: Vault) -> None:
    index = make_index(vault, status=RunStatus.COMPLETED, criteria_unmet=1, unmet_criteria=["AC-2 [unmet]: size"])
    assert index.status == RunStatus.COMPLETED_WITH_ISSUES and index.has_issues
    assert not make_index(vault, status=RunStatus.COMPLETED).has_issues


def test_criteria_and_export_round_trip_through_run_md(vault: Vault) -> None:
    exported = datetime(2026, 9, 29, 9, 30, 0)
    index = make_index(
        vault,
        status=RunStatus.COMPLETED_WITH_ISSUES,
        criteria_unmet=2,
        unmet_criteria=["AC-2 [unmet]: verified sources support each ITER claim", "clean-room [unmet]: Clean-room"],
        exported_at=exported,
        export_note="maf export: 57 file(s), 1.2 MB",
    )
    vault.create_run(index, [])
    text = vault.paths(RUN_ID).run_md.read_text()
    assert "\ncriteria_unmet: 2\n" in text and "\nexported_at: 2026-09-29T09:30:00\n" in text
    assert vault.read_index(RUN_ID) == index
    assert "- Deliverables exported: 2026-09-29T09:30:00 (maf export: 57 file(s), 1.2 MB)" in text
    assert "- Acceptance criteria not met: 2" in text


def test_render_run_body_callout_for_unmet_criteria(vault: Vault) -> None:
    index = make_index(
        vault,
        status=RunStatus.COMPLETED_WITH_ISSUES,
        criteria_unmet=2,
        unmet_criteria=["AC-2 [unmet]: verified sources support each ITER claim", "clean-room [unmet]: rebuild"],
        handoffs=["04-crosscheck", "05-final"],
    )
    callout, rest = render_run_body(index).split("## Status\n\n", 1)[1].split("\n\n", 1)
    assert callout.splitlines() == [
        "> [!warning] Completed with 2 acceptance criteria not met",
        "> 2 acceptance criteria not met; see `## Acceptance` in [[05-final]]:",
        "> - AC-2 [unmet]: verified sources support each ITER claim",
        "> - clean-room [unmet]: rebuild",
        "> Do not treat this run's deliverables as verified.",
    ]
    assert rest.startswith("- Status: **completed_with_issues**\n")
    assert describe_issues(index) == "2 acceptance criteria not met (AC-2, clean-room)"


def test_render_run_body_callout_for_both_kinds_of_issue(vault: Vault) -> None:
    index = make_index(
        vault,
        status=RunStatus.COMPLETED_WITH_ISSUES,
        unresolved_critical=3,
        criteria_unmet=1,
        unmet_criteria=["clean-room [unmet]: rebuild"],
        handoffs=["04-crosscheck-r3", "05-final"],
    )
    callout = render_run_body(index).split("## Status\n\n", 1)[1].split("\n\n", 1)[0]
    lines = callout.splitlines()
    assert lines[0] == "> [!warning] Completed with 3 unresolved critical issue(s) and 1 acceptance criterion not met"
    assert "[[04-crosscheck-r3]]" in lines[1] and lines[2].startswith("> 1 acceptance criterion not met")
    assert describe_issues(index) == (
        "3 unresolved critical issue(s) after the cross-check loop cap; 1 acceptance criterion not met (clean-room)"
    )
    assert describe_issues(make_index(vault)) == ""


def test_format_bytes() -> None:
    assert [format_bytes(n) for n in (0, 999, 1000, 1_234_567, 5_000_000_000)] == ["0 B", "999 B", "1.0 kB", "1.2 MB", "5.0 GB"]


@pytest.mark.parametrize(
    ("rel", "excluded"),
    [
        (".maf/prompt.md", True),
        ("sub/.claude/x", True),
        ("FreeRTOS-Kernel/tasks.c", True),
        ("inputs/spec.pdf", True),
        ("tests/inputs/case1.txt", False),  # inputs/* is the root folder only
        ("build/alloc.o", True),
        ("tests/build/x.o", True),
        ("build", False),  # a build script, not a build directory
        ("scripts/build", False),
        ("pkg.egg-info/PKG-INFO", True),
        ("src/__pycache__/a.pyc", True),
        ("src/mod.pyc", True),
        ("src/alloc.c", False),
        (".gitignore", False),
        ("prog.o", True),
        ("lib/libx.so", True),
        ("fw/test.elf", True),
        ("a.out", True),
        ("x/CMakeFiles/y", True),
        ("REPRO_EXIT", True),
        ("results/metrics.csv", False),
    ],
)
def test_default_export_excludes(rel: str, excluded: bool) -> None:
    assert export_excluded(rel, DEFAULT_EXPORT_EXCLUDES) is excluded


def _tree(root: Path, files: dict[str, str | bytes]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content)


def _listing(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}


def test_export_workspace_copies_the_tree_safely(vault: Vault, tmp_path: Path) -> None:
    paths = vault.create_run(make_index(vault), [])
    ws = paths.workspace
    outside = tmp_path / "secret.txt"
    outside.write_text("do not export")
    _tree(ws, {
        "README.md": "readme", "src/alloc.c": "int x;", "run.sh": "#!/bin/sh\n", "src/__init__.py": "",
        "web/package.json": "",  # empty but not at the root: a real file of the deliverable
        ".env": "", ".env.local": "", "package.json": "", ".npmrc": "registry=x",  # the last is not empty: kept
        ".maf/p.md": "p", "build/a.o": b"o", "empty-dir/.keep": "",
    })
    (ws / "run.sh").chmod(0o755)
    (ws / "empty-dir" / ".keep").unlink()
    (ws / "alias.c").symlink_to(ws / "src" / "alloc.c")
    (ws / "leak.txt").symlink_to(outside)
    (ws / "dangling").symlink_to(ws / "nowhere")
    (ws / "srclink").symlink_to(ws / "src", target_is_directory=True)
    os.mkfifo(ws / "pipe")

    result = vault.export_workspace(RUN_ID)

    got = _listing(paths.deliverables)
    assert got == {"README.md", "src/alloc.c", "run.sh", "src/__init__.py", "web/package.json", ".npmrc", "alias.c"}
    assert result.files == tuple(sorted(got))
    assert result.total_bytes == sum((paths.deliverables / rel).stat().st_size for rel in got)
    assert sorted(result.placeholders) == [".env", ".env.local", "package.json"]
    assert sorted(result.skipped) == [
        "dangling (dangling symlink)", "leak.txt (symlink out of the workspace)", "pipe (not a regular file)",
        "srclink/ (symlinked directory)",
    ]
    assert not (paths.deliverables / "alias.c").is_symlink()
    assert (paths.deliverables / "alias.c").read_text() == "int x;"
    assert os.access(paths.deliverables / "run.sh", os.X_OK)
    assert not (paths.deliverables / "empty-dir").exists()
    assert paths.deliverables.stat().st_mode & 0o777 != 0o700  # not the staging folder's private mode
    assert result.excluded == ("build/",)  # .maf/ is pipeline state, not reported
    assert result.describe() == f"7 file(s), {result.total_bytes} B; excluded: build/"


def test_symlinks_into_excluded_paths_are_not_exported(vault: Vault) -> None:
    """Lint and the source audit never read ``.maf/`` or ``inputs/``, so a link into them must not ship their content
    as a regular file."""
    paths = vault.create_run(make_index(vault), [])
    ws = paths.workspace
    _tree(ws, {".maf/execution-r1.md": "PROMPT: internal pipeline notes", "inputs/data.csv": "private", "a.md": "a"})
    (ws / "docs").mkdir()
    os.symlink("../.maf/execution-r1.md", ws / "docs" / "brief.md")
    os.symlink("../inputs/data.csv", ws / "docs" / "data.csv")
    os.symlink("../a.md", ws / "docs" / "a.md")
    result = vault.export_workspace(RUN_ID)
    assert _listing(paths.deliverables) == {"a.md", "docs/a.md"}
    assert sorted(result.skipped) == [
        "docs/brief.md (symlink into the excluded .maf/execution-r1.md)",
        "docs/data.csv (symlink into the excluded inputs/data.csv)",
    ]


def test_in_source_build_output_is_left_out_and_reported(vault: Vault) -> None:
    paths = vault.create_run(make_index(vault), [])
    _tree(paths.workspace, {
        "prog.c": "int main(void){return 0;}", "prog.o": b"\x7fELF", "a.out": b"\x7fELF", "lib/libx.a": b"!<arch>",
        "lib/x.h": "x", "fw/test.elf": b"\x7fELF", "cmake/CMakeFiles/x.o": b"o", "cmake/CMakeCache.txt": "c",
        "cmake/CMakeLists.txt": "project(x)", "src/__pycache__/m.pyc": b"\x00", "src/m.py": "",
        ".maf/p.md": "p", "inputs/spec.pdf": b"%PDF",
    })
    result = vault.export_workspace(RUN_ID)
    assert result.files == ("cmake/CMakeLists.txt", "lib/x.h", "prog.c", "src/m.py")
    assert result.excluded == (
        "a.out", "cmake/CMakeCache.txt", "cmake/CMakeFiles/", "fw/", "lib/libx.a", "prog.o", "src/__pycache__/",
    )
    assert result.describe().endswith(
        "excluded: a.out, cmake/CMakeCache.txt, cmake/CMakeFiles/, fw/, lib/libx.a and 2 more"
    )


def test_re_include_patterns_bring_back_build_files_but_never_pipeline_state(vault: Vault) -> None:
    """``build/*`` drops a hand-written ``build/`` too; a ``!`` pattern (``Settings.export_include``) restores it, and
    the protected patterns, repeated last, still win."""
    paths = vault.create_run(make_index(vault), [])
    _tree(paths.workspace, {
        "build/package/Dockerfile": "FROM x", "build/toolchain.cmake": "set(x)", "build/a.o": b"o",
        ".maf/p.md": "p", "inputs/i.md": "i",
    })
    patterns = (*DEFAULT_EXPORT_EXCLUDES, "!build/package/*", "!build/*.cmake", "!*.md", *PROTECTED_EXPORT_EXCLUDES)
    result = vault.export_workspace(RUN_ID, excludes=patterns)
    assert result.files == ("build/package/Dockerfile", "build/toolchain.cmake")
    assert result.excluded == ("build/a.o",)


def test_export_workspace_replaces_the_previous_tree_and_cleans_leftovers(vault: Vault) -> None:
    paths = vault.create_run(make_index(vault), [])
    _tree(paths.workspace, {"a.txt": "a"})
    _tree(paths.deliverables, {"stale/old.txt": "old"})
    leftover = paths.root / ".deliverables-crashed.tmp"
    _tree(leftover, {"half.txt": "h"})
    vault.export_workspace(RUN_ID, excludes=())
    assert _listing(paths.deliverables) == {"a.txt"}
    assert sorted(p.name for p in paths.root.iterdir() if p.name.startswith(".deliverables")) == []


def test_export_workspace_restores_the_old_tree_if_the_swap_fails(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = vault.create_run(make_index(vault), [])
    _tree(paths.workspace, {"new.txt": "n"})
    _tree(paths.deliverables, {"old.txt": "o"})
    real_replace = os.replace
    calls: list[str] = []

    def flaky(src: object, dst: object) -> None:
        calls.append(str(dst))
        if len(calls) == 2:
            raise OSError("disk trouble")
        real_replace(src, dst)  # type: ignore[arg-type]

    monkeypatch.setattr("maf.vault.os.replace", flaky)
    with pytest.raises(OSError, match="disk trouble"):
        vault.export_workspace(RUN_ID)
    monkeypatch.setattr("maf.vault.os.replace", real_replace)
    assert _listing(paths.deliverables) == {"old.txt"}
    assert not [p for p in paths.root.iterdir() if p.name.startswith(".deliverables")]


def test_export_workspace_caps_and_errors(vault: Vault) -> None:
    paths = vault.create_run(make_index(vault), [])
    _tree(paths.workspace, {"data/big.bin": b"x" * 5000, "src/a.c": "int a;", "README.md": "r"})
    _tree(paths.deliverables, {"keep.txt": "k"})
    with pytest.raises(ExportTooLarge) as too_big:
        vault.export_workspace(RUN_ID, max_bytes=4000)
    message = str(too_big.value)
    assert "would be 5.0 kB in 3 file(s), over the cap of 4.0 kB and 20000 files" in message
    assert "largest entries: data/ (5.0 kB), src/ (6 B), README.md (1 B)" in message
    with pytest.raises(ExportTooLarge, match="over the cap of .* and 2 files"):
        vault.export_workspace(RUN_ID, max_files=2)
    assert _listing(paths.deliverables) == {"keep.txt"}  # nothing was touched
    assert vault.export_workspace(RUN_ID, excludes=("data",), max_files=2).files == ("README.md", "src/a.c")

    with pytest.raises(FileNotFoundError):
        vault.export_workspace("2026-09-28-nope")
    shutil.rmtree(paths.workspace)
    with pytest.raises(ExportError, match="does not exist"):
        vault.export_workspace(RUN_ID)

