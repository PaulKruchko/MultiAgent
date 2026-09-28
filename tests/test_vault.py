"""Tests for maf.vault: naming, atomic writes, run lifecycle, run.md rendering and copy guards."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from maf.handoff import HandoffInvalid, HandoffKind, HandoffMeta, build_handoff
from maf.types import RunStatus
from maf.vault import (
    RunIndex,
    Vault,
    atomic_copy,
    atomic_write_text,
    embed,
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
    index.status = RunStatus.COMPLETED
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
