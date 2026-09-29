"""Deliverable linter: pipeline leakage and Markdown that renders wrong. Pure standard library, no maf imports.

Owner: stages.

Deliverables are read on their own, outside the pipeline. So they must not cite the pipeline's notes, talk about
its revisions and reviews, or ship Markdown that Obsidian renders wrong. ``lint_markdown`` checks one text;
``lint_deliverables`` walks the ``*.md`` files under a root (skipping ``exclude``) and also resolves local links.
Fenced code blocks, indented code blocks (four spaces or a tab after a blank line, outside a list) and inline code
are never linted, with one exception: GFM splits a table cell on a ``|`` even inside inline code, so the table column
count reads the raw row.

Rules (``LintIssue.rule``, severity):
- ``pipeline-wikilink`` (critical): a link or embed to a pipeline note (``[[01-ingestion]]``, ``[[03-execution-r2]]``,
  ``[text](02-strategy.md)``, ...) or any ``runs/...`` path, except an embedded asset file under a run's
  ``deliverables/`` or ``assets/`` (final rewrites image embeds to those vault paths) and, with a root, a target that
  resolves to a file of the deliverable tree itself (a chapter named ``01-ingestion.md``, a ``runs/`` output folder).
  Also a citation of the ingestion's internal source ids (``[S2]``, ``[S1, S3]``) in prose.
- ``meta-commentary`` (major): remarks about the pipeline's revisions and reviews ("the previous revision",
  "earlier revisions", "as proposed in review", "because review identified", "the cross-check found", "the
  ingestion's value"). Whole phrases, case-insensitive, chosen so that ordinary uses ("we cross-check the model
  against", "a critique of the Lawson criterion", "the critique is well founded", "a systematic review found") pass.
- ``gfm-table-pipe-in-math`` (major): a table row with an unescaped ``|`` inside ``$...$``, which splits the cell.
- ``gfm-table-columns`` (major): a table row, or header, whose cell count differs from the delimiter row.
- ``placeholder`` (major): ``{{name}}``, TODO, TBD, FIXME, XXX, "lorem ipsum", "[citation needed]", "[insert ...]".
- ``null-rendering`` (minor): n/a, None, nan, null or undefined as a cell of a numeric table column, or as a value
  in prose (after ``=``, ``:`` or in parentheses, on a line that has digits).
- ``unbalanced-math`` (major): an inline ``$`` that opens math and is never closed, or an unclosed ``$$``.
  A ``$`` followed by a digit with no closing ``$`` is currency, not math, and one followed by an upper-case
  identifier or ``{`` (``$PATH``, ``${var}``) is a shell variable, which Obsidian shows literally.
- ``broken-link`` (major; ``lint_deliverables`` or ``root=`` only): an embed, wikilink or relative Markdown link that
  resolves neither relative to the file, the root or a common asset directory (``ASSET_DIRS``), nor (wikilinks and
  embeds, as Obsidian does) by file name anywhere under the root. Targets outside the root or excluded from it, and
  absolute paths, count as broken: they do not survive a copy of the deliverable tree. A target the filesystem cannot
  even name (too long, an embedded NUL) is broken too, never a crash.
"""

from __future__ import annotations

import fnmatch
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import unquote

LintSeverity = Literal["critical", "major", "minor"]

BLOCKING: frozenset[str] = frozenset({"critical", "major"})
"""Severities that the execution note reports (``maf.stages.execution``); minor issues are only logged."""

DEFAULT_EXCLUDE: tuple[str, ...] = (
    ".maf",
    ".git",
    "inputs",
    "FreeRTOS-Kernel",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
)
"""Never linted: pipeline metadata, the user's own input files, third-party trees. See ``excluded``."""

ASSET_DIRS: tuple[str, ...] = ("assets", "plots", "figures", "figs", "images", "img", "media")
"""Directories (under the file's directory or the root) where a bare embed target is also looked up."""

MAX_FILE_BYTES = 5_000_000
"""Larger Markdown files are skipped: generated data, not prose."""


@dataclass(frozen=True)
class LintIssue:
    rule: str
    severity: LintSeverity
    path: str
    """Root-relative POSIX path (``lint_deliverables``) or the ``path`` given to ``lint_markdown``."""
    line: int
    """1-based."""
    message: str

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.path else f"line {self.line}"
        return f"[{self.severity}] {self.rule} {where}: {self.message}"


def blocking(issues: Iterable[LintIssue]) -> list[LintIssue]:
    """The critical and major issues, in order."""
    return [issue for issue in issues if issue.severity in BLOCKING]


def _names(rel: str) -> tuple[str, ...]:
    parts = PurePosixPath(rel).parts
    return (*parts, *("/".join(parts[: k + 1]) for k in range(len(parts))))


def _last_match(rel: str, patterns: tuple[str, ...]) -> int | None:
    names = _names(rel)
    found = None
    for k, pattern in enumerate(patterns):
        if any(fnmatch.fnmatchcase(name, pattern.removeprefix("!")) for name in names):
            found = k
    return found


def excluded(rel: str, patterns: Iterable[str]) -> bool:
    """True if ``patterns`` exclude the root-relative POSIX path ``rel``. A pattern (``fnmatch``, case-sensitive)
    matches when it matches a component of ``rel`` or a leading part of it: ``.git`` and ``*.pyc`` match at any depth,
    ``build/tmp`` only at the root (and then everything below it). As in a gitignore, the last matching pattern
    decides, and one starting with ``!`` re-includes what it matches (``build/*`` then ``!build/toolchain.cmake``)."""
    patterns = tuple(patterns)
    last = _last_match(rel, patterns)
    return last is not None and not patterns[last].startswith("!")


def excluded_dir(rel: str, patterns: Iterable[str]) -> bool:
    """Whether a walk may skip the directory ``rel`` and everything below it: it is ``excluded``, and no re-include
    pattern after the one that excludes it could match a path below it (the excluding pattern matches every such
    path too, so only a later ``!`` pattern can win)."""
    patterns = tuple(patterns)
    last = _last_match(rel, patterns)
    if last is None or patterns[last].startswith("!"):
        return False
    for pattern in patterns[last + 1 :]:
        if not pattern.startswith("!"):
            continue
        body = pattern[1:]
        if "/" not in body:
            return False  # a component pattern can match at any depth
        literal = re.split(r"[*?\[]", body, maxsplit=1)[0]
        if literal.startswith(f"{rel}/") or f"{rel}/".startswith(literal):
            return False
    return True


def lint_markdown(text: str, path: str = "", *, root: Path | None = None) -> list[LintIssue]:
    """Lint one Markdown text. ``path`` labels the issues; with ``root`` (the deliverable tree containing the file at
    ``root/path``) local links and embeds are resolved too."""
    resolver = _Resolver(root, DEFAULT_EXCLUDE) if root is not None else None
    return _lint(text, path, resolver)


def lint_deliverables(root: Path, exclude: Iterable[str] = DEFAULT_EXCLUDE) -> list[LintIssue]:
    """Lint every ``*.md`` file under ``root`` whose path is not ``excluded``, sorted by path and line. Symlinks are
    not followed; files over ``MAX_FILE_BYTES`` are skipped. A missing root has no issues."""
    root = Path(root)
    if not root.is_dir():
        return []
    patterns = tuple(exclude)
    resolver = _Resolver(root, patterns)
    issues: list[LintIssue] = []
    for rel in resolver.files:
        if not rel.lower().endswith(".md"):
            continue
        file = root / rel
        if file.stat().st_size > MAX_FILE_BYTES:
            continue
        issues += _lint(file.read_text(encoding="utf-8", errors="replace"), rel, resolver)
    return sorted(issues, key=lambda i: (i.path, i.line))


# ---------------------------------------------------------------------------
# Scanning: classify lines, mask inline code and math.

LineKind = Literal["text", "code", "math", "front"]


@dataclass
class _Line:
    no: int
    raw: str
    kind: LineKind
    visible: str = ""
    """``raw`` with inline code spans blanked (same length)."""
    prose: str = ""
    """``visible`` with inline math blanked too."""
    code: list[tuple[int, int]] = field(default_factory=list)
    math: list[tuple[int, int]] = field(default_factory=list)


_FENCE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_INDENTED = re.compile(r"^(?: {4}| {0,3}\t)")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d{1,9}[.)])(?:\s|$)")
_BACKTICKS = re.compile(r"`+")
_HEADING = re.compile(r"^ {0,3}#{1,6}(?:\s|$)")
_BLOCK_START = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|>)")
_PIPE = re.compile(r"(?<!\\)\|")
_QUOTE_PREFIX = re.compile(r"^\s*(?:>\s?)*")


@dataclass
class _Scan:
    lines: list[_Line]
    issues: list[LintIssue]


def _scan(text: str, path: str) -> _Scan:
    raws = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    front = _frontmatter_end(raws)
    lines: list[_Line] = []
    issues: list[LintIssue] = []
    fence: str | None = None
    display: int | None = None  # line number of an open $$ block
    blank_before = True  # the document start counts as a blank line
    in_list = False  # the last block is a list: an indented line continues an item instead of opening code
    indented = False  # inside an indented code block
    for i, raw in enumerate(raws):
        kind: LineKind = "text"
        in_block = False  # this line belongs to an indented code block
        body = _QUOTE_PREFIX.sub("", raw)  # fences and $$ blocks also open inside blockquotes and callouts
        stripped = body.strip()
        if i < front:
            kind = "front"
        elif fence is not None:
            kind = "code"
            if stripped and set(stripped) == {fence[0]} and len(stripped) >= len(fence):
                fence = None
        elif display is not None:
            kind = "math"
            if stripped.endswith("$$"):
                display = None
        elif raw.strip() and _INDENTED.match(raw) and (indented or (blank_before and not in_list)):
            kind = "code"  # an indented code block: it cannot interrupt a paragraph or a list item
            in_block = True
        elif (m := _FENCE.match(body)) and not (m["fence"][0] == "`" and "`" in m["info"]):
            fence, kind = m["fence"], "code"
        elif stripped.startswith("$$"):
            kind = "math"
            if not (len(stripped) >= 4 and stripped.endswith("$$")):
                display = i + 1
        if kind == "front" or not raw.strip():
            blank_before = True
        else:
            indented = in_block  # blank lines keep an indented block open; any other line ends it
            if kind == "text":
                if _LIST_ITEM.match(body):
                    in_list = True
                elif _HEADING.match(raw) or (blank_before and not raw[:1].isspace()):
                    in_list = False
            blank_before = False
        line = _Line(i + 1, raw, kind, visible=raw, prose=raw)
        if kind == "text":
            line.visible, line.code = _mask_code(raw)
            line.prose = line.visible
        lines.append(line)
    if display is not None:
        issues.append(LintIssue("unbalanced-math", "major", path, display, "`$$` display math is never closed"))
    issues += _mask_math(lines, path)
    return _Scan(lines, issues)


def _frontmatter_end(raws: list[str]) -> int:
    if not raws or raws[0].strip() != "---":
        return 0
    for i in range(1, len(raws)):
        if raws[i].strip() in ("---", "..."):
            return i + 1
    return 0


def _mask_code(line: str) -> tuple[str, list[tuple[int, int]]]:
    """Blank inline code spans (a backtick run closed by a run of the same length on the same line)."""
    runs = list(_BACKTICKS.finditer(line))
    spans: list[tuple[int, int]] = []
    j = 0
    while j < len(runs):
        width = len(runs[j].group())
        close = next((k for k in range(j + 1, len(runs)) if len(runs[k].group()) == width), None)
        if close is None:
            j += 1
            continue
        spans.append((runs[j].start(), runs[close].end()))
        j = close + 1
    return _blank(line, spans), spans


def _blank(text: str, spans: Iterable[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for p in range(start, end):
            if chars[p] != "\n":
                chars[p] = " "
    return "".join(chars)


def _units(lines: list[_Line]) -> list[list[_Line]]:
    """Groups of text lines that inline math may span: a paragraph or one list item. Headings and lines with a table
    pipe stand alone; blank lines and code/math/frontmatter lines end a group."""
    units: list[list[_Line]] = []
    current: list[_Line] | None = None
    for line in lines:
        if line.kind != "text" or not line.raw.strip():
            current = None
            continue
        single = bool(_HEADING.match(line.raw) or _PIPE.search(line.visible))
        if single or current is None or _BLOCK_START.match(line.raw):
            current = [line]
            units.append(current)
            if single:
                current = None
        else:
            current.append(line)
    return units


def _mask_math(lines: list[_Line], path: str) -> list[LintIssue]:
    """Find inline math in each unit, record it per line, blank it in ``prose``; report unmatched delimiters."""
    issues: list[LintIssue] = []
    for unit in _units(lines):
        joined = "\n".join(line.visible for line in unit)
        starts = [0]
        for line in unit[:-1]:
            starts.append(starts[-1] + len(line.visible) + 1)
        spans, unmatched = _math_spans(joined)
        for offset, token in unmatched:
            k = max(i for i, s in enumerate(starts) if s <= offset)
            what = "`$$` display math" if token == "$$" else "inline math `$`"
            issues.append(LintIssue("unbalanced-math", "major", path, unit[k].no, f"{what} is opened but never closed"))
        for k, line in enumerate(unit):
            lo, hi = starts[k], starts[k] + len(line.visible)
            line.math = [(max(s, lo) - lo, min(e, hi) - lo) for s, e in spans if s < hi and e > lo]
            line.prose = _blank(line.visible, line.math)
    return issues


def _math_spans(s: str) -> tuple[list[tuple[int, int]], list[tuple[int, str]]]:
    """Inline math spans ``[start, end)`` in ``s`` and unmatched openers ``(offset, "$" or "$$")``.

    Obsidian's rules: an opening ``$`` has a non-space right after it, a closing ``$`` a non-space right before it;
    ``\\$`` is a literal dollar. An unmatched ``$`` followed by a digit is currency, and one followed by an upper-case
    identifier or ``{`` is a shell variable (``_SHELL_VARIABLE``). (Pandoc also refuses a closing
    ``$`` followed by a digit; renderers write ``$T=$5 keV``, which Obsidian shows as intended.)"""
    spans: list[tuple[int, int]] = []
    unmatched: list[tuple[int, str]] = []
    n = len(s)
    i = 0
    while i < n:
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c != "$":
            i += 1
            continue
        if s.startswith("$$", i):
            close = _find_unescaped(s, "$$", i + 2)
            if close is None:
                unmatched.append((i, "$$"))
                i += 2
            else:
                spans.append((i, close + 2))
                i = close + 2
            continue
        nxt = s[i + 1] if i + 1 < n else ""
        if not nxt or nxt.isspace():
            i += 1
            continue
        j, close = i + 1, None
        while j < n:
            if s[j] == "\\":
                j += 2
                continue
            if s[j] == "$" and not s[j - 1].isspace():
                close = j
                break
            j += 1
        if close is None:
            if not nxt.isdigit() and not _SHELL_VARIABLE.match(s, i + 1):
                unmatched.append((i, "$"))
            i += 1
            continue
        spans.append((i, close + 1))
        i = close + 1
    return spans, unmatched


_SHELL_VARIABLE = re.compile(r"\{|[A-Z_][A-Z0-9_]*\b")
"""What follows the ``$`` of a shell variable (``$PATH``, ``$CC_FLAGS``, ``${var}``): unclosed, it is literal text."""


def _find_unescaped(s: str, token: str, start: int) -> int | None:
    i = start
    while (i := s.find(token, i)) != -1:
        if i == 0 or s[i - 1] != "\\":
            return i
        i += 1
    return None


# ---------------------------------------------------------------------------
# Rules

TEMPLATE_NAMES: tuple[str, ...] = ("*template*", "*.tmpl.md", "*.j2.md", "*.jinja.md", "*.jinja2.md")
"""File names (``fnmatch``, case-insensitive) linted as templates: their ``{{ field }}`` and ``{% tag %}`` markup is
replaced by zeros of the same length first (a field usually renders a number, and blanks would unbalance ``$x={{v}}$``),
so fields are not placeholders and filter pipes (``{{ x|round }}``) do not split table cells."""

_TEMPLATE_MARKUP = re.compile(r"\{\{.*?\}\}|\{%.*?%\}")


def _lint(text: str, path: str, resolver: _Resolver | None) -> list[LintIssue]:
    name = PurePosixPath(path).name.lower()
    if any(fnmatch.fnmatchcase(name, p) for p in TEMPLATE_NAMES):
        text = _TEMPLATE_MARKUP.sub(lambda m: "0" * len(m.group()), text)
    scan = _scan(text, path)
    issues = list(scan.issues)
    issues += _links(scan.lines, path, resolver)
    issues += _source_ids(scan.lines, path)
    issues += _meta_commentary(scan.lines, path)
    issues += _tables(scan.lines, path)
    issues += _placeholders(scan.lines, path)
    issues += _null_values(scan.lines, path)
    return sorted(issues, key=lambda i: i.line)


def _snippet(text: str, limit: int = 60) -> str:
    """``text`` as an inline-code span for a message: never a live link in a vault note, never a broken span."""
    flat = " ".join(text.replace("`", "'").split())
    return f"`{flat if len(flat) <= limit else flat[: limit - 3] + '...'}`"


_PIPELINE_NOTE = re.compile(
    r"(?:01-ingestion|01a-routing|02-strategy|03-execution|04-crosscheck|04a-critique-(?:chatgpt|gemini|claude)"
    r"|04b-rebuttal|04c-adjudication|05-final)(?:-r\d+)?",
    re.IGNORECASE,
)
"""Note names of ``maf.vault.note_name``, matched against a link target's file name (without ``.md``)."""

_WIKILINK = re.compile(r"(?P<embed>!?)\[\[(?P<target>[^\[\]\n]*)\]\]")
_MDLINK = re.compile(
    r"(?P<embed>!?)\[(?P<text>[^\]\n]*)\]\(\s*(?P<target><[^>\n]*>|[^)\s]*)"
    r"(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^)\n]*\)))?\s*\)"
)
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_ASSET_SUFFIX = re.compile(r"\.(?!md$)[A-Za-z0-9]{1,5}$", re.IGNORECASE)


def _wikilink_target(inner: str) -> str:
    target = inner.split("|", 1)[0].rstrip("\\")
    return re.split(r"[#^]", target, maxsplit=1)[0].strip()


def _mdlink_target(target: str) -> str:
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    return unquote(re.split(r"[#?]", target, maxsplit=1)[0].strip())


def _is_pipeline_target(target: str, embed: bool) -> bool:
    parts = [p for p in target.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    if not parts:
        return False
    if parts[0] == "runs":
        asset = embed and _ASSET_SUFFIX.search(parts[-1]) and ("deliverables" in parts or "assets" in parts)
        return not asset
    name = parts[-1][:-3] if parts[-1].lower().endswith(".md") else parts[-1]
    return _PIPELINE_NOTE.fullmatch(name) is not None


def _links(lines: list[_Line], path: str, resolver: _Resolver | None) -> list[LintIssue]:
    issues: list[LintIssue] = []
    for line in lines:
        if line.kind not in ("text", "front"):
            continue
        text = line.visible  # not ``prose``: a stray ``$5 ... $x$`` pairing must not hide a link
        found: list[tuple[str, str, bool, bool]] = []  # (shown, target, embed, wikilink)
        for m in _WIKILINK.finditer(text):
            found.append((m.group(), _wikilink_target(m["target"]), bool(m["embed"]), True))
        for m in _MDLINK.finditer(_WIKILINK.sub(lambda w: " " * len(w.group()), text)):
            raw_target = m["target"]
            if _SCHEME.match(raw_target) or raw_target.startswith("#"):
                continue
            found.append((m.group(), _mdlink_target(raw_target), bool(m["embed"]), False))
        for shown, target, embed, wikilink in found:
            if not target:
                continue
            if _is_pipeline_target(target, embed) and not (  # pipeline notes never live in the tree itself
                resolver is not None and resolver.in_tree(target, path, by_name=wikilink)
            ):
                issues.append(LintIssue(
                    "pipeline-wikilink", "critical", path, line.no,
                    f"link to the internal pipeline note {_snippet(shown)}; a deliverable must stand alone, so cite "
                    "the original source instead",
                ))
            elif resolver is not None and line.kind == "text":
                problem = resolver.problem(target, path, by_name=wikilink)
                if problem:
                    kind = "embed" if embed else "link"
                    message = f"{kind} {_snippet(shown)}: {problem}"
                    issues.append(LintIssue("broken-link", "major", path, line.no, message))
    return issues


_META_PATTERNS: tuple[str, ...] = (
    # the deliverable's own revisions and drafts (not a file's revision in version control, nor a standard's)
    r"\b(?:previous|earlier|prior|last|this|former) revisions?\b"
    r"(?!\s+(?:of|in)\s+(?:the\s+|this\s+|that\s+)?(?:file|repo(?:sitory)?|standard|specification|spec|IEEE|ISO|RFC)\b"
    r"|\s+(?:number|control|history|id|hash)\b)",
    r"\b(?:previous|earlier|prior|first|former) (?:drafts?|versions?|iterations?) of (?:this|the) "
    r"(?:document|thesis|report|chapter|section|readme|code|implementation|analysis|paper|text)\b",
    # the pipeline's rounds and passes (nouns only: "the fix passes all tests" is a verb)
    r"\b(?:previous|earlier|prior|first|second|last|next) (?:fix|review|cross-?check|critique) "
    r"(?:round|pass|cycle|iteration)s?\b",
    r"\b(?:fix|execution)(?:-| )(?:rounds?|pass)\b",
    r"\b(?:previous|earlier|later|several|multiple|both|all|these|those|two|three) (?:fix|execution)(?:-| )passes\b",
    r"\b(?:cross-?check|critique|adjudication) (?:round|stage|pass|cycle|issue|finding|note|verdict|ruling)s?\b",
    r"\breview (?:round|pass|cycle|verdict|ruling)s?\b",
    # reviews as the reason for a change
    r"\b(?:the )?cross-?checks? (?:found|flagged|noted|raised|reported|revealed|requested|required|asked|identified"
    r"|pointed out)\b",
    r"\bas (?:proposed|suggested|requested|noted|flagged|pointed out|raised|required|recommended) (?:in|by|during) "
    r"(?:the )?(?:reviews?|reviewers?|critiques?|critics?|cross-?checks?|adjudicat\w*)\b",
    r"\breview pointed out\b",
    r"(?:^|[.!?;:]\s+|\b(?:because|since|after|when|as|once|until|where|and|but|so|that|then)\s+)review "
    r"(?:identified|found|showed|revealed|noted|flagged|raised|requested|asked)\b",
    r"\b(?:after|following) (?:the )?(?:review|critique|cross-?check)s?\b(?!\s+(?:of|by|in)\b)",
    r"\breviewers? (?:pointed out|noted|requested|suggested|flagged|asked|raised|found|objected)\b",
    r"\bthe reviewers? (?:of this|wanted|required)\b",
    r"\bin response to (?:the )?(?:reviews?|reviewers?|critiques?|critics?|feedback|cross-?checks?)\b",
    r"\bper (?:the )?(?:review|reviewers?|critique|adjudicat\w*|ruling)\b",
    r"\bthe critiques? (?:raised|noted|found)\b(?!\s+(?:in|by)\b)",
    r"\bthe critiques? (?:asked|requested|pointed out|flagged|identified|suggested|required)\b",
    r"\bcritics? (?:noted|raised|flagged|pointed out|found|objected|requested)\b",
    r"\bthe adjudicator\b",
    # the pipeline's acceptance criteria and notes
    r"\b(?:hard|soft) acceptance criteri(?:on|a)\b",
    r"\bacceptance criteri(?:on|a)\b[^.\n]{0,40}?\b(?:AC-\d+|strategy|brief)\b",
    r"\bcriteri(?:on|a) AC-\d+\b",
    r"\b(?:ingestion|strategy|routing|execution|cross-?check) (?:note|handoff)s?\b",
    r"\b(?:ingestion|routing) (?:report|brief)\b",
    r"\bthe ingestion\b(?!\s+(?:of|rate|rates)\b)",
    r"\bexecution brief\b",
)
_META = re.compile("|".join(f"(?:{p})" for p in _META_PATTERNS), re.IGNORECASE)


_SOURCE_ID = re.compile(r"(?<![\[\w])\[S\d{1,4}(?:\s*[,;\u2013-]\s*S?\d{1,4})*\]")
"""A citation of the ingestion's internal source ids (``[S2]``, ``[S1, S3]``, ``[S1-S4]``)."""


def _source_ids(lines: list[_Line], path: str) -> list[LintIssue]:
    """``[S<n>]`` citations in prose (code and math excluded): ids of the ingestion's ``## Sources``, which is not
    shipped, so the citation points nowhere. Reported as ``pipeline-wikilink``, the same leak as a link to the note."""
    issues: list[LintIssue] = []
    for line in lines:
        if line.kind != "text":
            continue
        for m in _SOURCE_ID.finditer(line.prose):
            issues.append(LintIssue(
                "pipeline-wikilink", "critical", path, line.no,
                f"internal source id {_snippet(m.group())} of the ingestion report; a deliverable must stand alone, "
                "so cite the work by its bibliographic record instead",
            ))
    return issues


def _meta_commentary(lines: list[_Line], path: str) -> list[LintIssue]:
    issues: list[LintIssue] = []
    for line in lines:
        if line.kind == "text" and (m := _META.search(line.visible)):
            issues.append(LintIssue(
                "meta-commentary", "major", path, line.no,
                f"pipeline meta-commentary {_snippet(m.group())}; write the deliverable as a finished work and record "
                "changes in the execution note instead",
            ))
    return issues


_PLACEHOLDER = re.compile(
    r"\{\{\s*(?:[A-Za-z_][^{}\n]*)?\}\}|\b(?:TODO|TBD|FIXME|XXX)\b"
    r"|(?i:lorem ipsum|\[citation needed\]|\[insert\b[^\]\n]*\]|<insert\b[^>\n]*>)"
)


def _placeholders(lines: list[_Line], path: str) -> list[LintIssue]:
    """Math included (``$\\tau={{tau}}$`` is a leftover field), code excluded. ``{{2}}`` and ``{{\\rm d}}`` are
    LaTeX, not fields."""
    issues: list[LintIssue] = []
    for line in lines:
        if line.kind in ("text", "math") and (m := _PLACEHOLDER.search(line.visible)):
            message = f"unfilled placeholder {_snippet(m.group())}"
            issues.append(LintIssue("placeholder", "major", path, line.no, message))
    return issues


NULL_TOKENS = frozenset({"n/a", "nan", "-nan", "null", "undefined"})
"""What a renderer writes for a missing value (compared case-insensitively), besides Python's ``None`` (compared
exactly: a lowercase "none" is usually meant)."""

_NULL = r"(?:(?i:n/a|-?nan|null|undefined)|None)"
_NULL_VALUE = re.compile(
    rf"(?:[=:≈]\s*(?P<a>{_NULL})\b(?!/|\s+(?i:of|the|a|an|is|are|was|were)\b)|\(\s*(?P<b>{_NULL})\s*\))"
)


_DIGIT = re.compile(r"\d")


def _is_null(cell: str) -> bool:
    token = cell.strip("*_` ")
    return token == "None" or token.lower() in NULL_TOKENS


def _null_values(lines: list[_Line], path: str) -> list[LintIssue]:
    """Prose only (table cells are checked by ``_tables``)."""
    issues: list[LintIssue] = []
    for line in lines:
        if line.kind != "text" or _PIPE.search(line.visible) or not _DIGIT.search(line.prose):
            continue
        if m := _NULL_VALUE.search(line.prose):
            token = m["a"] or m["b"]
            issues.append(LintIssue(
                "null-rendering", "minor", path, line.no,
                f"{_snippet(token)} where a number belongs; a renderer should fail on a missing value instead",
            ))
    return issues


_DELIMITER = re.compile(r"^\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?$")


def _cells(row: str) -> list[str]:
    s = _QUOTE_PREFIX.sub("", row).strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [cell.strip() for cell in _PIPE.split(s)]


def _is_delimiter(line: _Line) -> bool:
    s = _QUOTE_PREFIX.sub("", line.raw).strip()
    return line.kind == "text" and "|" in s and "-" in s and _DELIMITER.match(s) is not None


def _pipes_in(line: _Line, spans: list[tuple[int, int]]) -> bool:
    """True if an unescaped ``|`` of the row lies inside one of ``spans``."""
    return any(s < m.start() < e for m in _PIPE.finditer(line.raw) for s, e in spans)


def _tables(lines: list[_Line], path: str) -> list[LintIssue]:
    issues: list[LintIssue] = []
    i = 0
    while i + 1 < len(lines):
        header, delimiter = lines[i], lines[i + 1]
        if header.kind != "text" or not _PIPE.search(header.raw) or not _is_delimiter(delimiter):
            i += 1
            continue
        width = len(_cells(delimiter.raw))
        rows = [header]
        j = i + 2
        while j < len(lines) and lines[j].kind == "text" and lines[j].raw.strip() and _PIPE.search(lines[j].raw):
            rows.append(lines[j])
            j += 1
        issues += [issue for row in rows if (issue := _row_issue(row, row is header, width, path))]
        issues += _null_cells(header, rows[1:], width, path)
        i = j
    return issues


def _row_issue(row: _Line, is_header: bool, width: int, path: str) -> LintIssue | None:
    """A cell-count mismatch, blamed on math when a ``|`` lies inside ``$...$``. Math spans are found across the
    whole row, so with an unbalanced ``$`` one can straddle a cell boundary; a row of the right width is therefore
    never blamed on math."""
    count = len(_cells(row.raw))
    if count == width:
        return None
    what = "header" if is_header else "row"
    effect = "; the table does not render" if is_header else ""
    if _pipes_in(row, row.math):
        return LintIssue(
            "gfm-table-pipe-in-math", "major", path, row.no,
            f"a `|` inside `$...$` splits this table {what} into {count} cells (the table has {width}){effect}; "
            "write `\\lvert x \\rvert`, `\\mid` or `\\vert` instead",
        )
    hint = "; a `|` inside inline code still splits the cell, so escape it as `\\|`" if _pipes_in(row, row.code) else ""
    message = f"table {what} has {count} cells but the table has {width}{effect}{hint}"
    return LintIssue("gfm-table-columns", "major", path, row.no, message)


def _null_cells(header: _Line, rows: list[_Line], width: int, path: str) -> list[LintIssue]:
    grid = [_cells(r.raw) for r in rows]
    names = _cells(header.raw)
    issues: list[LintIssue] = []
    for r, cells in enumerate(grid):
        if len(cells) != width:
            continue
        for c, cell in enumerate(cells):
            if not _is_null(cell):
                continue
            token = cell.strip("*_` ")
            numeric = any(
                len(other) == width and _DIGIT.search(other[c]) and not _is_null(other[c])
                for k, other in enumerate(grid) if k != r
            )
            if numeric:
                column = names[c] if c < len(names) and names[c] else f"column {c + 1}"
                issues.append(LintIssue(
                    "null-rendering", "minor", path, rows[r].no,
                    f"{_snippet(token)} in the numeric column {_snippet(column)}; a renderer should fail on a missing "
                    "value instead",
                ))
    return issues


# ---------------------------------------------------------------------------
# Link resolution

class _Resolver:
    """Resolves link targets inside a deliverable tree (``root``), ignoring ``excluded`` paths."""

    def __init__(self, root: Path, exclude: Iterable[str]) -> None:
        self.root = Path(root).resolve()
        self.exclude = tuple(exclude)
        self.files = sorted(_walk(self.root, self.exclude))
        self.names: set[str] = {PurePosixPath(rel).name.lower() for rel in self.files}

    def _inside(self, candidate: Path) -> bool:
        try:
            resolved = candidate.resolve()
        except (OSError, ValueError):
            return False
        if not resolved.is_relative_to(self.root) or not _exists(resolved):
            return False
        rel = resolved.relative_to(self.root).as_posix()
        return rel == "." or not excluded(rel, self.exclude)

    def _resolves(self, target: str, path: str, *, by_name: bool, vault: bool) -> bool:
        base = (self.root / path).parent
        names = [target] if not by_name or _ASSET_SUFFIX.search(target) or target.lower().endswith(".md") else [
            target, f"{target}.md"
        ]
        for name in names:
            candidates = [base / name, self.root / name]
            candidates += [d / a / name for d in (base, self.root) for a in ASSET_DIRS]
            if any(self._inside(c) for c in candidates):
                return True
            if by_name and PurePosixPath(name).name.lower() in self.names:
                return True
            if vault and name.startswith("runs/") and any(_exists(a / name) for a in self.root.parents):
                return True
        return False

    @staticmethod
    def _absolute(target: str) -> bool:
        return target.startswith(("/", "~")) or re.match(r"^[A-Za-z]:[\\/]", target) is not None

    def problem(self, target: str, path: str, *, by_name: bool) -> str | None:
        """None if ``target`` (linked from the root-relative file ``path``) resolves, else why not. A vault path
        (``runs/...``) also resolves against the root's parents: final lints the export inside the vault."""
        if self._absolute(target):
            return "absolute paths do not resolve in a copy of the deliverables"
        if self._resolves(target, path, by_name=by_name, vault=True):
            return None
        return "not found in the deliverable tree"

    def in_tree(self, target: str, path: str, *, by_name: bool) -> bool:
        """Whether ``target`` resolves to a file of the tree itself (never through the vault's ``runs/``)."""
        return not self._absolute(target) and self._resolves(target, path, by_name=by_name, vault=False)


def _exists(path: Path) -> bool:
    """``path.exists()``, but False for a name the filesystem cannot hold (a component over 255 bytes raises
    ``ENAMETOOLONG``, an embedded NUL ``ValueError``): such a link target is broken, not a crash."""
    try:
        return path.exists()
    except (OSError, ValueError):
        return False


def _walk(root: Path, exclude: tuple[str, ...]) -> Iterator[str]:
    """Root-relative POSIX paths of the regular files under ``root`` that are not ``excluded`` (no symlinks)."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        prefix = "" if rel_dir == "." else f"{rel_dir}/"
        dirnames[:] = sorted(d for d in dirnames if not excluded_dir(prefix + d, exclude))
        for name in filenames:
            rel = prefix + name
            if not excluded(rel, exclude) and not os.path.islink(os.path.join(dirpath, name)):
                yield rel
