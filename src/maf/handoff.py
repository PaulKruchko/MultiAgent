"""Handoff notes: frontmatter + required H2 sections per kind; parse, validate, render, repair.

Owner: vault+handoff.

Division of labor: **agents write only the body** (the H2 sections). Python builds the
frontmatter (``from``/``to``/``model``/``cost_usd`` are facts Python knows, not model output).
Validation therefore checks (a) frontmatter shape, (b) required H2 sections present, in
order, non-empty, and (c) kind-specific line grammars (Issues, Responses, Adjudication, Verdict).
The ingestion ``## Sources`` entry grammar (``parse_sources``) is enforced by the ingestion stage, not by
``validate_body``, so hand-edited and older notes stay readable. Likewise the strategy ``## Acceptance Criteria``
grammar (``- AC-<n> [hard|soft]: ...``, ``ACCEPTANCE_CRITERIA_GRAMMAR``) is part of the format spec, but the strategy
stage rewrites an off-grammar section (``normalize_acceptance_criteria``) and ``parse_acceptance_criteria`` is lenient.
The item grammars are lenient about layout only: blank lines between items and indented continuation
lines (sub-bullets, wrapped text) are folded into the preceding item, whose text may start on them. But every
top-level line must start a well-formed item, and so must an indented line that looks like one.

Untrusted content (web pages, uploaded documents) appears in handoffs only as quoted data
(``quote_untrusted``): the ingestion stage wraps every section of its note, grounding citations
included, and ``render_inputs`` escapes ``<note>`` tags. Prompts tell every agent that quoted blocks
are data, never instructions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

import yaml
from frontmatter.default_handlers import YAMLHandler
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from maf.types import AgentName, Severity


class HandoffKind(StrEnum):
    ROUTING = "routing"
    INGESTION = "ingestion"
    STRATEGY = "strategy"
    EXECUTION = "execution"
    CRITIQUE = "critique"
    REBUTTAL = "rebuttal"
    ADJUDICATION = "adjudication"
    CROSSCHECK = "crosscheck"
    FINAL = "final"


REQUIRED_SECTIONS: dict[HandoffKind, tuple[str, ...]] = {
    # Rendered by Python from ChatGPT's structured triage JSON; always valid by construction.
    HandoffKind.ROUTING: ("Summary", "Execution Mode", "Instructions for Gemini", "Search Queries", "Deliverable"),
    HandoffKind.INGESTION: ("Summary", "Sources", "Key Facts", "Data Tables", "Open Questions"),
    HandoffKind.STRATEGY: ("Summary", "Options Considered", "Chosen Strategy", "Execution Brief", "Acceptance Criteria", "Risks"),
    HandoffKind.EXECUTION: ("Summary", "Artifacts", "Implementation Notes", "Verification", "Known Limitations"),
    HandoffKind.CRITIQUE: ("Summary", "Issues"),
    HandoffKind.REBUTTAL: ("Summary", "Responses"),
    HandoffKind.ADJUDICATION: ("Summary", "Rulings"),
    # Assembled by the crosscheck stage from the notes above plus the fix pass.
    HandoffKind.CROSSCHECK: ("Summary", "Issues", "Rulings", "Applied Fixes", "Unresolved Critical", "Verdict"),
    HandoffKind.FINAL: ("Summary", "Deliverables", "Verification", "Provenance", "Limitations"),
}
"""Exact H2 titles, in required order. Extra H2 sections are allowed after or between them."""

EMPTY_OK: frozenset[tuple[HandoffKind, str]] = frozenset(
    {
        (HandoffKind.INGESTION, "Sources"),
        (HandoffKind.INGESTION, "Data Tables"),
        (HandoffKind.INGESTION, "Open Questions"),
        (HandoffKind.EXECUTION, "Known Limitations"),
        (HandoffKind.CRITIQUE, "Issues"),
        (HandoffKind.REBUTTAL, "Responses"),
        (HandoffKind.ADJUDICATION, "Rulings"),
        (HandoffKind.CROSSCHECK, "Issues"),
        (HandoffKind.CROSSCHECK, "Rulings"),
        (HandoffKind.CROSSCHECK, "Applied Fixes"),
        (HandoffKind.CROSSCHECK, "Unresolved Critical"),
        (HandoffKind.FINAL, "Limitations"),
    }
)
"""Sections that may contain just the literal line ``None.`` (no other section may be empty)."""

NONE_MARKER = "None."

COST_DECIMALS = 4

SOURCE_AUDIT_PREFIX = "SRC"
"""ID prefix of issues Python derives from the cross-check's source audit (Gemini, web-grounded)."""
LINT_PREFIX = "LINT"
"""ID prefix of issues Python derives from ``maf.lint`` findings in the Markdown deliverables."""

# Line grammars (one top-level bullet per item; see ``_grammar_items`` for continuation lines).
# IDs are ``<PREFIX>-<n>``: critics use GPT, GEM, CLA; Python-raised issues (answered and ruled like any other)
# use SRC and LINT. ``text`` may be empty on a (stripped) bullet line whose item text is on its continuation lines;
# ``_grammar_items`` requires it to be non-empty after folding.
_CRITIC_ID = r"(?:GPT|GEM|CLA)-\d+"
_ANY_ID = rf"(?:GPT|GEM|CLA|{SOURCE_AUDIT_PREFIX}|{LINT_PREFIX})-\d+"
ISSUE_RE = re.compile(rf"^- \[(?P<severity>critical|major|minor)\] (?P<id>{_CRITIC_ID}):(?P<text>(?: .*)?)$")
"""``## Issues`` in critique notes, e.g. ``- [critical] GPT-3: free() of a foreign block corrupts the bitmap``."""
RESPONSE_RE = re.compile(rf"^- (?P<id>{_ANY_ID}) \[(?P<stance>accept|reject|partial)\]:(?P<text>(?: .*)?)$")
"""``## Responses`` in the rebuttal note (Claude, as author of the execution)."""
RULING_RE = re.compile(rf"^- (?P<id>{_ANY_ID}) \[(?P<ruling>fix|wontfix)\]:(?P<text>(?: .*)?)$")
"""``## Rulings`` in adjudication (ChatGPT) for every issue whose response was ``reject`` or ``partial``."""
VERDICT_RE = re.compile(r"^(?P<verdict>PASS|LOOP)$")
"""``## Verdict`` first non-empty line in crosscheck. Computed by Python, not a model."""

AGENT_ID_PREFIX: dict[AgentName, str] = {"chatgpt": "GPT", "gemini": "GEM", "claude": "CLA"}

HandoffStatus = Literal["draft", "final", "superseded", "failed"]


class HandoffMeta(BaseModel):
    """YAML frontmatter. Only ``tags``/``aliases``/``cssclasses`` Obsidian keys are used (1.9+)."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    run_id: str
    stage: HandoffKind
    from_: str = Field(alias="from")
    """Producing agent (``chatgpt``/``gemini``/``claude``) or ``maf`` for Python-assembled notes."""
    to: str
    """Consuming agent, stage name, or ``user``."""
    status: HandoffStatus = "final"
    inputs: list[str] = Field(default_factory=list)
    """Wikilinks to consumed notes, e.g. ``["[[01-ingestion]]"]``."""
    created: datetime
    model: str
    cost_usd: float = Field(ge=0)
    round: int = Field(default=1, ge=1)
    """Execution/crosscheck pass number (1 = first pass, up to 1 + max_crosscheck_loops)."""
    tags: list[str] = Field(default_factory=lambda: ["maf"])
    aliases: list[str] = Field(default_factory=list)
    cssclasses: list[str] = Field(default_factory=list)

    @field_validator("cost_usd")
    @classmethod
    def _round_cost(cls, value: float) -> float:
        # Notes show cost to 4 dp; rounding at construction keeps parse(render(h)) == h.
        return round(value, COST_DECIMALS)


class Handoff(BaseModel):
    """A parsed handoff. ``sections`` preserves document order; keys are exact H2 titles."""

    meta: HandoffMeta
    title: str = ""
    """Optional H1 (without ``# ``). Rendered before sections when set."""
    preamble: str = ""
    """Text between the H1 and the first H2 (usually empty)."""
    sections: dict[str, str]
    """H2 title to section body (without the ``## `` line, stripped)."""

    def section(self, name: str) -> str:
        """Body of section ``name``; ``KeyError`` if absent."""
        return self.sections[name]


IssueRaiser = AgentName | Literal["maf"]
"""Who raised an issue: a critic, Gemini for source-audit (``SRC``) issues, ``maf`` for lint (``LINT``) issues."""


class Issue(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    severity: Severity
    text: str
    raised_by: IssueRaiser


class Response(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    stance: Literal["accept", "reject", "partial"]
    text: str


class Ruling(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    ruling: Literal["fix", "wontfix"]
    text: str


class HandoffInvalid(ValueError):
    """Validation failed. ``errors`` are human/agent-readable one-liners used in the repair prompt."""

    def __init__(self, kind: HandoffKind, errors: list[str]) -> None:
        self.kind = kind
        self.errors = errors
        super().__init__(f"invalid {kind} handoff: " + "; ".join(errors))




# ---------------------------------------------------------------------------
# Section splitting

_H1_RE = re.compile(r"^ {0,3}#[ \t]+(?P<title>.+?)(?:[ \t]+#+)?[ \t]*$")
_H2_RE = re.compile(r"^ {0,3}##[ \t]+(?P<title>.+?)(?:[ \t]+#+)?[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")


class _BlockTracker:
    """Tracks whether the current line is inside a fenced code block or a ``$$`` math block."""

    def __init__(self) -> None:
        self._fence: str | None = None
        self._math = False

    def inside(self, line: str) -> bool:
        """Feed one line; True if it is part of (or delimits) a code/math block."""
        if self._fence is not None:
            stripped = line.strip()
            if (
                len(line) - len(line.lstrip(" ")) <= 3
                and stripped
                and set(stripped) == {self._fence[0]}
                and len(stripped) >= len(self._fence)
            ):
                self._fence = None
            return True
        if self._math:
            if line.strip().endswith("$$"):
                self._math = False
            return True
        fence = _FENCE_RE.match(line)
        if fence and not (fence["fence"][0] == "`" and "`" in fence["info"]):
            self._fence = fence["fence"]
            return True
        stripped = line.strip()
        if stripped.startswith("$$"):
            # "$$ x $$" on one line is self-contained; a bare "$$" or "$$ x" opens a block.
            self._math = not (len(stripped) >= 4 and stripped.endswith("$$"))
            return True
        return False


def _iter_lines(body: str) -> list[tuple[str, bool]]:
    tracker = _BlockTracker()
    return [(line, tracker.inside(line)) for line in body.replace("\r\n", "\n").split("\n")]


def split_sections(body: str) -> tuple[str, str, dict[str, str]]:
    """Split a markdown body into ``(h1_title, preamble, {h2_title: text})``.

    H2 detection ignores lines inside fenced code blocks (``` or ~~~) and ``$$`` math blocks.
    Duplicate H2 titles are kept as ``"Title"``, ``"Title (2)"``, and so on, which makes validation fail.
    The H1 title is recognised only as the first non-blank line before any H2.
    """
    title = ""
    preamble: list[str] = []
    sections: dict[str, list[str]] = {}
    seen: dict[str, int] = {}
    current: list[str] | None = None
    started = False
    for line, in_block in _iter_lines(body):
        h2 = None if in_block else _H2_RE.match(line)
        if h2:
            name = h2["title"].strip()
            seen[name] = seen.get(name, 0) + 1
            key = name if seen[name] == 1 else f"{name} ({seen[name]})"
            while key in sections:  # a literal "Title (2)" heading already exists
                seen[name] += 1
                key = f"{name} ({seen[name]})"
            current = sections[key] = []
            continue
        if current is not None:
            current.append(line)
            continue
        if not started and line.strip():
            started = True
            h1 = None if in_block else _H1_RE.match(line)
            if h1:
                title = h1["title"].strip()
                continue
        preamble.append(line)
    return title, "\n".join(preamble).strip(), {k: "\n".join(v).strip() for k, v in sections.items()}


# ---------------------------------------------------------------------------
# Frontmatter

class _FrontmatterDumper(yaml.SafeDumper):
    """SafeDumper emitting ISO-8601 datetimes (``T`` separator) and multi-line strings as blocks."""

    def ignore_aliases(self, data: Any) -> bool:
        return True  # never emit &anchors/*aliases: Obsidian's property editor does not understand them


def _represent_datetime(dumper: yaml.SafeDumper, value: datetime) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:timestamp", value.isoformat())


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.Node:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_FrontmatterDumper.add_representer(datetime, _represent_datetime)
_FrontmatterDumper.add_representer(str, _represent_str)


def dump_frontmatter(data: dict[str, Any]) -> str:
    """``---\\n<yaml>---\\n``. Values must be YAML-safe primitives or ``datetime`` (kept as timestamps)."""
    text = yaml.dump(data, Dumper=_FrontmatterDumper, sort_keys=False, allow_unicode=True, width=10_000)
    return f"---\n{text}---\n"


def load_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split note text into ``(metadata, content)``. Raises ``ValueError`` on malformed YAML or a
    frontmatter block that is not a mapping. Text without frontmatter gives ``({}, text)``."""
    handler = YAMLHandler()
    stripped = text.lstrip("﻿").strip()
    if not handler.detect(stripped):
        return {}, stripped
    try:
        fm, content = handler.split(stripped)
    except ValueError:
        raise ValueError("unterminated frontmatter block (missing closing '---')") from None
    try:
        data = handler.load(fm)
    except yaml.YAMLError as exc:
        raise ValueError(f"frontmatter is not valid YAML: {exc}") from None
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    return data, content.strip()


def _validation_messages(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "frontmatter"
        out.append(f"frontmatter '{loc}': {err['msg']}")
    return out


def parse_handoff(text: str) -> Handoff:
    """Parse a full note (frontmatter + body) via python-frontmatter. Raises ``HandoffInvalid``
    (kind taken from frontmatter if readable, else ``INGESTION``) on malformed frontmatter."""
    try:
        data, content = load_frontmatter(text)
    except ValueError as exc:
        raise HandoffInvalid(HandoffKind.INGESTION, [str(exc)]) from None
    try:
        kind = HandoffKind(data.get("stage"))
    except ValueError:
        kind = HandoffKind.INGESTION
    if not data:
        raise HandoffInvalid(kind, ["missing YAML frontmatter"])
    try:
        meta = HandoffMeta.model_validate(data)
    except ValidationError as exc:
        raise HandoffInvalid(kind, _validation_messages(exc)) from None
    title, preamble, sections = split_sections(content)
    return Handoff(meta=meta, title=title, preamble=preamble, sections=sections)


# ---------------------------------------------------------------------------
# Validation

_GRAMMARS: dict[tuple[HandoffKind, str], tuple[re.Pattern[str], str]] = {
    (HandoffKind.CRITIQUE, "Issues"): (ISSUE_RE, "- [critical|major|minor] <GPT|GEM|CLA>-<n>: <text>"),
    (HandoffKind.REBUTTAL, "Responses"): (RESPONSE_RE, "- <ID> [accept|reject|partial]: <text>"),
    (HandoffKind.ADJUDICATION, "Rulings"): (RULING_RE, "- <ID> [fix|wontfix]: <text>"),
}
_VERDICT_SECTION = (HandoffKind.CROSSCHECK, "Verdict")
_DUP_RE = re.compile(r"^(?P<base>.+) \((?P<n>\d+)\)$")
_CONTINUATION_RE = re.compile(r"^(?: {2,}|\t)")
"""An indented line (2+ spaces or a tab): continuation of the preceding item, sub-bullets included."""
_ID = rf"{_ANY_ID}\b"
_ITEM_LIKE: dict[tuple[HandoffKind, str], re.Pattern[str]] = {
    (HandoffKind.CRITIQUE, "Issues"): re.compile(rf"^[-*+]\s+(?:\[[^\]]*\]\s*{_ID}|{_ID}\s*[:\[])", re.IGNORECASE),
    (HandoffKind.REBUTTAL, "Responses"): re.compile(
        rf"^[-*+]\s+{_ID}\s*(?:[:\[]|(?:accept|reject|partial)\b)", re.IGNORECASE
    ),
    (HandoffKind.ADJUDICATION, "Rulings"): re.compile(rf"^[-*+]\s+{_ID}\s*(?:[:\[]|(?:fix|wontfix)\b)", re.IGNORECASE),
}
"""A (stripped) bullet shaped like an item that does not match the grammar: a ``[tag]`` before an id, or an id
followed by ``:``, ``[`` or a stance/ruling word (``- [high] GPT-2: ...``, ``- GEM-2 reject: ...``). Indented, it is
an error rather than a continuation: folded, it would silently drop an issue or turn a rejection into the default."""


def _clip(line: str, limit: int = 80) -> str:
    return line if len(line) <= limit else line[: limit - 3] + "..."


@dataclass(frozen=True)
class _Item:
    """One grammar item: the match of its bullet line, and its text with continuation lines folded in."""

    match: re.Match[str]
    text: str


def _grammar_items(kind: HandoffKind, name: str, text: str) -> tuple[list[_Item], list[str]]:
    """Split a grammar section into items, returning ``(items, errors)``.

    A line matching the item pattern (at any indentation, so a nested item is still its own item) starts
    an item. Blank lines are skipped, and an indented line (``_CONTINUATION_RE``) is folded into the
    preceding item's text, joined with single spaces; the item's text may start there (``- GPT-1 [accept]:``
    followed by indented sub-bullets). Errors: any other top-level line, a top-level bullet that does not
    match included; an indented line that looks like an item (``_ITEM_LIKE``) but does not match; an
    indented first line; and an item whose text is still empty after folding. A malformed item must never
    vanish into its neighbour. Indented lines after an erroneous line are not reported again.
    """
    pattern, shape = _GRAMMARS[(kind, name)]
    item_like = _ITEM_LIKE[(kind, name)]
    items: list[tuple[int, re.Match[str], list[str]]] = []
    errors: list[str] = []
    ids: set[str] = set()
    current: list[str] | None = None  # continuation lines of the item being read; None after an error line
    for lineno, line in enumerate(text.split("\n"), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        match = pattern.match(stripped)
        if match:
            if match["id"] in ids:
                errors.append(f"'## {name}' line {lineno}: duplicate id {match['id']}")
            ids.add(match["id"])
            items.append((lineno, match, []))
            current = items[-1][2]
            continue
        indented = _CONTINUATION_RE.match(line) is not None
        if indented and not item_like.match(stripped):
            if current is not None:
                current.append(stripped)
            elif not items and not errors:
                errors.append(f"'## {name}' line {lineno}: {_clip(line)!r} is indented, but the section must start "
                              f"with an item `{shape}`")
            continue
        if indented:
            hint = " (an indented line that starts like an item must be a well-formed item; reword a mere note)"
        elif stripped.startswith(("- ", "* ", "+ ")):
            hint = ""
        else:
            hint = " (continuation lines must be indented by two spaces)"
        errors.append(f"'## {name}' line {lineno}: {_clip(line)!r} does not match `{shape}`{hint}")
        current = None
    folded = []
    for lineno, match, extra in items:
        item_text = " ".join(part for part in (match["text"].strip(), *extra) if part)
        if not item_text:
            errors.append(f"'## {name}' line {lineno}: item {match['id']} has no text after the colon or on "
                          "indented lines below it")
        folded.append(_Item(match, item_text))
    return folded, errors


def _grammar_errors(kind: HandoffKind, name: str, text: str) -> list[str]:
    if (kind, name) == _VERDICT_SECTION:
        first = text.split("\n", 1)[0].strip()
        if not VERDICT_RE.match(first):
            return [f"'## Verdict' must start with a line that is exactly PASS or LOOP, got {first!r}"]
        return []
    if (kind, name) not in _GRAMMARS or text == NONE_MARKER:
        return []
    return _grammar_items(kind, name, text)[1]


def validate_body(body: str, kind: HandoffKind) -> list[str]:
    """Return errors (empty list = valid) for an agent-written body.

    Checks: every ``REQUIRED_SECTIONS[kind]`` present exactly once and in order; each non-empty
    (or exactly ``None.`` when allowed by ``EMPTY_OK``); kind grammars:
    CRITIQUE ``## Issues`` lines match ``ISSUE_RE``; REBUTTAL ``## Responses`` match ``RESPONSE_RE``;
    ADJUDICATION ``## Rulings`` match ``RULING_RE``; CROSSCHECK ``## Verdict`` matches ``VERDICT_RE``.
    In the item grammars, blank lines and indented continuation lines belong to the preceding item;
    any other top-level line that is not a well-formed item is an error, and so is an indented line that
    looks like a malformed item (see ``_grammar_items``).
    ``None.`` is accepted only in the sections listed in ``EMPTY_OK``.
    """
    _, _, sections = split_sections(body)
    return _validate_sections(sections, kind)


def _validate_sections(sections: dict[str, str], kind: HandoffKind) -> list[str]:
    errors: list[str] = []
    required = REQUIRED_SECTIONS[kind]
    counts: dict[str, int] = {}
    for key in sections:
        dup = _DUP_RE.match(key)
        base = dup["base"] if dup and dup["base"] in sections else key
        counts[base] = counts.get(base, 0) + 1
    for base, n in counts.items():
        if n > 1:
            errors.append(f"section '## {base}' appears {n} times; it must appear exactly once")

    keys = list(sections)
    present = [name for name in required if name in sections]
    for name in required:
        if name not in sections:
            errors.append(f"missing required section '## {name}'")
    positions = [keys.index(name) for name in present]
    if positions != sorted(positions):
        found = ", ".join(k for k in keys if k in required)
        errors.append(f"required sections are out of order: expected {', '.join(required)}; found {found}")

    for name in present:
        text = sections[name]
        if not text:
            hint = f" (write exactly '{NONE_MARKER}' if there is nothing to report)" if (kind, name) in EMPTY_OK else ""
            errors.append(f"section '## {name}' is empty{hint}")
            continue
        if text == NONE_MARKER and (kind, name) not in EMPTY_OK:
            errors.append(f"section '## {name}' may not be '{NONE_MARKER}'")
            continue
        errors.extend(_grammar_errors(kind, name, text))
    return errors


def validate_handoff(handoff: Handoff) -> list[str]:
    """``validate_body`` on the rendered body, using ``handoff.meta.stage`` as the kind."""
    return validate_body(render_body(handoff.sections), handoff.meta.stage)


_WRAP_OPEN_RE = re.compile(r"^(?P<fence>`{3,}|~{3,})[ \t]*(?:markdown|md)?[ \t]*$", re.IGNORECASE)


def _unwrap(body: str) -> str:
    """Strip a surrounding ```markdown fence and/or a leading frontmatter block added by a model."""
    text = body.lstrip("﻿").strip()
    for _ in range(2):  # the two wrappers may be nested in either order
        lines = text.split("\n")
        opener = _WRAP_OPEN_RE.match(lines[0]) if lines else None
        if opener and len(lines) >= 2 and lines[-1].strip() == opener["fence"]:
            text = "\n".join(lines[1:-1]).strip()
        if text.startswith("---"):
            try:
                _, content = load_frontmatter(text)
            except ValueError:
                pass
            else:
                text = content
    return text


def build_handoff(body: str, meta: HandoffMeta) -> Handoff:
    """Parse ``body`` into a ``Handoff`` with ``meta``; raise ``HandoffInvalid`` if ``validate_body`` fails.

    Strips a leading frontmatter block or a surrounding ```markdown fence if the model added one.
    """
    text = _unwrap(body)
    if not text:
        raise HandoffInvalid(meta.stage, ["the response is empty; expected the note body"])
    title, preamble, sections = split_sections(text)
    errors = _validate_sections(sections, meta.stage)
    if errors:
        raise HandoffInvalid(meta.stage, errors)
    return Handoff(meta=meta, title=title, preamble=preamble, sections=sections)


# ---------------------------------------------------------------------------
# Rendering

def render_body(sections: dict[str, str]) -> str:
    """Render only ``## Title\\n\\ntext`` blocks (used for Python-assembled notes and prompts)."""
    blocks = [f"## {name}\n\n{text.strip()}" if text.strip() else f"## {name}" for name, text in sections.items()]
    return "\n\n".join(blocks) + "\n" if blocks else ""


def render_handoff(handoff: Handoff) -> str:
    """Serialize to note text: YAML frontmatter (key ``from`` not ``from_``, ISO timestamps,
    ``cost_usd`` rounded to 4 dp), optional ``# title``, preamble, then ``## `` sections in order.
    ``parse_handoff(render_handoff(h)) == h`` must hold."""
    data = handoff.meta.model_dump(mode="json", by_alias=True)
    data["created"] = handoff.meta.created
    data["cost_usd"] = round(handoff.meta.cost_usd, COST_DECIMALS)
    parts = [dump_frontmatter(data)]
    if handoff.title.strip():
        parts.append(f"# {handoff.title.strip()}\n")
    if handoff.preamble.strip():
        parts.append(handoff.preamble.strip() + "\n")
    body = render_body(handoff.sections)
    if body:
        parts.append(body)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Prompts

_SECTION_HINTS: dict[tuple[HandoffKind, str], str] = {
    (HandoffKind.ROUTING, "Summary"): "one-paragraph restatement of the request",
    (HandoffKind.ROUTING, "Execution Mode"): "exactly one of `code`, `prose`, `mixed`",
    (HandoffKind.ROUTING, "Instructions for Gemini"): "what to research, read or extract",
    (HandoffKind.ROUTING, "Search Queries"): "bullet list of web search queries",
    (HandoffKind.ROUTING, "Deliverable"): "what the final output must be",
    (HandoffKind.INGESTION, "Summary"): "the findings in a few sentences",
    (HandoffKind.INGESTION, "Sources"): "the verified sources later stages may cite, one entry each (grammar below)",
    (HandoffKind.INGESTION, "Key Facts"): (
        "bullet list of facts, each citing its source id, e.g. `[S1]`, or labelled as an inference"
    ),
    (HandoffKind.INGESTION, "Data Tables"): "markdown tables of extracted numbers",
    (HandoffKind.INGESTION, "Open Questions"): "gaps or contradictions still unresolved",
    (HandoffKind.STRATEGY, "Summary"): "the chosen approach in a few sentences",
    (HandoffKind.STRATEGY, "Options Considered"): "the alternatives and their trade-offs",
    (HandoffKind.STRATEGY, "Chosen Strategy"): "the selected option and why",
    (HandoffKind.STRATEGY, "Execution Brief"): "concrete instructions for the executor",
    (HandoffKind.STRATEGY, "Acceptance Criteria"): (
        "numbered, testable criteria, one `- AC-<n> [hard|soft]: <criterion>` bullet each (grammar below)"
    ),
    (HandoffKind.STRATEGY, "Risks"): "bullet list of risks and mitigations",
    (HandoffKind.EXECUTION, "Summary"): "what was produced and its status",
    (HandoffKind.EXECUTION, "Artifacts"): "bullets of the form ``- `rel/path` - description`` (paths relative to the workspace)",
    (HandoffKind.EXECUTION, "Implementation Notes"): "design decisions and non-obvious details",
    (HandoffKind.EXECUTION, "Verification"): "what was run or checked, with results",
    (HandoffKind.EXECUTION, "Known Limitations"): "remaining gaps, or `None.`",
    (HandoffKind.CRITIQUE, "Summary"): "overall assessment in a few sentences",
    (HandoffKind.CRITIQUE, "Issues"): "one bullet per issue (grammar below)",
    (HandoffKind.REBUTTAL, "Summary"): "overall response in a few sentences",
    (HandoffKind.REBUTTAL, "Responses"): "one bullet per issue (grammar below)",
    (HandoffKind.ADJUDICATION, "Summary"): "overall ruling in a few sentences",
    (HandoffKind.ADJUDICATION, "Rulings"): "one bullet per disputed issue (grammar below)",
    (HandoffKind.CROSSCHECK, "Summary"): "outcome of the cross-check",
    (HandoffKind.CROSSCHECK, "Issues"): "all issues raised",
    (HandoffKind.CROSSCHECK, "Rulings"): "adjudication rulings",
    (HandoffKind.CROSSCHECK, "Applied Fixes"): "fixes applied, one bullet per issue id",
    (HandoffKind.CROSSCHECK, "Unresolved Critical"): "critical issues left unfixed",
    (HandoffKind.CROSSCHECK, "Verdict"): "`PASS` or `LOOP` on the first line",
    (HandoffKind.FINAL, "Summary"): "the result in a few sentences",
    (HandoffKind.FINAL, "Deliverables"): "bullet list of deliverables (wikilinks where possible)",
    (HandoffKind.FINAL, "Verification"): "how the acceptance criteria were checked",
    (HandoffKind.FINAL, "Provenance"): "wikilink bullets to the notes this builds on, e.g. `- [[02-strategy]]`",
    (HandoffKind.FINAL, "Limitations"): "remaining limitations and unresolved issues, or `None.`",
}

_GRAMMAR_EXAMPLES: dict[HandoffKind, tuple[str, str]] = {
    HandoffKind.CRITIQUE: ("Issues", "- [critical] GPT-3: free() of a foreign block corrupts the bitmap"),
    HandoffKind.REBUTTAL: ("Responses", "- GPT-3 [partial]: added a debug-only range check; release builds unchanged"),
    HandoffKind.ADJUDICATION: ("Rulings", "- GPT-3 [fix]: the requirements imply robustness against misuse"),
}


_PYTHON_IDS_NOTE = (
    f" Issues raised by maf itself use `{SOURCE_AUDIT_PREFIX}` (the web-grounded source audit) and `{LINT_PREFIX}` "
    "(the Markdown lint); treat them exactly like critics' issues."
)

SOURCES_GRAMMAR = f"""### `## Sources` entry grammar

List only sources whose bibliographic record you checked yourself (the publisher's page, the DOI record, or the
attached file), one entry per source, or write exactly `{NONE_MARKER}` when nothing can be cited. Later stages may
cite these entries and nothing else. Each entry is a top-level bullet followed by field lines indented two spaces:

```
- [S1] <title exactly as published>
  - Authors: <every author, or the organisation>
  - Venue: <journal, proceedings, publisher or site; volume, issue, pages where they exist>
  - Year: <four-digit year, or n.d.>
  - DOI: <10.xxxx/...>
  - URL: <https://...>
  - File: <inputs/... for an attached file>
  - Excerpt (<locator: section, page, table, figure or equation>): "<verbatim quote>"
```

Authors, Venue and Year appear exactly once. Give at least one of DOI, URL or File (each at most once). Give one
Excerpt line per fact a later stage may attribute to the source (at least one): a verbatim quote in double quotes
with a precise locator. Keep every field on its own single line. IDs are `S1`, `S2`, ...; `## Key Facts` cites them as
`[S1]`. A search result you could not verify is not a source: mention it under `## Open Questions` instead."""
"""Injected into the ingestion format spec (and so into its repair prompt)."""


def format_spec(kind: HandoffKind) -> str:
    """Markdown description of the required sections and line grammar for ``kind``, injected
    into every generation prompt so agents know the contract up front."""
    lines = [
        f"## Output format: {kind.value} handoff",
        "",
        "Reply with the note body only, in Markdown. Do not add YAML frontmatter, an H1 title or a "
        "surrounding code fence: Python adds the metadata.",
        "",
        "Use these H2 sections, each exactly once, in this order, with these exact titles:",
        "",
    ]
    for i, name in enumerate(REQUIRED_SECTIONS[kind], start=1):
        hint = _SECTION_HINTS.get((kind, name), "")
        empty = f" May be exactly `{NONE_MARKER}` if there is nothing to report." if (kind, name) in EMPTY_OK else ""
        lines.append(f"{i}. `## {name}`" + (f": {hint}." if hint else ".") + empty)
    lines += [
        "",
        "No section may be empty. You may add extra `## ` sections, but never repeat a required one. "
        "Headings inside code fences or `$$` blocks do not count. Use `$...$`/`$$...$$` for math and "
        "`[[note-name]]` wikilinks for other notes.",
    ]
    grammar = next(((name, g) for (k, name), g in _GRAMMARS.items() if k == kind), None)
    if grammar is not None:
        name, (_, shape) = grammar
        example = _GRAMMAR_EXAMPLES[kind][1]
        lines += [
            "",
            f"### `## {name}` line grammar",
            "",
            f"Every item of `## {name}` starts with a top-level bullet of the form `{shape}`, or the section must "
            f"be exactly `{NONE_MARKER}`. Keep each item on its one bullet line where you can; extra detail may "
            "follow on continuation lines indented by two spaces (indented sub-bullets are fine) and is folded into "
            "that item. Every line that is not indented must start a new item: no prose, headings or other bullets "
            "at the top level. An indented bullet that starts with an id or a `[...]` tag must be a well-formed item "
            "itself. Each id appears at most once. IDs are the critic's prefix (`GPT` for ChatGPT, `GEM` "
            "for Gemini, `CLA` for Claude) plus a number."
            + (_PYTHON_IDS_NOTE if kind in (HandoffKind.REBUTTAL, HandoffKind.ADJUDICATION) else ""),
            "",
            f"Example: `{example}`",
        ]
    if kind == HandoffKind.INGESTION:
        lines += ["", SOURCES_GRAMMAR]
    if kind == HandoffKind.STRATEGY:
        lines += ["", ACCEPTANCE_CRITERIA_GRAMMAR]
    if kind == HandoffKind.CROSSCHECK:
        lines += ["", "The first line of `## Verdict` must be exactly `PASS` or `LOOP`."]
    return "\n".join(lines) + "\n"


def _fence_for(text: str) -> str:
    longest = max((len(m) for m in re.findall(r"`{3,}", text)), default=0)
    return "`" * max(3, longest + 1)


def repair_prompt(kind: HandoffKind, bad_body: str, errors: list[str]) -> str:
    """The single repair attempt's user prompt: lists ``errors``, restates the required
    sections/grammars for ``kind`` (via ``format_spec``), includes ``bad_body``, and asks for the
    full corrected body only."""
    fence = _fence_for(bad_body)
    problems = "\n".join(f"- {e}" for e in errors) or "- (no specific errors reported)"
    return (
        f"Your previous {kind.value} handoff did not pass validation.\n\n"
        f"# Problems\n\n{problems}\n\n"
        f"# Required format\n\n{format_spec(kind)}\n"
        f"# Your previous reply\n\n{fence}markdown\n{bad_body.strip()}\n{fence}\n\n"
        "Fix every problem listed above while keeping the content. Reply with the complete corrected "
        "note body only: no commentary, no frontmatter, no surrounding code fence."
    )


# ---------------------------------------------------------------------------
# Line-grammar parsers

def _parse_items(section_text: str, kind: HandoffKind, name: str) -> list[_Item]:
    text = section_text.strip()
    if text == NONE_MARKER or not text:
        return []
    items, errors = _grammar_items(kind, name, text)
    if errors:
        raise HandoffInvalid(kind, errors)
    return items


def parse_issues(section_text: str, raised_by: AgentName) -> list[Issue]:
    """Parse ``## Issues`` (``None.`` means no issues). Continuation lines are folded into ``text``.
    Raises ``HandoffInvalid`` on bad lines."""
    return [
        Issue(id=i.match["id"], severity=i.match["severity"], text=i.text, raised_by=raised_by)  # type: ignore[arg-type]
        for i in _parse_items(section_text, HandoffKind.CRITIQUE, "Issues")
    ]


def parse_responses(section_text: str) -> list[Response]:
    """Parse ``## Responses`` of a rebuttal (``None.`` means none). Continuation lines are folded into ``text``.
    Raises ``HandoffInvalid`` on bad lines."""
    return [
        Response(id=i.match["id"], stance=i.match["stance"], text=i.text)  # type: ignore[arg-type]
        for i in _parse_items(section_text, HandoffKind.REBUTTAL, "Responses")
    ]


def parse_rulings(section_text: str) -> list[Ruling]:
    """Parse ``## Rulings`` of an adjudication (``None.`` means none). Continuation lines are folded into ``text``.
    Raises ``HandoffInvalid`` on bad lines."""
    return [
        Ruling(id=i.match["id"], ruling=i.match["ruling"], text=i.text)  # type: ignore[arg-type]
        for i in _parse_items(section_text, HandoffKind.ADJUDICATION, "Rulings")
    ]


# ---------------------------------------------------------------------------
# Strategy acceptance criteria

ACCEPTANCE_CRITERIA = "Acceptance Criteria"
"""The strategy section whose items Python reads (critique prompts, the final stage's acceptance gate)."""

CRITERION_RE = re.compile(r"^- (?P<id>AC-\d+) \[(?P<level>hard|soft)\]:(?P<text>(?: .*)?)$")
"""Strict grammar of one ``## Acceptance Criteria`` item, e.g. ``- AC-3 [hard]: make test passes 3 runs in a row``."""

_TASK_MARKER = r"(?:\[[ xX]\]\s+)?"
"""A GFM task-list checkbox (``- [ ] ...``, ``- [x] ...``) before a criterion's label: dropped, never criterion text."""
_CRITERION_LIKE = re.compile(
    rf"^(?:[-*+]|\d+[.)])\s+{_TASK_MARKER}(?:\*\*|__)?\s*(?:\[(?:hard|soft)\]|AC-?\d+)", re.IGNORECASE
)
_CRITERION_BULLET = re.compile(rf"^(?:[-*+]|\d+[.)])\s+{_TASK_MARKER}(?P<rest>.*)$")
_CRITERION_LABEL = re.compile(
    r"^(?:\*\*|__)?\s*(?:[\[(](?P<pre>hard|soft)[\])]\s*)?(?P<id>AC-?\d+)?\s*(?:\*\*|__)?\s*"
    r"(?:[\[(](?P<level>hard|soft)[\])])?\s*(?:\*\*|__)?\s*[:–—-]?\s*(?:\*\*|__)?\s*",
    re.IGNORECASE,
)

ACCEPTANCE_CRITERIA_GRAMMAR = f"""### `## {ACCEPTANCE_CRITERIA}` line grammar

Python reads this section. Write one item per criterion, numbered from 1, each a top-level bullet of exactly the
form `- AC-<n> [hard]: <criterion>` or `- AC-<n> [soft]: <criterion>`, with nothing else at the top level; details
may follow on continuation lines indented by two spaces. `hard`: the deliverable fails without it, and the run does
not count as complete while it is unmet. `soft`: a quality goal, reported but not blocking. Each id appears once.

Example: `- AC-2 [hard]: arm-none-eabi-size reports .text < 2048 bytes for alloc.o built with -Os`"""
"""Injected into the strategy format spec (and so into its repair prompt)."""


@dataclass(frozen=True)
class Criterion:
    """One acceptance criterion of ``02-strategy`` (``parse_acceptance_criteria``)."""

    id: str
    """``AC-<n>``."""
    hard: bool
    """A hard criterion must hold for the run to complete; a soft one is a quality goal."""
    text: str
    """Continuation lines folded in, joined with single spaces."""

    @property
    def line(self) -> str:
        """The item in the strict grammar (``CRITERION_RE``)."""
        return f"- {self.id} [{'hard' if self.hard else 'soft'}]: {self.text}"


def parse_acceptance_criteria(source: Handoff | str) -> list[Criterion]:
    """Criteria of a strategy note (or of the text of its ``## Acceptance Criteria`` section), in order.

    Lenient, so hand-edited and older notes still yield criteria: an item starts at a line matching ``CRITERION_RE``
    (at any indentation) or at any top-level bullet or numbered line, whose label may be loose
    (``1. **AC-2 (soft):** ...``, ``2. **AC-2** (soft) — ...``, ``- [hard] ...``) and may follow a task-list checkbox
    (``- [ ] AC-2 [soft]: ...``, which is dropped). Indented lines, and unindented non-bullet lines after an item,
    are folded into that item; an item left without text is dropped. A missing label means ``hard``; a missing or
    repeated id gets the next free ``AC-<n>``. Text before the first item is an introduction and is dropped, unless
    there is no item at all: then the whole section is one hard criterion. ``None.``, an empty or a missing section
    yields no criteria."""
    text = source.sections.get(ACCEPTANCE_CRITERIA, "") if isinstance(source, Handoff) else source
    text = text.strip()
    if not text or text == NONE_MARKER:
        return []
    items: list[tuple[str | None, bool, list[str]]] = []  # (id, hard, text parts)
    intro: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        strict = CRITERION_RE.match(stripped)
        bullet = _CRITERION_BULLET.match(line)  # top level only: ``line`` still has its indentation
        if strict:
            items.append((strict["id"], strict["level"] == "hard", [strict["text"].strip()]))
        elif bullet:
            rest = bullet["rest"]
            label = _CRITERION_LABEL.match(rest)
            assert label is not None  # every part of the label is optional
            level = (label["level"] or label["pre"] or "hard").lower()
            if label["id"] or label["pre"] or label["level"]:
                rest = rest[label.end() :]
                ident = f"AC-{label['id'][2:].lstrip('-')}" if label["id"] else None
            else:
                ident = None
            items.append((ident, level == "hard", [rest.strip()]))
        elif items:
            items[-1][2].append(stripped)
        else:
            intro.append(stripped)
    if not items:
        return [Criterion("AC-1", True, " ".join(intro))]
    items = [(ident, hard, parts) for ident, hard, parts in items if any(parts)]
    used = {ident.upper() for ident, _hard, _parts in items if ident}
    seen: set[str] = set()
    criteria: list[Criterion] = []
    next_number = 1
    for ident, hard, parts in items:
        if ident is None or ident.upper() in seen:
            while f"AC-{next_number}" in used or f"AC-{next_number}" in seen:
                next_number += 1
            ident = f"AC-{next_number}"
        ident = ident.upper()
        seen.add(ident)
        criteria.append(Criterion(ident, hard, " ".join(p for p in parts if p)))
    return criteria


def acceptance_criteria_errors(section: str) -> list[str]:
    """Errors of ``section`` against the strict grammar (empty list = valid): every top-level line must be a
    ``CRITERION_RE`` item, an indented line that starts like a criterion must be one too, ids are unique, and no item
    is empty after folding its continuation lines. At least one criterion is required."""
    name = ACCEPTANCE_CRITERIA
    errors: list[str] = []
    ids: set[str] = set()
    items: list[tuple[int, str, list[str]]] = []
    for lineno, line in enumerate(section.strip().split("\n"), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        match = CRITERION_RE.match(stripped)
        if match:
            if match["id"] in ids:
                errors.append(f"'## {name}' line {lineno}: duplicate id {match['id']}")
            ids.add(match["id"])
            items.append((lineno, match["id"], [match["text"].strip()]))
        elif _CONTINUATION_RE.match(line) and items and not _CRITERION_LIKE.match(stripped):
            items[-1][2].append(stripped)
        else:
            errors.append(f"'## {name}' line {lineno}: {stripped[:80]!r} does not match "
                          "`- AC-<n> [hard|soft]: criterion`")
    for lineno, ident, parts in items:
        if not " ".join(p for p in parts if p):
            errors.append(f"'## {name}' line {lineno}: criterion {ident} has no text")
    if not items and not errors:
        errors.append(f"'## {name}' must list at least one criterion `- AC-<n> [hard|soft]: criterion`")
    return errors


def normalize_acceptance_criteria(section: str) -> str:
    """``section`` unchanged if it follows the strict grammar, else its ``parse_acceptance_criteria`` reading, one
    ``Criterion.line`` per criterion."""
    if not acceptance_criteria_errors(section):
        return section
    criteria = parse_acceptance_criteria(section)
    return "\n".join(c.line for c in criteria) if criteria else section


# ---------------------------------------------------------------------------
# Ingestion sources

SOURCE_FIELDS: tuple[str, ...] = ("Authors", "Venue", "Year", "DOI", "URL", "File", "Excerpt")
_SOURCE_FIELD_NAMES = {name.lower(): name for name in SOURCE_FIELDS}
SOURCE_ENTRY_RE = re.compile(r"^- \[(?P<id>S\d+)\] (?P<title>\S.*)$")
"""First line of an ingestion ``## Sources`` entry: ``- [S1] <title as published>``."""
_SOURCE_FIELD_RE = re.compile(r"^[-*+]\s+(?P<field>[A-Za-z]+)(?:\s*\((?P<locator>.*?)\))?\s*:(?P<value>.*)$")
_QUOTE_RE = re.compile(r'^["\u201c](?P<quote>.*\S.*)["\u201d]$')
_YEAR_RE = re.compile(r"^(?:\d{4}[a-z]?|n\.d\.)$")
_DOI_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?(?P<doi>10\.\d{4,9}/\S+)$", re.IGNORECASE)
_URL_RE = re.compile(r"^<?(?P<url>https?://[^\s<>]+)>?$")
_SOURCES_SHAPE = "- [S<n>] <title>"


class SourceExcerpt(BaseModel):
    model_config = ConfigDict(frozen=True)

    locator: str
    """Where the quote is: section, page, table, figure or equation."""
    quote: str


class Source(BaseModel):
    """One verified entry of ingestion ``## Sources`` (see ``SOURCES_GRAMMAR``)."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    authors: str
    venue: str
    year: str
    doi: str = ""
    """Bare DOI (``10.xxxx/...``), without resolver prefix."""
    url: str = ""
    file: str = ""
    excerpts: tuple[SourceExcerpt, ...]


def _source_entries(text: str) -> tuple[list[Source], list[str]]:
    """Split ``## Sources`` into entries and validate each (``SOURCES_GRAMMAR``), returning ``(sources, errors)``.
    Indented lines under an erroneous top-level line are not reported again."""
    errors: list[str] = []
    entries: list[tuple[int, re.Match[str], list[tuple[int, re.Match[str]]]]] = []
    current: list[tuple[int, re.Match[str]]] | None = None
    for lineno, line in enumerate(text.split("\n"), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if _CONTINUATION_RE.match(line) is None:
            entry = SOURCE_ENTRY_RE.match(stripped)
            if entry:
                entries.append((lineno, entry, []))
                current = entries[-1][2]
            else:
                errors.append(f"'## Sources' line {lineno}: {_clip(line)!r} does not start an entry `{_SOURCES_SHAPE}`")
                current = None
            continue
        if current is None:
            if not entries and not errors:
                errors.append(f"'## Sources' line {lineno}: {_clip(line)!r} is indented, but the section must start "
                              f"with an entry `{_SOURCES_SHAPE}`")
            continue
        field = _SOURCE_FIELD_RE.match(stripped)
        if field is None or field["field"].lower() not in _SOURCE_FIELD_NAMES:
            errors.append(f"'## Sources' line {lineno}: {_clip(line)!r} is not a field line `  - <Field>: <value>` "
                          f"(fields: {', '.join(SOURCE_FIELDS)})")
            continue
        current.append((lineno, field))

    sources: list[Source] = []
    seen: set[str] = set()
    for lineno, entry, fields in entries:
        sid = entry["id"]
        where = f"'## Sources' entry {sid} (line {lineno})"
        if sid in seen:
            errors.append(f"{where}: duplicate id {sid}")
        seen.add(sid)
        values: dict[str, str] = {}
        excerpts: list[SourceExcerpt] = []
        for fl, field in fields:
            name = _SOURCE_FIELD_NAMES[field["field"].lower()]
            value = field["value"].strip()
            locator = field["locator"]
            if name == "Excerpt":
                quote = _QUOTE_RE.match(value)
                if not (locator or "").strip():
                    errors.append(f"'## Sources' line {fl}: an Excerpt needs a locator: "
                                  '`Excerpt (<section, page, table, figure or equation>): "<quote>"`')
                elif quote is None:
                    errors.append(f"'## Sources' line {fl}: an Excerpt must be a verbatim quote in double quotes")
                else:
                    excerpts.append(SourceExcerpt(locator=" ".join(locator.split()), quote=quote["quote"].strip()))
                continue
            if locator is not None:
                errors.append(f"'## Sources' line {fl}: only Excerpt takes a (locator); write `{name}: <value>`")
            if name in values:
                errors.append(f"{where}: {name} appears more than once")
            if not value:
                errors.append(f"'## Sources' line {fl}: {name} is empty")
            values[name] = value.strip("`")
        missing = [name for name in ("Authors", "Venue", "Year") if not values.get(name)]
        if missing:
            errors.append(f"{where}: missing {', '.join(missing)}")
        if not any(values.get(name) for name in ("DOI", "URL", "File")):
            errors.append(f"{where}: give at least one of DOI, URL or File")
        if not excerpts:
            errors.append(f"{where}: give at least one Excerpt line with a locator and a verbatim quote")
        year = values.get("Year", "")
        if year and not _YEAR_RE.match(year):
            errors.append(f"{where}: Year must be a four-digit year or n.d., got {year!r}")
        doi = _DOI_RE.match(values.get("DOI", ""))
        if values.get("DOI") and doi is None:
            errors.append(f"{where}: DOI must look like 10.xxxx/..., got {_clip(values['DOI'], 60)!r}")
        url = _URL_RE.match(values.get("URL", ""))
        if values.get("URL") and url is None:
            errors.append(f"{where}: URL must be an http(s) URL, got {_clip(values['URL'], 60)!r}")
        sources.append(
            Source(
                id=sid,
                title=entry["title"].strip(),
                authors=values.get("Authors", ""),
                venue=values.get("Venue", ""),
                year=year,
                doi=doi["doi"] if doi else "",
                url=url["url"] if url else "",
                file=values.get("File", ""),
                excerpts=tuple(excerpts),
            )
        )
    return sources, errors


def source_errors(section_text: str) -> list[str]:
    """Errors of an ingestion ``## Sources`` body against ``SOURCES_GRAMMAR`` (``None.`` is valid)."""
    text = section_text.strip()
    if text == NONE_MARKER:
        return []
    if not text:
        return ["'## Sources' is empty (write exactly 'None.' if nothing can be cited)"]
    return _source_entries(text)[1]


def parse_sources(section_text: str) -> list[Source]:
    """Parse an ingestion ``## Sources`` body (``None.`` means no citable source). Raises ``HandoffInvalid``."""
    text = section_text.strip()
    if text == NONE_MARKER or not text:
        return []
    sources, errors = _source_entries(text)
    if errors:
        raise HandoffInvalid(HandoffKind.INGESTION, errors)
    return sources


def quote_untrusted(text: str, source: str) -> str:
    """Render fetched/uploaded content as a blockquote prefixed with ``> [!quote] Source: <source>``
    (an Obsidian callout), every line ``> ``-prefixed, so it can never start a heading or instruction."""
    source_line = " ".join(source.split()) or "unknown"
    body = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    lines = [f"> [!quote] Source: {source_line}"]
    lines += [f"> {line}" if line.strip() else ">" for line in body.split("\n")] if body else []
    return "\n".join(lines)
