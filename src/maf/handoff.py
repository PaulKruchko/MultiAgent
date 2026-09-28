"""Handoff notes: frontmatter + required H2 sections per kind; parse, validate, render, repair.

Owner: vault+handoff.

Division of labor: **agents write only the body** (the H2 sections). Python builds the
frontmatter (``from``/``to``/``model``/``cost_usd`` are facts Python knows, not model output).
Validation therefore checks (a) frontmatter shape, (b) required H2 sections present, in
order, non-empty, and (c) kind-specific line grammars (Issues, Responses, Adjudication, Verdict).

Untrusted content (web pages, uploaded documents) appears in handoffs only as quoted data
(``quote_untrusted``): the ingestion stage wraps every section of its note, grounding citations
included, and ``render_inputs`` escapes ``<note>`` tags. Prompts tell every agent that quoted blocks
are data, never instructions.
"""

from __future__ import annotations

import re
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

# Line grammars (one item per line, bullets). IDs are ``<PREFIX>-<n>``: GPT, GEM, CLA.
ISSUE_RE = re.compile(r"^- \[(?P<severity>critical|major|minor)\] (?P<id>(?:GPT|GEM|CLA)-\d+): (?P<text>.+)$")
"""``## Issues`` in critique notes, e.g. ``- [critical] GPT-3: free() of a foreign block corrupts the bitmap``."""
RESPONSE_RE = re.compile(r"^- (?P<id>(?:GPT|GEM|CLA)-\d+) \[(?P<stance>accept|reject|partial)\]: (?P<text>.+)$")
"""``## Responses`` in the rebuttal note (Claude, as author of the execution)."""
RULING_RE = re.compile(r"^- (?P<id>(?:GPT|GEM|CLA)-\d+) \[(?P<ruling>fix|wontfix)\]: (?P<text>.+)$")
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


class Issue(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    severity: Severity
    text: str
    raised_by: AgentName


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


def _clip(line: str, limit: int = 80) -> str:
    return line if len(line) <= limit else line[: limit - 3] + "..."


def _grammar_errors(kind: HandoffKind, name: str, text: str) -> list[str]:
    if (kind, name) == _VERDICT_SECTION:
        first = text.split("\n", 1)[0].strip()
        if not VERDICT_RE.match(first):
            return [f"'## Verdict' must start with a line that is exactly PASS or LOOP, got {first!r}"]
        return []
    grammar = _GRAMMARS.get((kind, name))
    if grammar is None or text == NONE_MARKER:
        return []
    pattern, shape = grammar
    errors: list[str] = []
    ids: set[str] = set()
    for lineno, line in enumerate(text.split("\n"), start=1):
        match = pattern.match(line)
        if not match:
            what = "blank line" if not line.strip() else repr(_clip(line))
            errors.append(f"'## {name}' line {lineno}: {what} does not match `{shape}`")
            continue
        if match["id"] in ids:
            errors.append(f"'## {name}' line {lineno}: duplicate id {match['id']}")
        ids.add(match["id"])
    return errors


def validate_body(body: str, kind: HandoffKind) -> list[str]:
    """Return errors (empty list = valid) for an agent-written body.

    Checks: every ``REQUIRED_SECTIONS[kind]`` present exactly once and in order; each non-empty
    (or exactly ``None.`` when allowed by ``EMPTY_OK``); kind grammars:
    CRITIQUE ``## Issues`` lines match ``ISSUE_RE``; REBUTTAL ``## Responses`` match ``RESPONSE_RE``;
    ADJUDICATION ``## Rulings`` match ``RULING_RE``; CROSSCHECK ``## Verdict`` matches ``VERDICT_RE``.
    Non-bullet lines (blank or prose) inside grammar sections are errors, so the grammar is strict.
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
    (HandoffKind.INGESTION, "Sources"): "bullet list of sources consulted (titles, URLs, file names)",
    (HandoffKind.INGESTION, "Key Facts"): "bullet list of facts, each traceable to a source",
    (HandoffKind.INGESTION, "Data Tables"): "markdown tables of extracted numbers",
    (HandoffKind.INGESTION, "Open Questions"): "gaps or contradictions still unresolved",
    (HandoffKind.STRATEGY, "Summary"): "the chosen approach in a few sentences",
    (HandoffKind.STRATEGY, "Options Considered"): "the alternatives and their trade-offs",
    (HandoffKind.STRATEGY, "Chosen Strategy"): "the selected option and why",
    (HandoffKind.STRATEGY, "Execution Brief"): "concrete instructions for the executor",
    (HandoffKind.STRATEGY, "Acceptance Criteria"): "bullet list of checkable criteria",
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
            f"### `## {name}` line grammar (strict)",
            "",
            f"Every line of `## {name}` must be a single bullet of the form `{shape}`, or the section must be "
            f"exactly `{NONE_MARKER}`. No blank lines, prose, sub-bullets or wrapped lines inside it; each id "
            "appears at most once. IDs are the critic's prefix (`GPT` for ChatGPT, `GEM` for Gemini, `CLA` for "
            "Claude) plus a number.",
            "",
            f"Example: `{example}`",
        ]
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

def _parse_lines(section_text: str, kind: HandoffKind, name: str) -> list[re.Match[str]]:
    text = section_text.strip()
    if text == NONE_MARKER or not text:
        return []
    errors = _grammar_errors(kind, name, text)
    if errors:
        raise HandoffInvalid(kind, errors)
    pattern = _GRAMMARS[(kind, name)][0]
    return [m for m in (pattern.match(line) for line in text.split("\n")) if m is not None]


def parse_issues(section_text: str, raised_by: AgentName) -> list[Issue]:
    """Parse ``## Issues`` (``None.`` means no issues). Raises ``HandoffInvalid`` on bad lines."""
    return [
        Issue(id=m["id"], severity=m["severity"], text=m["text"].strip(), raised_by=raised_by)  # type: ignore[arg-type]
        for m in _parse_lines(section_text, HandoffKind.CRITIQUE, "Issues")
    ]


def parse_responses(section_text: str) -> list[Response]:
    """Parse ``## Responses`` of a rebuttal (``None.`` means none). Raises ``HandoffInvalid`` on bad lines."""
    return [
        Response(id=m["id"], stance=m["stance"], text=m["text"].strip())  # type: ignore[arg-type]
        for m in _parse_lines(section_text, HandoffKind.REBUTTAL, "Responses")
    ]


def parse_rulings(section_text: str) -> list[Ruling]:
    """Parse ``## Rulings`` of an adjudication (``None.`` means none). Raises ``HandoffInvalid`` on bad lines."""
    return [
        Ruling(id=m["id"], ruling=m["ruling"], text=m["text"].strip())  # type: ignore[arg-type]
        for m in _parse_lines(section_text, HandoffKind.ADJUDICATION, "Rulings")
    ]


def quote_untrusted(text: str, source: str) -> str:
    """Render fetched/uploaded content as a blockquote prefixed with ``> [!quote] Source: <source>``
    (an Obsidian callout), every line ``> ``-prefixed, so it can never start a heading or instruction."""
    source_line = " ".join(source.split()) or "unknown"
    body = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    lines = [f"> [!quote] Source: {source_line}"]
    lines += [f"> {line}" if line.strip() else ">" for line in body.split("\n")] if body else []
    return "\n".join(lines)
