"""Tests for ``maf.lint``: the deliverable Markdown linter (pure; the filesystem only for links and walking)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from maf import lint
from maf.lint import DEFAULT_EXCLUDE, LintIssue, blocking, excluded, lint_deliverables, lint_markdown


def rules(text: str, **kw: object) -> list[tuple[str, int]]:
    return [(i.rule, i.line) for i in lint_markdown(text, "doc.md", **kw)]  # type: ignore[arg-type]


def only(text: str, rule: str) -> list[LintIssue]:
    return [i for i in lint_markdown(text, "doc.md") if i.rule == rule]


# A clean deliverable in the thesis' style: nothing here may be flagged.
CLEAN = r"""---
title: Burn control of a DT plasma
tags: [thesis]
---
# Burn control of a DT plasma

## Model

The fusion power $P_f = n^2 \langle\sigma v\rangle E_f V / 4$ is regulated with fueling, auxiliary heating and
impurity seeding. The energy balance is

$$
\frac{dW}{dt} = P_\alpha + P_{aux} - \frac{W}{\tau_E}
$$

where $\tau_E$ follows the IPB98(y,2) scaling (Doyle et al., 2007). We cross-check the model against the ITER
baseline, and a critique of the Lawson criterion is out of scope. The cost is \$5 per shot, or $5 to $10 per day.
The deviation $\lvert e \rvert < 0.1$ holds when $x \mid y$.

| Quantity | Symbol | Value |
|---|---|---|
| Absolute error | $\lvert e \rvert$ | 0.1 |
| Conditional | $p(x \mid y)$ | 0.3 |
| Escaped | $a \| b$ | 2 |
| Status | none | 5 |

> [!note] Display math inside a callout
> $$
> Q = \frac{P_f}{P_{aux}}
> $$

```python
# TODO: fenced code is never linted: [[01-ingestion]] $unbalanced | a | b |
print("n/a", None)
```

~~~
As proposed in review (inside a tilde fence).
~~~

- A list item with `[[02-strategy]]` and `TODO` in inline code, and $\alpha$
  continued on a second line with $\beta +
  \gamma$ spanning lines.

Limitations: None.

![[burn.png]]
"""


def test_clean_document_has_no_issues(tmp_path: Path) -> None:
    (tmp_path / "plots").mkdir()
    (tmp_path / "plots" / "burn.png").write_bytes(b"\x89PNG")
    (tmp_path / "doc.md").write_text(CLEAN, encoding="utf-8")
    assert lint_markdown(CLEAN, "doc.md") == []
    assert lint_markdown(CLEAN, "doc.md", root=tmp_path) == []
    assert lint_deliverables(tmp_path) == []


# ---------------------------------------------------------------------- pipeline-wikilink


@pytest.mark.parametrize(
    "link",
    [
        "[[01-ingestion]]",
        "[[01a-routing]]",
        "[[02-strategy]]",
        "[[03-execution]]",
        "[[03-execution-r2]]",
        "[[04-crosscheck-r3]]",
        "[[04a-critique-gemini]]",
        "[[04b-rebuttal]]",
        "[[04c-adjudication-r2]]",
        "[[05-final]]",
        "[[01-ingestion|the ingestion report]]",
        "[[01-ingestion#Key Facts]]",
        "[[01-ingestion.md]]",
        "![[02-strategy]]",
        "[[runs/2026-09-28-burn-control/05-final]]",
        "[[runs/2026-09-28-burn-control/deliverables/document|document]]",
        "[the report](01-ingestion.md)",
        "[the report](../../runs/2026-09-28-x/03-execution.md)",
        "[notes](runs/2026-09-28-x/assets/plot.png)",
    ],
)
def test_pipeline_links_are_critical(link: str) -> None:
    issues = lint_markdown(f"As shown in {link}, the scaling holds.\n", "thesis.md")
    (issue,) = [i for i in issues if i.severity == "critical"]  # an alias may add meta-commentary
    assert (issue.rule, issue.severity, issue.path, issue.line) == ("pipeline-wikilink", "critical", "thesis.md", 1)
    assert "`" in issue.message and "cite the original source" in issue.message


@pytest.mark.parametrize(
    "link",
    [
        "[[01-introduction]]",
        "[[04-results]]",
        "[[02-strategy-options]]",
        "[[Ingestion]]",
        "![[runs/2026-09-28-x/deliverables/plots/burn.png]]",  # final's relinked image embed
        "![[runs/2026-09-28-x/assets/burn.png]]",
        "[ITER](https://www.iter.org/01-ingestion)",
    ],
)
def test_ordinary_links_are_not_pipeline_links(link: str) -> None:
    assert only(f"See {link}.\n", "pipeline-wikilink") == []


def test_every_pipeline_link_is_reported_and_the_thesis_shape_is_caught() -> None:
    thesis = (
        "#### Confinement\n\n"
        "The IPB98(y,2) scaling [[01-ingestion]] gives $\\tau_E$ [[01-ingestion]]; see [[03-execution]].\n"
        "Parameters follow [[01-ingestion]] and [[02-strategy]].\n"
    )
    issues = only(thesis, "pipeline-wikilink")
    assert [i.line for i in issues] == [3, 3, 3, 4, 4]


def test_a_currency_dollar_cannot_hide_a_pipeline_link() -> None:
    (issue,) = only("It costs $5 per shot [[01-ingestion]] and $x$ more.\n", "pipeline-wikilink")
    assert issue.line == 1


@pytest.mark.parametrize("citation", ["[S2]", "[S1, S3]", "[S1-S4]", "[S12; S3]"])
def test_internal_source_ids_are_critical_pipeline_leaks(citation: str) -> None:
    """The ingestion's ``[S<n>]`` ids point into a note that is not shipped: the new form of ``[[01-ingestion]]``."""
    (issue,) = lint_markdown(f"The IPB98 scaling gives $\\tau_E$ = 3.7 s {citation}.\n", "thesis.md")
    assert (issue.rule, issue.severity, issue.line) == ("pipeline-wikilink", "critical", 1)
    assert "internal source id" in issue.message and "bibliographic record" in issue.message


@pytest.mark.parametrize("text", ["Series [S] and [Sx].", "`[S2]` in code.", "Math $[S2]$.", "Wikilink [[S2]].", "[1, 2]"])
def test_source_id_lookalikes_pass(text: str) -> None:
    assert only(f"{text}\n", "pipeline-wikilink") == []


def test_targets_that_resolve_inside_the_tree_are_not_pipeline_notes(tmp_path: Path) -> None:
    """Simulation output under ``runs/`` and a chapter named like a pipeline note are the deliverable's own files."""
    (tmp_path / "runs" / "exp1").mkdir(parents=True)
    (tmp_path / "runs" / "exp1" / "log.txt").write_text("x\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "01-ingestion.md").write_text("# Data ingestion\n")
    text = "[log](runs/exp1/log.txt) [ingest](docs/01-ingestion.md) [[01-ingestion]] [[02-strategy]]\n"
    (issue,) = lint_markdown(text, "README.md", root=tmp_path)
    assert (issue.rule, issue.severity) == ("pipeline-wikilink", "critical") and "02-strategy" in issue.message
    assert len(only(text, "pipeline-wikilink")) == 4  # without a root, the names alone decide


def test_pipeline_notes_in_the_vault_are_still_pipeline_links(tmp_path: Path) -> None:
    """Final lints the export inside the vault, where ``runs/<id>/05-final.md`` exists: still a pipeline link."""
    run = tmp_path / "vault" / "runs" / "r1"
    (run / "deliverables").mkdir(parents=True)
    (run / "05-final.md").write_text("x\n")
    (issue,) = lint_markdown("See [[runs/r1/05-final]].\n", "README.md", root=run / "deliverables")
    assert issue.rule == "pipeline-wikilink"


def test_pipeline_links_in_frontmatter_are_reported() -> None:
    text = "---\nsource: \"[[01-ingestion]]\"\n---\n# Title\n"
    assert rules(text) == [("pipeline-wikilink", 2)]


# ---------------------------------------------------------------------- meta-commentary


@pytest.mark.parametrize(
    "sentence",
    [
        "In the previous revision the gain was fixed.",
        "The earlier revision used a PI controller.",
        "This revision adds impurity seeding.",
        "As proposed in review, the controller is now gain-scheduled.",
        "As suggested by the reviewers, we added a sweep.",
        "Review pointed out that the table was inconsistent.",
        "The reviewer asked for more plots.",
        "Reviewers noted a unit error.",
        "The cross-check found a sign error.",
        "The crosscheck flagged the missing test.",
        "This was addressed in the second fix round.",
        "After the fix pass the tests pass.",
        "Execution round 3 changed no code.",
        "The critique raised three issues.",
        "In response to the critique, the model was extended.",
        "Per the adjudicator, the scope is limited.",
        "The hard acceptance criterion AC-3 is met.",
        "All acceptance criteria from 02-strategy hold.",
        "This satisfies criterion AC-2.",
        "Values are taken from the ingestion report.",
        "The execution brief asked for three scenarios.",
        "See the strategy note for the options.",
        "The first review round found nothing.",
        # missed in the 2026-09-28 thesis (plurals, review as the cause, the ingestion's numbers)
        "It replaces the 3.52 MeV integer-mass convention used in earlier revisions.",
        "The coordinated split exists because review identified a real defect in the static one.",
        "The ingestion's 1.85 is plausibly a separatrix elongation.",
        "Earlier versions of this document used 3.52 MeV.",
        "In previous revisions the guard was a ramp.",
        "Following review, the gain was reduced.",
        "Both fix passes changed two files.",
    ],
)
def test_meta_commentary_is_major(sentence: str) -> None:
    (issue,) = only(f"Intro.\n\n{sentence}\n", "meta-commentary")
    assert (issue.severity, issue.line) == ("major", 3)
    assert "finished work" in issue.message


@pytest.mark.parametrize(
    "sentence",
    [
        "We cross-check the model against ITER data.",
        "The cross-check against the analytic solution agrees to 1 %.",
        "A critique of the Lawson criterion is out of scope.",
        "The critique by Wesson (2011) still applies.",
        "The review by Doyle et al. (2007) summarises the scaling laws.",
        "A 2019 review found that seeding raises the density limit.",
        "Peer review is not required for internal reports.",
        "The acceptance test runs in CI.",
        "The controller revises its setpoint every 10 ms.",
        "Revision 3 of the ITER baseline is used.",
        "The strategy is to keep $Q > 10$.",
        "Data ingestion is out of scope.",
        # false positives found on third-party Markdown and literature prose
        "The critique is well founded: Lawson's criterion ignores radiation losses [3].",
        "The critiques raised in [4] concern the choice of confinement scaling.",
        "The reviewers of the ITER Physics Basis noted the density limit.",
        "Systematic review findings suggest that seeding helps.",
        "In the literature review stage we screened 120 papers.",
        "A final note on units: all energies are in keV.",
        "## Final notes",
        "The cross-check passes for all 33 runs.",
        "The cross-check confirms the Jacobian.",
        "Execution passes all 40 unit tests.",
        "The fix passes all tests.",
        "Compare it with the previous revision of the file in git.",
        "This revision of the standard (IEEE 754-2019) adds fused multiply-add.",
        "The reviewer should check the port layer first.",
        "The ingestion of impurities raises $Z_{eff}$.",
    ],
)
def test_ordinary_prose_is_not_meta_commentary(sentence: str) -> None:
    assert only(f"{sentence}\n", "meta-commentary") == []


def test_meta_commentary_is_case_insensitive_and_once_per_line() -> None:
    issues = only("AS PROPOSED IN REVIEW, and in the previous revision too.\n", "meta-commentary")
    assert len(issues) == 1 and "AS PROPOSED IN REVIEW" in issues[0].message


# ---------------------------------------------------------------------- tables


THESIS_TABLE = """| Quantity | Symbol | Value |
|---|---|---|
| Absolute error | $|e|$ | 0.1 |
| Relative error | $|e|/|x|$ | 0.02 |
| Power | $P$ | 500 |
"""


def test_pipe_inside_math_in_a_table_row_is_major() -> None:
    issues = lint_markdown(THESIS_TABLE, "doc.md")
    assert [(i.rule, i.severity, i.line) for i in issues] == [
        ("gfm-table-pipe-in-math", "major", 3),
        ("gfm-table-pipe-in-math", "major", 4),
    ]
    assert "\\lvert" in issues[0].message and "5 cells (the table has 3)" in issues[0].message


def test_a_row_of_the_right_width_is_never_blamed_on_math() -> None:
    """``$R=$6.08 m | ok`` in a renderer's row: a math span found across the row may straddle a cell boundary."""
    table = "| a | b | c |\n|---|---|---|\n| centroid at $R=$6.08 m | $x | ok |\n"
    assert only(table, "gfm-table-pipe-in-math") == [] and only(table, "gfm-table-columns") == []


def test_pipe_inside_math_in_the_header_breaks_the_table() -> None:
    table = "| $|x|$ | y |\n|---|---|\n| 1 | 2 |\n"
    (issue,) = lint_markdown(table, "doc.md")
    assert (issue.rule, issue.line) == ("gfm-table-pipe-in-math", 1)
    assert "does not render" in issue.message


def test_table_column_mismatch_is_major() -> None:
    table = "| a | b |\n|---|---|\n| 1 | 2 | 3 |\n| 4 |\n| 5 | 6 |\n"
    issues = lint_markdown(table, "doc.md")
    assert [(i.rule, i.line) for i in issues] == [("gfm-table-columns", 3), ("gfm-table-columns", 4)]
    assert "3 cells but the table has 2" in issues[0].message


def test_header_and_delimiter_mismatch_is_reported() -> None:
    (issue,) = lint_markdown("| a | b | c |\n|---|---|\n", "doc.md")
    assert (issue.rule, issue.line) == ("gfm-table-columns", 1) and "does not render" in issue.message


def test_pipe_in_inline_code_splits_a_cell_too() -> None:
    table = "| op | meaning |\n|---|---|\n| `a|b` | or |\n| `a\\|b` | escaped |\n"
    (issue,) = lint_markdown(table, "doc.md")
    assert (issue.rule, issue.line) == ("gfm-table-columns", 3) and "escape it as `\\|`" in issue.message


def test_tables_without_outer_pipes_and_in_blockquotes() -> None:
    assert lint_markdown("a | b\n--- | :---:\n1 | 2\n", "doc.md") == []
    quoted = "> | a | b |\n> |---|---|\n> | $|x|$ | 2 |\n"
    assert rules(quoted) == [("gfm-table-pipe-in-math", 3)]


def test_pipes_outside_tables_are_fine() -> None:
    text = "The norm $|x|$ is small, and a | b is prose.\n\n---\n\nabs(x) | y\n"
    assert lint_markdown(text, "doc.md") == []


def test_table_ends_at_a_blank_line() -> None:
    text = "| a | b |\n|---|---|\n| 1 | 2 |\n\nx | y | z\n"
    assert lint_markdown(text, "doc.md") == []


# ---------------------------------------------------------------------- placeholders


@pytest.mark.parametrize(
    "token",
    ["{{P_fus}}", "{{ value }}", "TODO", "TODO:", "TBD", "FIXME", "XXX", "Lorem ipsum", "[citation needed]",
     "[insert figure 3]", "<insert table>"],
)
def test_placeholders_are_major(token: str) -> None:
    (issue,) = only(f"Fusion power is {token} MW.\n", "placeholder")
    assert issue.severity == "major"


@pytest.mark.parametrize("text", ["A todo list.", "TODOs are tracked elsewhere.", "`TODO` in code.", "$x^{{2}}$"])
def test_placeholder_lookalikes_pass(text: str) -> None:
    assert only(text + "\n", "placeholder") == []


# ---------------------------------------------------------------------- null-rendering


def test_null_in_a_numeric_table_column_is_minor() -> None:
    table = "| Case | Q | Note |\n|---|---|---|\n| A | 10.2 | fine |\n| B | n/a | None |\n| C | NaN | ok |\n"
    issues = only(table, "null-rendering")
    assert [(i.severity, i.line) for i in issues] == [("minor", 4), ("minor", 5)]
    assert "`Q`" in issues[0].message


def test_null_in_a_text_column_is_fine() -> None:
    table = "| Feature | Support |\n|---|---|\n| Locks | None |\n| Hooks | optional |\n"
    assert lint_markdown(table, "doc.md") == []


def test_python_none_is_exact_but_other_nulls_ignore_case() -> None:
    """The allocator README's ``Loops`` column says "none" in English; a renderer writes ``None``, ``NaN``, ``N/A``."""
    table = "| Op | Loops |\n|---|---|\n| malloc | 0 |\n| free | none |\n| realloc | **none** |\n"
    assert lint_markdown(table, "doc.md") == []
    for token in ("None", "NaN", "N/A", "NULL", "-nan"):
        (issue,) = only(table.replace("| none |", f"| {token} |"), "null-rendering")
        assert issue.line == 4


@pytest.mark.parametrize(
    "text", ["Q = nan at t = 5 s.", "Peak power: None MW after 3 s.", "The gain (n/a) at 2 Hz.", "P = NaN W at 4 s."]
)
def test_null_values_in_prose_near_numbers_are_minor(text: str) -> None:
    (issue,) = only(text + "\n", "null-rendering")
    assert issue.severity == "minor"


@pytest.mark.parametrize(
    "text",
    ["Limitations: None.", "Status: none of the 5 runs failed.", "The null hypothesis is rejected at p = 0.05.",
     "Result: `nan` 5", "Seeding: none until t = 5 s."],
)
def test_ordinary_nulls_in_prose_pass(text: str) -> None:
    assert only(text + "\n", "null-rendering") == []


# ---------------------------------------------------------------------- unbalanced-math


@pytest.mark.parametrize(
    ("text", "line"),
    [
        ("The price $x + y is broken.\n", 1),
        ("Fine $a$ and broken $b.\n", 1),
        ("Intro.\n\nText $$x = 1 and more.\n", 3),
        ("A\n\n$$\n\\frac{a}{b}\n", 3),
    ],
)
def test_unbalanced_math_is_major(text: str, line: int) -> None:
    (issue,) = only(text, "unbalanced-math")
    assert (issue.severity, issue.line) == ("major", line)


@pytest.mark.parametrize(
    "text",
    [
        "Costs $5 and $10 per shot.",
        "An escaped \\$ sign and $x$.",
        "$$ E = mc^2 $$",
        "Inline display $$x^2$$ in text.",
        "Math $a +\nb$ across lines of one paragraph.",
        "A dollar $ followed by a space.",
        "`$ unbalanced` in code.",
    ],
)
def test_balanced_math_and_currency_pass(text: str) -> None:
    assert only(text + "\n", "unbalanced-math") == []


def test_math_does_not_span_paragraphs_or_list_items() -> None:
    """Paired across the break, these would pass; a stray closing ``$`` (space after it) is literal."""
    assert [i.line for i in only("Open $a\n\nclose b$ here.\n", "unbalanced-math")] == [1]
    assert [i.line for i in only("- item $a\n- item b$\n", "unbalanced-math")] == [1]
    assert only("Open $a\nclose b$ here.\n", "unbalanced-math") == []


def test_unclosed_display_block_hides_the_rest_but_is_reported() -> None:
    text = "$$\nx\n\nAs proposed in review, [[01-ingestion]].\n"
    assert rules(text) == [("unbalanced-math", 1)]
    assert rules("$$\n\\tau = {{tau}} TODO\n$$\n") == [("placeholder", 2)]  # placeholders count inside math


# ---------------------------------------------------------------------- templates


@pytest.mark.parametrize("name", ["docs/document_template.md", "report.tmpl.md", "Thesis.J2.md", "TEMPLATE.md"])
def test_template_fields_are_not_placeholders_and_filter_pipes_do_not_split_cells(name: str) -> None:
    """The thesis rendered ``docs/document_template.md``; its fields are markup, but its prose ships."""
    text = (
        "Confinement $\\tau_E={{trim.tauE|.3f}}$ s, {% if ok %}stable{% endif %}.\n\n"
        "| Quantity | Value |\n|---|---|\n| Q | {{trim.Q|.1f}} |\n\n"
        "As quoted in [[01-ingestion]], TODO.\n"
    )
    assert [(i.rule, i.line) for i in lint_markdown(text, name)] == [("pipeline-wikilink", 7), ("placeholder", 7)]
    assert ("placeholder", 1) in [(i.rule, i.line) for i in lint_markdown(text, "document.md")]


# ---------------------------------------------------------------------- code is never linted


def test_fenced_and_inline_code_are_skipped() -> None:
    text = (
        "```\n[[01-ingestion]] TODO $ | x |\n```\n"
        "  ```python\n  # the previous revision\n  ```\n"
        "````md\n```\nstill code TODO\n```\n````\n"
        "``code with ` backtick [[05-final]]`` and `TBD`.\n"
    )
    assert lint_markdown(text, "doc.md") == []


def test_indented_code_blocks_are_skipped() -> None:
    """Four spaces (or a tab) after a blank line, outside a list, is a code block, as Obsidian renders it."""
    text = "Some text.\n\n    $ export PATH=$HOME/bin:$PATH\n    [[01-ingestion]] TODO\n\n\tmore TODO\n\nBack TODO.\n"
    assert rules(text) == [("placeholder", 8)]


def test_indented_lines_that_are_not_code_are_linted() -> None:
    """A list item's indented paragraph and a paragraph's indented continuation line are prose."""
    assert rules("- item\n\n    continued TODO\n") == [("placeholder", 3)]
    assert rules("A paragraph\n    continued TODO\n") == [("placeholder", 2)]


@pytest.mark.parametrize("text", ["Add the toolchain to $PATH first.", "Set ${CC} and $CC_FLAGS.", "Run $HOME/bin/sim."])
def test_shell_variables_are_not_unbalanced_math(text: str) -> None:
    assert only(f"{text}\n", "unbalanced-math") == []


def test_an_unclosed_fence_swallows_the_rest() -> None:
    assert lint_markdown("```\nTODO\n", "doc.md") == []


def test_line_numbers_survive_crlf_and_frontmatter() -> None:
    text = "---\r\na: 1\r\n---\r\n\r\nTODO\r\n"
    assert rules(text) == [("placeholder", 5)]


# ---------------------------------------------------------------------- broken-link


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    for rel in ("plots/burn.png", "figures/fig 1.png", "chapters/ch2.md", "Other Note.md", "FreeRTOS-Kernel/README.md",
                "sub/data.csv"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
    (tmp_path / "outside.png").write_bytes(b"x")
    return root


@pytest.mark.parametrize(
    "link",
    [
        "![[burn.png]]",
        "![[plots/burn.png]]",
        "![[burn.png|400]]",
        "![](plots/burn.png)",
        "![fig](<figures/fig 1.png>)",
        "![fig](figures/fig%201.png)",
        "[chapter 2](chapters/ch2.md#results)",
        "[[Other Note]]",
        "[[ch2]]",
        "[[#Local heading]]",
        "[data](sub/data.csv \"title\")",
        "[site](https://example.org/x.png)",
        "[mail](mailto:a@b.c)",
        "[top](#model)",
    ],
)
def test_resolvable_links_pass(tree: Path, link: str) -> None:
    assert lint_markdown(f"See {link}.\n", "doc.md", root=tree) == []


@pytest.mark.parametrize(
    ("link", "why"),
    [
        ("![[missing.png]]", "not found"),
        ("![](plots/missing.png)", "not found"),
        ("[chapter](chapters/ch9.md)", "not found"),
        ("[[Missing Note]]", "not found"),
        ("![](../outside.png)", "not found"),
        ("[kernel](FreeRTOS-Kernel/README.md)", "not found"),
        ("![](/home/user/plot.png)", "absolute paths"),
        ("![](~/plot.png)", "absolute paths"),
    ],
)
def test_broken_links_are_major(tree: Path, link: str, why: str) -> None:
    (issue,) = lint_markdown(f"See {link}.\n", "doc.md", root=tree)
    assert (issue.rule, issue.severity) == ("broken-link", "major") and why in issue.message


def test_links_resolve_relative_to_the_file(tree: Path) -> None:
    links = "[back](<../Other Note.md>) ![](../plots/burn.png) [s](ch2.md)\n"
    assert lint_markdown(links, "chapters/ch2.md", root=tree) == []
    (issue,) = lint_markdown("[back](../chapters/ch3.md)\n", "chapters/ch2.md", root=tree)
    assert issue.rule == "broken-link"


def test_link_targets_the_filesystem_cannot_name_are_broken_not_a_crash(tree: Path) -> None:
    """A prose wikilink over 255 bytes (``ENAMETOOLONG``) or a ``%00`` (an embedded NUL) used to raise out of the
    lint, and so out of the stage after a paid Claude Code session."""
    title = "Reiter, Wolf and Kever: burn condition, helium particle confinement and exhaust efficiency " * 3
    (tree / "doc.md").write_text(f"See [[{title}]].\n\n[x](foo%00bar.md)\n", encoding="utf-8")
    issues = lint_deliverables(tree)
    assert [(i.path, i.line, i.rule) for i in issues] == [("doc.md", 1, "broken-link"), ("doc.md", 3, "broken-link")]


def test_links_are_not_checked_without_a_root() -> None:
    assert lint_markdown("![[missing.png]] [x](nope.md)\n", "doc.md") == []


def test_relinked_vault_embeds_resolve_from_the_vault(tmp_path: Path) -> None:
    deliverables = tmp_path / "vault" / "runs" / "r1" / "deliverables"
    (deliverables / "plots").mkdir(parents=True)
    (deliverables / "plots" / "burn.png").write_bytes(b"x")
    text = "![[runs/r1/deliverables/plots/burn.png]] ![[runs/r1/deliverables/plots/gone.png]]\n"
    (issue,) = lint_markdown(text, "document.md", root=deliverables)
    assert issue.rule == "broken-link" and "gone.png" in issue.message


# ---------------------------------------------------------------------- lint_deliverables


def test_lint_deliverables_walks_markdown_and_skips_excluded(tree: Path) -> None:
    (tree / "README.md").write_text("Run `make`. TODO\n", encoding="utf-8")
    (tree / "chapters" / "ch2.md").write_text("See [[01-ingestion]].\n\nAs proposed in review.\n", encoding="utf-8")
    (tree / "notes.txt").write_text("TODO\n", encoding="utf-8")
    for rel in (".maf/execution-r1.md", "inputs/user.md", "FreeRTOS-Kernel/README.md", ".git/x.md", "build/tmp/a.md"):
        path = tree / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("TODO [[05-final]]\n", encoding="utf-8")

    issues = lint_deliverables(tree, (*DEFAULT_EXCLUDE, "build/tmp"))
    assert [(i.path, i.line, i.rule) for i in issues] == [
        ("README.md", 1, "placeholder"),
        ("chapters/ch2.md", 1, "pipeline-wikilink"),
        ("chapters/ch2.md", 3, "meta-commentary"),
    ]
    assert {i.path for i in lint_deliverables(tree)} >= {"build/tmp/a.md"}


def test_lint_deliverables_does_not_follow_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("TODO\n", encoding="utf-8")
    root = tmp_path / "ws"
    root.mkdir()
    os.symlink(outside / "secret.md", root / "link.md")
    os.symlink(outside, root / "linkdir")
    assert lint_deliverables(root) == []


def test_lint_deliverables_skips_huge_files_and_missing_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert lint_deliverables(tmp_path / "nope") == []
    (tmp_path / "big.md").write_text("TODO " * 10 + "\n", encoding="utf-8")
    monkeypatch.setattr(lint, "MAX_FILE_BYTES", 10)
    assert lint_deliverables(tmp_path) == []


def test_lint_deliverables_resolves_links_across_the_tree(tree: Path) -> None:
    (tree / "chapters" / "ch2.md").write_text("![[burn.png]] ![[gone.png]] [[Other Note]]\n", encoding="utf-8")
    issues = lint_deliverables(tree)
    assert [(i.path, i.rule) for i in issues] == [("chapters/ch2.md", "broken-link")]


# ---------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("rel", "patterns", "expected"),
    [
        (".git/config", (".git",), True),
        ("src/pkg/__pycache__/m.pyc", ("__pycache__",), True),
        ("src/m.pyc", ("*.pyc",), True),
        ("build/tmp/a.md", ("build/tmp",), True),
        ("src/build/tmp/a.md", ("build/tmp",), False),
        ("docs/readme.md", (".git", "*.pyc"), False),
        ("Inputs/a.md", ("inputs",), False),
    ],
)
def test_excluded(rel: str, patterns: tuple[str, ...], expected: bool) -> None:
    assert excluded(rel, patterns) is expected


def test_issue_str_and_blocking() -> None:
    issues = [
        LintIssue("placeholder", "major", "a.md", 3, "unfilled placeholder `TODO`"),
        LintIssue("null-rendering", "minor", "a.md", 4, "x"),
        LintIssue("pipeline-wikilink", "critical", "", 1, "y"),
    ]
    assert str(issues[0]) == "[major] placeholder a.md:3: unfilled placeholder `TODO`"
    assert str(issues[2]) == "[critical] pipeline-wikilink line 1: y"
    assert blocking(issues) == [issues[0], issues[2]]


def test_messages_never_contain_live_links() -> None:
    (issue,) = lint_markdown("See [[01-ingestion]].\n", "doc.md")
    assert "`[[01-ingestion]]`" in issue.message
    (issue,) = lint_markdown("Use `x` TODO\n", "doc.md")
    assert issue.message.count("`") == 2


def test_the_thesis_failure_shapes_together() -> None:
    """The shapes found in the 2026-09-28 thesis deliverable: pipeline links cited as authority, fix-round
    commentary, a table broken by ``$|x|$``, a renderer's ``n/a`` and an embed of a plot that was never written."""
    text = (
        "#### Burn control\n\n"
        "The confinement time follows [[01-ingestion]] (see [[03-execution]] for the runs).\n"
        "As proposed in review, the controller now saturates. The earlier revision did not.\n\n"
        "| Case | $|\\Delta T|$ max | Q |\n"
        "|---|---|---|\n"
        "| nominal | 0.4 | 10.1 |\n"
        "| mismatch | $|e|$ | n/a |\n"
        "| seeded | 0.2 | 9.8 |\n\n"
        "![[runaway.png]]\n"
    )
    root_issues = sorted((i.rule, i.severity, i.line) for i in lint_markdown(text, "document.md"))
    assert root_issues == [
        ("gfm-table-pipe-in-math", "major", 6),
        ("gfm-table-pipe-in-math", "major", 9),
        ("meta-commentary", "major", 4),
        ("pipeline-wikilink", "critical", 3),
        ("pipeline-wikilink", "critical", 3),
    ]
