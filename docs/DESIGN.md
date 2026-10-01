# MultiAgent — Design

Decisions settled in the design interview on 2026-09-28. This is the spec the
implementation must follow.

## Purpose

A personal, from-scratch Python framework that coordinates **ChatGPT (OpenAI)**,
**Gemini (Google)** and **Claude (Anthropic)** for two reasons at once:

- **Specialization/routing**: each model does the work it is best at.
- **Ensemble cross-checking**: the models critique each other's output before the
  final result is produced.

Single user (the author). It runs from the CLI, or is triggered from the ChatGPT
app through an MCP server.

## Roles

| Agent | Provider / default model | Responsibilities |
|---|---|---|
| ChatGPT | OpenAI Responses API, `gpt-6-sol` | Orchestrator *within stages*: triage of the raw request, routing instructions to Gemini, ideation, creative brainstorming, strategy, rough drafts, adjudication of disputes in cross-check |
| Gemini | google-genai SDK, `gemini-3.8-flash` | Large-context ingestion, massive document sets, web search (Google Search grounding), multimodal analysis (images/video/PDF) |
| Claude | `claude-opus-5-5`: Messages API for writing/editing; **Claude Code headless (`claude -p`)** for coding/execution in a sandboxed workspace | Deep reasoning, structural logic, long-horizon reasoning, editorial refinement, definitive code, final documents/reports |

Model IDs are pinned in config and can be overridden per stage. `--tier max`
switches ChatGPT to `gpt-6-astra` and Claude to `claude-fable-5-1`.

## Control flow

**Python owns the stage order.** This is a deterministic state machine and can
be resumed after a crash. ChatGPT decides the *content* of each stage, not the
sequence.

```
01 Ingestion      ChatGPT triages the request → writes routing brief → Gemini ingests/searches/parses → structured ingestion report
02 Strategy       ChatGPT consumes ingestion report → broad options/angles → chosen strategy + execution brief
   [optional human gate with --review or config review: true (--no-review skips it for one run): user picks/edits direction]
03 Execution      Claude (Code for code, Messages for prose) consumes ingestion + strategy → definitive artifacts
04 Cross-check    All three critique independently (## Issues with severity) → one rebuttal round →
                  Claude applies accepted fixes; ChatGPT adjudicates disputes.
                  Unresolved critical issues → loop back to 03 (max 2 loops), if the budget
                  can pay for another round and final.
05 Final          Claude assembles the final deliverable; run index updated
```

- Ingestion hands off **verified sources**: Gemini checks each source's record
  (authors, title, venue, year, DOI/URL) and quotes the passages later stages
  may cite, with section/page/table/equation locators. Later stages have no web
  access and may cite only these sources, never the pipeline's notes.
- Before the critiques, the cross-check lints the Markdown deliverables
  (`LINT` issues) and, when they cite references, has Gemini audit every
  reference with web search: exists, metadata correct, supports the claims
  attributed to it (`SRC` issues; internal notes and unfound works are
  critical, unsupported claims major, metadata errors minor). Both kinds are
  debated and fixed like critics' issues, and count toward unresolved criticals.
  Export, lint and audit see the same tree (the workspace without pipeline
  state, the kernel, `inputs/` and build output). Each defect is raised once: a
  cited pipeline link is a LINT issue only, and critics and final read the
  cross-check's lint, not execution's. After the fix pass, Python lints the
  tree again and Gemini audits the documents the fix changed; what they find
  is raised before the verdict, since the fixer has no web access. The
  auditor's text reaches the notes only as quoted data.
- An acceptance criterion that is not demonstrably met is a critical issue
  (major if marked soft), and adjudication rules it `wontfix` only on evidence. Critics also probe
  robustness: model-mismatch sensitivity, unexamined modelling choices, prose
  against data, tests and negative controls run once, clean rebuilds.
- Adjudication also sees the unmet criteria the author accepted, and may rule
  any unmet hard criterion `relax`: over-specified relative to the brief (for
  example "every computed number recomputed"), with a one-line justification.
  The issue becomes major and the criterion soft for the rest of the run:
  critics may raise it as major at most, and final reports it under
  `## Relaxed Criteria` without blocking. The last two unresolved criticals of
  the 2026-09-29 thesis rerun were such criteria.
- Before a `LOOP` goes back to execution, Python estimates another round from
  the run's ledger (the last round's spend, at least the smallest round it
  would start: an execution session, the cross-check's calls at their worst
  case and a fix session) plus final's reserve (the report's worst case and
  the clean room). If the run cannot pay for that, the loop is skipped
  (`Loop skipped: budget` in the cross-check note) and final runs, so the run
  ends `completed_with_issues` with the open criticals listed instead of
  running out of money mid-loop. The approved round keeps that promise even
  when it costs more than the last: its execution session holds back the
  cross-check, a fix session and final (as far as its own minimum allows), and
  every fix session keeps final's reserve. A fix session that cannot get its
  minimum while keeping it is skipped (`Fix pass skipped: budget`, its issues
  stay open) instead of stopping the run, and one that runs out of that cut
  budget counts its issues as not fixed.
- Final gates the result. For code and mixed runs, Claude Code rebuilds the
  exported deliverables in an empty copy (beside the workspace, which the
  session cannot read; generated files are dated older than their sources, so
  they are rebuilt) with the reproduction command the README documents;
  Python checks the exit code and log it wrote. Python also records its own
  `source-audit` verdict (the latest audit covered the shipped text and
  verified every reference) and `lint` verdict (no critical lint finding in
  the export). The final report must give a verdict (met, partial or unmet,
  with evidence) for every acceptance criterion. A hard criterion that is not
  met, the clean-room rebuild included, ends the run `completed_with_issues`.
  An extra round (`maf resume --extra-round`) gives execution those verdicts.
  It is exactly one execution and cross-check pass, then final: its
  cross-check never loops, even after a loop skipped for budget.

## Shared memory: the Obsidian vault

- The vault is at `~/Obsidian/MultiAgent`, configurable. The snap uses classic
  confinement, so any path works. Settings are validated when they load (unknown
  keys, unpriced models, sandbox limits), before a run spends anything.
- Each run gets its own folder: `runs/<YYYY-MM-DD>-<slug>/`, containing:
  - `run.md`: run index. Frontmatter holds status, stage and spend. The body has
    links to every handoff and a per-agent cost table.
  - `01-ingestion.md`, `02-strategy.md`, `03-execution.md`,
    `04-crosscheck.md` (plus critique/rebuttal notes), `05-final.md`
  - `assets/`: plots and images, embedded with `![[...]]`
  - `deliverables/`: final artifacts copied into the vault (thesis, allocator
    source). For code and mixed runs this is the whole workspace tree, without
    pipeline state (`.maf/`, `.claude/`), the provisioned FreeRTOS kernel, the
    user's `inputs/`, build output (in-source objects and binaries too) and
    empty sandbox placeholder files, capped at 200 MB. The config's
    `export_exclude` adds patterns to these built-in excludes (it never replaces
    them), and `export_include` brings back a hand-written file an exclude
    pattern catches. `deliverables/` is replaced as a whole, and `maf export <run_id>` redoes it
    from the workspace without model calls. A code or mixed run that stops
    `failed` or `budget_exceeded` after execution wrote files gets the same
    export, marked in run.md as partial and unverified, unless an earlier final
    already exported it (an extra round stopped): then `deliverables/` keeps
    final's verified export.
- Large code trees and simulation data are **outside** the vault, in
  `workspaces/<run_id>/`, and handoffs link to them. `workspaces/` defaults to
  the source checkout maf runs from (`~/MultiAgent/workspaces` for a clone
  there), or `~/.local/share/maf/workspaces` for an installed copy; the ChatGPT
  inbox sits next to it. Claude Code gets the venv maf runs from. run.md records
  each run's workspace path; when the current `workspaces_path` puts it
  elsewhere and nothing is there, `resume`/`export` refuse the run (exit 2,
  before any provider is built), so a moved default never starts a paid session
  in an empty tree.
- Writes are atomic: write to a temporary file, then `os.replace`. The framework
  writes the files directly and does not use the REST plugin.
- Only the property keys `tags`, `aliases` and `cssclasses` are used; Obsidian
  1.9+ dropped `tag`, `alias` and `cssclass`. Links are wikilinks. Math is
  `$…$`/`$$…$$`. Mermaid is allowed.

### Handoff contract (strict)

Every handoff is one `.md` file:

- YAML frontmatter with `run_id`, `stage`, `from`, `to`, `status`,
  `inputs: ["[[...]]"]`, `created`, `model`, `cost_usd`
- **Required H2 sections for each stage type**, e.g. Ingestion: `## Sources`,
  `## Key Facts`, `## Data Tables`, `## Open Questions`

Python parses and validates each handoff. If validation fails, the agent gets
**one repair attempt**; if that also fails, the run stops. Content fetched from
the web is kept as quoted data inside handoffs, never as instructions.

Strategy acceptance criteria are numbered, testable lines that Python reads,
`- AC-<n> [hard|soft]: ...`, a grammar in the strategy's format spec (Python
rewrites off-grammar criteria instead of repairing them). Code and mixed runs
always get a hard clean-room reproduction criterion (and a 3-run repeatability
one); documents that cite sources get a hard criterion that every reference
passes the source audit. At most 6 criteria are hard, those included: each one
the brief directly requires, demonstrable within the run's budget, and naming
how it is checked. Everything else is soft, and absolute provenance or coverage
demands ("every", "all") appear only when the brief makes them. More hard
criteria fail the strategy's validation (one repair, as usual). Every unmet
hard criterion is a critical issue that loops the cross-check: the thesis rerun
of 2026-09-29 had ten, and they drove the loops that exhausted its budget.
The strategy prompt states the run's budget, the loop cap and what one Claude
Code work session may spend, so "demonstrable within the run's budget" can be
judged. Hand-edited strategies at the review gate are not held to the cap.

## Budget

- The per-run cap is **$25 USD** by default, set with `--budget`.
- A cost ledger for each run records usage from every provider call (and
  `total_cost_usd` from Claude Code JSON). Before each call it checks the
  worst-case cost against the remaining budget, and it stops cleanly when the
  cap is reached. Spend per agent appears in `run.md`. The clean-room rebuild
  leaves the final report's worst case unspent.
- Running out of the run's budget is never a failure. A Claude Code work
  session (execution, fix pass) that hits its `--max-budget-usd` after the
  run's remaining budget clamped it below the session budget it asked for ends
  the run `budget_exceeded`, which `maf resume --budget` continues (one that
  used up its own, unclamped budget still fails the run). A work session
  that the clamp would leave below its minimum is not started at all: an
  execution stops the run `budget_exceeded` (checked before the sandbox
  preflight is paid for too) and a fix pass is skipped. The minimum is
  `claude_code_min_session_usd` ($3 by default), but never more than
  `claude_code_min_session_share` (a quarter) of the run's budget, so a small
  run such as ChatGPT's $5 default still gets its execution session. The
  stop names the budget to resume with, and resuming with exactly that works:
  it covers one turn of headroom, the preflight a resumed process pays again,
  and a minimum that grows with the raised cap. (On 2026-09-29 a session
  clamped to $1.43 ran out and the run ended `failed` with a nearly finished
  thesis in its workspace.)
- A code or mixed run that ends `failed` or `budget_exceeded` after an
  execution pass wrote files still exports its workspace to `deliverables/`
  (no model calls), and run.md marks it partial and unverified: no acceptance,
  clean-room or source-audit gate ran. An export error never replaces the
  run's own error.
- A price table in config carries an effective date. Gemini 3.8 Flash prices
  double on 2027-01-01.

## Execution sandbox

- Claude Code runs with its cwd set to `workspaces/<run_id>/`, with an allowed
  list of tools (edit, build, test, QEMU) and no writes outside the workspace.
  The one exception is a private 0700 `TMPDIR` (`/tmp/maf-<random>`). It lives
  outside the workspace because the sandbox creates Unix sockets under
  `TMPDIR`, and long workspace paths overflow the 108-byte socket path limit.
  That overflow was the incident of 2026-09-28.
- A Claude Code session may run 90 minutes (`claude_code_timeout_s`). One that
  times out is charged its worst case (the cap stays hard), and an execution or
  fix session then gets exactly one continuation session in the same
  workspace: it inspects what the previous session left, finishes only what is
  incomplete, reruns the reproduction and tests, and keeps each command under
  10 minutes. A second timeout ends the run `failed`. (On 2026-09-29 the first
  execution was killed at 60 minutes; a manual resume finished it for $1.13.)
  Streaming the session (`--output-format stream-json`) would let a killed
  session's spend be estimated; it is not used until a recorded stream confirms
  its result event.
- A Bash command may run up to 75 % of the session timeout (Claude Code's own
  cap is 10 minutes), so long simulations and QEMU batteries finish.
- Before the first code-mode Claude Code call, every run does a **sandbox
  preflight**: a nonce-hash check that proves sandboxed Bash can run and can
  write to `TMPDIR`. If the sandbox fails, the run stops immediately as
  `failed`.
- If a run reaches the cross-check loop cap (or skips a loop for budget) with unresolved critical issues, or
  final finds a hard acceptance criterion unmet (the clean-room rebuild
  included), it ends as `completed_with_issues` (CLI exit code 2), never as
  `completed`.
- Toolchain: gcc, arm-none-eabi-gcc and newlib, qemu-system-arm, pandoc, a
  FreeRTOS kernel clone in the workspace, and numpy/scipy/matplotlib in the
  project venv.
- Claude Code has no network, so deliverables cite only the ingestion's
  verified sources, never pipeline notes. They carry no remarks about
  revisions or reviews, take their numbers from the data, and reproduce from a
  clean copy with one documented command. Every test suite and negative control
  runs at least 3 times. After execution, Python lints the workspace Markdown
  (`maf.lint`: pipeline links, meta-commentary, broken tables, math and links,
  placeholders, rendered nulls) and lists critical and major findings in the
  execution note.

## ChatGPT integration

- `maf serve` starts an MCP server (streamable HTTP on 127.0.0.1) with the tools
  `start_run(brief, files)` (a write tool), `get_run_status(run_id)`,
  `get_run_result(run_id)` and `list_runs`.
- `start_run` returns a run_id immediately, and the pipeline runs in the
  background. This is required because ChatGPT enforces a hard 1-minute limit on
  each tool call. Status and result carry the open criticals, the unmet and the
  relaxed criteria, and a loop skipped for budget, so ChatGPT can say what was
  not verified even for a `completed` run.
- An MCP run's budget is capped at `mcp_max_budget_usd` ($5 by default). A code
  run on that budget gets its execution session (the session minimum is a
  quarter of a small run's budget), but little after it; CHATGPT.md explains.
- Stopping `maf serve` waits for the in-flight stage: the unit's
  `TimeoutStopSec` covers three Claude Code sessions plus a margin (5 hours at
  the 90-minute session timeout), since a session killed by systemd never
  reaches the cost ledger.
- Exposed through **OpenAI Secure MCP Tunnel** (tunnel-client), run as a systemd
  user unit. Needs manual steps: create the tunnel_id in the Platform dashboard
  and enable Developer Mode.
- Inside the pipeline, "ChatGPT" means the OpenAI API. The ChatGPT app only
  launches runs and inspects them.

## Extensibility

Each stage calls a pluggable backend (`run_stage(input_notes) -> output_note,
cost`). The OpenAI Agents API (public beta 2026-09-10) was evaluated and
rejected as the orchestrator: it is OpenAI-only, has no spend cap, keeps
session state on OpenAI's side, and `gpt-6-sol` support is unconfirmed. It can
be added later as a backend for a single stage.

## Demos (acceptance)

1. **Portable small-memory allocator.** The pipeline picks the design itself.
   Hard requirements:
   - O(1) worst case
   - no libc dependency
   - memory region supplied by the user
   - optional lock hooks
   - code size under 2 KB on Cortex-M

   It must pass tests on **POSIX** (gcc, unit and stress tests), on **FreeRTOS**
   (POSIX simulator port), and on **bare-metal** (Cortex-M3 under
   qemu-system-arm with semihosting).
2. **Master's-thesis-style research.** Subject: 0-D burn control of a DT burning
   plasma (ITER-like). Regulate fusion power and temperature against thermal
   runaway using fueling, auxiliary heating and impurity seeding, with a
   nonlinear controller. Deliverable: about 40–60 pages of Obsidian markdown
   with LaTeX equations, simulations actually run, and PNG plots. PDF export via
   pandoc is optional later.

## Build order

1. Core: stage runner, handoff validation, ledger, provider adapters, CLI
2. Tests using recorded provider fixtures, at zero cost
3. Cheap live smoke run
4. Allocator demo
5. Thesis demo
6. MCP server and tunnel

The total spend authorized while building is $150; stop and ask before going
past $100.
