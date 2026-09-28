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
   [optional human gate with --review: user picks/edits direction]
03 Execution      Claude (Code for code, Messages for prose) consumes ingestion + strategy → definitive artifacts
04 Cross-check    All three critique independently (## Issues with severity) → one rebuttal round →
                  Claude applies accepted fixes; ChatGPT adjudicates disputes.
                  Unresolved critical issues → loop back to 03 (max 2 loops).
05 Final          Claude assembles the final deliverable; run index updated
```

## Shared memory: the Obsidian vault

- The vault is at `~/Obsidian/MultiAgent`, configurable. The snap uses classic
  confinement, so any path works.
- Each run gets its own folder: `runs/<YYYY-MM-DD>-<slug>/`, containing:
  - `run.md`: run index. Frontmatter holds status, stage and spend. The body has
    links to every handoff and a per-agent cost table.
  - `01-ingestion.md`, `02-strategy.md`, `03-execution.md`,
    `04-crosscheck.md` (plus critique/rebuttal notes), `05-final.md`
  - `assets/`: plots and images, embedded with `![[...]]`
  - `deliverables/`: final artifacts copied into the vault (thesis, allocator
    source)
- Large code trees and simulation data are **outside** the vault, in
  `workspaces/<run_id>/`, and handoffs link to them.
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

## Budget

- The per-run cap is **$25 USD** by default, set with `--budget`.
- A cost ledger for each run records usage from every provider call (and
  `total_cost_usd` from Claude Code JSON). Before each call it checks the
  worst-case cost against the remaining budget, and it stops cleanly when the
  cap is reached. Spend per agent appears in `run.md`.
- A price table in config carries an effective date. Gemini 3.8 Flash prices
  double on 2027-01-01.

## Execution sandbox

- Claude Code runs with its cwd set to `workspaces/<run_id>/`, with an allowed
  list of tools (edit, build, test, QEMU) and no writes outside the workspace.
- Toolchain: gcc, arm-none-eabi-gcc and newlib, qemu-system-arm, pandoc, a
  FreeRTOS kernel clone in the workspace, and numpy/scipy/matplotlib in the
  project venv.

## ChatGPT integration

- `maf serve` starts an MCP server (streamable HTTP on 127.0.0.1) with the tools
  `start_run(brief, files)` (a write tool), `get_run_status(run_id)`,
  `get_run_result(run_id)` and `list_runs`.
- `start_run` returns a run_id immediately, and the pipeline runs in the
  background. This is required because ChatGPT enforces a hard 1-minute limit on
  each tool call.
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
