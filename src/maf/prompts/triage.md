# Task: triage the user's request

Read the request and any attached file names. Decide:
- `execution_mode`: `code` (software that must build and pass tests), `prose` (a document written
  from gathered knowledge), or `mixed` (a document backed by code, simulations or plots that must
  actually be run).
- `gemini_instructions`: precisely what Gemini should read, search and extract for the later stages. The
  later stages have no web access and may cite only the sources Gemini verifies, so name the primary
  sources (papers, standards, datasheets, official pages) whose records and passages Gemini must check, and
  the specific numbers, formulas or claims it must find quoted in them.
- `search_queries`: concrete web searches (an empty list if the web is not needed), including searches
  that locate those primary sources.
- `deliverable`: the final artifact and its acceptance bar.

## User request

{{brief}}

## Attached files

{{files}}
