# Task: triage the user's request

Read the request and any attached file names. Decide:
- `execution_mode`: `code` (software that must build and pass tests), `prose` (a document written
  from gathered knowledge), or `mixed` (a document backed by code, simulations or plots that must
  actually be run).
- `gemini_instructions`: precisely what Gemini should read, search and extract for the later stages.
- `search_queries`: concrete web searches (an empty list if the web is not needed).
- `deliverable`: the final artifact and its acceptance bar.

## User request

{{brief}}

## Attached files

{{files}}
