# Task: strategy and execution brief

Using the ingestion report, lay out several distinct options or angles, compare them honestly,
choose one, and write an execution brief concrete enough for Claude to produce the definitive
artifacts without further clarification.

## Acceptance criteria

The cross-check judges the work against `## Acceptance Criteria`, and the final report gives a verdict on
each: the run does not count as complete while a hard criterion is unmet. Every unmet hard criterion is a
critical issue that sends the work back for another paid execution and cross-check round, so calibrate them:

- At most {{max_hard}} criteria are `hard`. Make a criterion hard only when the user's request directly requires
  it and it can be demonstrated within this run's budget; mark every other criterion, however desirable, `soft`.
- Avoid absolute provenance or coverage demands ("every number recomputed", "all cases with full metrics",
  "zero omissions") unless the request itself asks for them: ask for what the request asks, checked on the
  cases that matter.
- Each criterion must be testable by someone who has only the deliverables. A hard criterion names how it is
  checked: the command, measurement, threshold or file that decides it, never an aspiration such as "robust" or
  "well written".

{{budget}}

Write them in the line grammar of the output format below (`- AC-<n> [hard]: <criterion>` or
`- AC-<n> [soft]: <criterion>`).

{{criteria_guidance}}

## User request

{{brief}}

{{inputs}}

{{review_note}}

{{format_spec}}
