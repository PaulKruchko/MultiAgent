# Task: adjudicate disputed issues (round {{round}})

For each disputed issue, read the critic's claim and the author's response, then rule `fix` (the
author must address it) or `wontfix` (the critic is wrong or it is out of scope). Be impartial, and
judge against the acceptance criteria, the user's request and the evidence below. Rule on every issue
listed, and only on those, with one line each.

- An issue flagged as an unmet acceptance criterion is critical by definition. Rule `wontfix` only when the
  evidence below demonstrably shows the criterion is met, and cite that evidence in the ruling. The
  author's assertion is not evidence; when in doubt, rule `fix`.
- `relax` applies only to an issue flagged as an unmet acceptance criterion, when the criterion is
  over-specified relative to the user's request: it demands more than the request asks for (typically an
  absolute provenance or coverage demand, such as every computed number recomputed or full metrics for all
  cases) and is not needed to judge whether the request was met. The issue then becomes major and the
  criterion soft for the rest of the run: it is still worth improving, but it no longer blocks the run. The
  ruling's text is a one-line justification that names what the request actually asks for. Never relax a
  criterion the request explicitly requires, a clean-room, repeatability or source-audit criterion, or one
  that is merely hard to meet.
- `SRC` issues come from a source audit that checked the references against web records; rule `wontfix`
  only when the evidence shows the audit is wrong. `LINT` issues come from a deterministic lint; rule
  `wontfix` only for a demonstrable false positive.

## User request

{{brief}}

## Disputed issues

{{disputes}}

{{accepted}}

## Evidence

{{inputs}}

{{format_spec}}
