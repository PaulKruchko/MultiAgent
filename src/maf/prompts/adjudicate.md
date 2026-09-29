# Task: adjudicate disputed issues (round {{round}})

For each disputed issue, read the critic's claim and the author's response, then rule `fix` (the
author must address it) or `wontfix` (the critic is wrong or it is out of scope). Be impartial, and
judge against the acceptance criteria and the evidence below. Rule on every issue listed, and only on
those, with one line each.

- An issue flagged as an unmet acceptance criterion is critical by definition. Rule `wontfix` only when the
  evidence below demonstrably shows the criterion is met, and cite that evidence in the ruling. The
  author's assertion is not evidence; when in doubt, rule `fix`.
- `SRC` issues come from a source audit that checked the references against web records; rule `wontfix`
  only when the evidence shows the audit is wrong. `LINT` issues come from a deterministic lint; rule
  `wontfix` only for a demonstrable false positive.

## Disputed issues

{{disputes}}

## Evidence

{{inputs}}

{{format_spec}}
