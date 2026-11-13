# The model is confined to judgement checks

Rule checks always run for their submission type. A model never chooses which ones run, never sets severity, never routes and never removes or softens a rule finding. A model owns only the judgement checks (`PURPOSE_UNCLEAR`, `RECEIPT_MISMATCH`, `ALCOHOL_NOT_PERMITTED`, and the justification half of `WEEKEND_SPEND_UNJUSTIFIED`), plans retrieval, and plans what each judgement check investigates. Every model finding then passes a deterministic verification: the code must be in the closed set, the clause must be one that was retrieved, each quote must appear verbatim in the named field, and any cited precedent must be one that was supplied. The cost of this is that the agent can do less. The benefits are that a model can never switch off a control, rule findings stay testable as pure functions, a prompt injection can affect at most one submission's judgement findings (and only under the auto-approve threshold), and model comparisons measure judgement rather than whether a planner remembered to run a check.

## Considered Options

- **The model selects the checks for each submission (the original draft).** Rejected: when the planner misses a check, the check silently doesn't run.
- **A model reflection pass in verification.** Deferred: it costs an extra call and is non-deterministic. It gets added only if measured precision is poor.
