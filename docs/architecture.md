# How SpendGuard works

Every submission runs through one LangGraph graph:

```text
intake ─┬─ run_rule_checks ─────────────────────────────────────┬─ verify ─ route ─┬─ execute ─ finalize
        └─ plan ─ retrieve_policy ─ run_judgement_checks ───────┘                  └─ human_review ─ execute ─ finalize
```

- **Intake** validates the submission, converts the amount to the policy's base currency (pounds) from the rates table, strips control and zero-width characters, caps field lengths, and redacts card numbers, emails and phone numbers before any prompt is built.
- **Rule checks and judgement checks run in parallel** and join at verification.

## Rules versus judgement

See [ADR 0001](adr/0001-model-confined-to-judgement-checks.md).

- **Rule checks** (category limits, receipts, duplicates, split purchases, approved suppliers, purchase orders, budgets, the weekend or holiday gate, and claimed exchange rates) are pure Python over the submission, the seed data and the policy table. Every rule check that applies to the submission type runs every time. A model never chooses which run, never sets severity and never removes a rule finding.
- **Judgement checks** (unclear purpose, receipt mismatch, alcohol, and whether weekend or holiday spend is justified) are the only work a model does. A model plans two to four retrieval queries, policy clauses are retrieved once from pgvector, and a tool-calling agent with a capped number of tool calls reads the case. Its tools are bound to the submission, so it can look up this employee, supplier, cost centre, the public holiday on the submission's date and precedents, and nobody else's.
- **Verify** is pure Python. A model finding survives only if its code is a judgement code, its clause was retrieved, every quote appears verbatim in the named field of the redacted text the model saw, and any precedent it cites was one it was given. Duplicates collapse. Severity comes from the policy, not the model. Every drop is recorded in the audit trail with its reason.

The cost is that the agent can do less. In return a model cannot switch off a control, and a prompt injection can affect at most one submission's judgement findings.

## Routing, review and decisions

- **Route** is a pure function. A submission is auto-approved only when it has no findings, no check was cleared by a precedent, every judgement check got a verdict that verify kept, its amount is at or under the policy's auto-approve threshold and `FORCE_HUMAN_REVIEW` is off. A check with no verdict, or whose flag or clearance verify drops, sends the submission to human review. Nothing is ever auto-rejected.
- **Recommendation** is a pure function, first match wins: a high-severity finding without a precedent gives reject; every finding having a precedent (or there being none) gives approve; otherwise there is no recommendation. The review the approver reads is built from a template, not written by a model.
- **Approvers** each have a bearer token, stored only as a hash. An approver cannot decide their own spend or anything over their approval limit (403). The decision and the move from `pending_review` to `decided` happen in one conditional update, so of two approvers racing, one wins and the other gets 409.
- **Execute** sends a signed decision event with the submission ID as its idempotency key. The API includes a mock ERP receiver that checks the signature and records each event once.

## Memory

See [ADR 0002](adr/0002-precedents-annotate-never-suppress.md).

When an approver dismisses a finding whose code can rest on a standing fact (a non-approved supplier, an unclear purpose, unjustified weekend spend), the dismissal becomes a precedent for that employee or supplier. Precedents are matched by subject and code, the three most recent, and stop applying when their clause's text changes. They annotate findings and inform the recommendation. They never remove, downgrade or reroute a finding, and a judgement check cleared by a precedent sends the submission to a person and shows the precedent.

## Runtime

See [ADR 0003](adr/0003-lambda-and-fifo-queue-runtime.md).

- Webhooks check a timestamped HMAC signature (`X-Signature`), record the submission idempotently on the sender's own identity, enqueue a `start` message and return 202. A retry with the same body returns the existing ID; a changed body gets 409.
- Every graph advance (`start` or `resume`) is a message on an SQS FIFO queue grouped by submission, consumed by a worker Lambda. A Postgres checkpointer lets a redelivered message continue from the last completed step, a status check turns stale or duplicate messages into no-ops, and a message that fails three times goes to a dead-letter queue with an email alarm.
- A sweeper runs every five minutes and re-enqueues submissions stuck longer than a threshold set above the queue's visibility timeout, which heals a crash between recording a decision and enqueueing it.

## Privacy and tracing

- Redaction runs before every prompt and again on traces before they leave. Untrusted submission fields reach the model inside labelled data tags. Logs are JSON lines carrying IDs only.
- Each graph advance is one Langfuse trace, grouped into a session by submission ID and flushed before the Lambda returns. Guardrail events (redactions, hygiene changes, verification drops) are recorded on the trace as metadata.
- **OpenTelemetry**: the Langfuse Python SDK pinned in `uv.lock` (5.2.0) is built on OpenTelemetry. It exports spans over OTLP and accepts a supplied tracer provider or span exporter, so the traces are OpenTelemetry-compatible. Masking happens at export: the SDK's `mask_otel_spans` hook redacts every string attribute, and a wrapping exporter redacts span status descriptions and exception events, which that hook cannot change. Tests check the exported spans, including those of a failed advance, carry no card numbers, emails or phone numbers.
