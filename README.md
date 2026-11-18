# SpendGuard

An agent that reviews expense claims and supplier invoices for the fictional Halden Group against a written policy, with a human approver in the loop. Everything here is fictional: no real companies, people, data or credentials.

Clean, low-value submissions go through without anyone touching them. Everything else reaches an approver with findings that code has checked, each tied to a policy clause, and the approver's decision is acted on exactly once.

[Watch the demo](docs/media/spendguard-demo.mp4) (about 100 seconds; [narrated version](docs/media/spendguard-demo-narrated.mp4), about 2 minutes): a clean claim approved automatically, a vague purpose flagged with its evidence, an injection attempt sent to a human, the approver's decisions, and the Langfuse trace behind the injection case. The terminal panes are the real output of a recorded run with real models.

Domain terms (submission, requester, finding, disposition, precedent) are defined in [GLOSSARY.md](GLOSSARY.md). Design decisions live in [docs/adr/](docs/adr/). The policy is [docs/halden-policy.md](docs/halden-policy.md), rendered from [src/spendguard/policy/policy.yaml](src/spendguard/policy/policy.yaml).

## How it works

Every submission runs through one LangGraph graph:

```text
intake ─┬─ run_rule_checks ─────────────────────────────────────┬─ verify ─ route ─┬─ execute ─ finalize
        └─ plan ─ retrieve_policy ─ run_judgement_checks ───────┘                  └─ human_review ─ execute ─ finalize
```

- **Intake** validates the submission, converts the amount to the policy's base currency (pounds) from the rates table, strips control and zero-width characters, caps field lengths, and redacts card numbers, emails and phone numbers before any prompt is built.
- **Rule checks and judgement checks run in parallel** and join at verification.

### Rules versus judgement

See [ADR 0001](docs/adr/0001-model-confined-to-judgement-checks.md).

- **Rule checks** (category limits, receipts, duplicates, split purchases, approved suppliers, purchase orders, budgets, the weekend or holiday gate, and claimed exchange rates) are pure Python over the submission, the seed data and the policy table. Every rule check that applies to the submission type runs every time. A model never chooses which run, never sets severity and never removes a rule finding.
- **Judgement checks** (unclear purpose, receipt mismatch, alcohol, and whether weekend or holiday spend is justified) are the only work a model does. A model plans two to four retrieval queries, policy clauses are retrieved once from pgvector, and a tool-calling agent with a capped number of tool calls reads the case. Its tools are bound to the submission, so it can look up this employee, supplier, cost centre and precedents, and nobody else's.
- **Verify** is pure Python. A model finding survives only if its code is a judgement code, its clause was retrieved, every quote appears verbatim in the named field of the redacted text the model saw, and any precedent it cites was one it was given. Duplicates collapse. Severity comes from the policy, not the model. Every drop is recorded in the audit trail with its reason.

The cost is that the agent can do less. In return a model cannot switch off a control, and a prompt injection can affect at most one submission's judgement findings.

### Routing, review and decisions

- **Route** is a pure function. A submission is auto-approved only when it has no findings, no check was cleared by a precedent, its amount is at or under the policy's auto-approve threshold and `FORCE_HUMAN_REVIEW` is off. Nothing is ever auto-rejected.
- **Recommendation** is a pure function, first match wins: a high-severity finding without a precedent gives reject; every finding having a precedent (or there being none) gives approve; otherwise there is no recommendation. The review the approver reads is built from a template, not written by a model.
- **Approvers** each have a bearer token, stored only as a hash. An approver cannot decide their own spend or anything over their approval limit (403). The decision and the move from `pending_review` to `decided` happen in one conditional update, so of two approvers racing, one wins and the other gets 409.
- **Execute** sends a signed decision event with the submission ID as its idempotency key. The API includes a mock ERP receiver that checks the signature and records each event once.

### Memory

See [ADR 0002](docs/adr/0002-precedents-annotate-never-suppress.md).

When an approver dismisses a finding whose code can rest on a standing fact (a non-approved supplier, an unclear purpose, unjustified weekend spend), the dismissal becomes a precedent for that employee or supplier. Precedents are matched by subject and code, the three most recent, and stop applying when their clause's text changes. They annotate findings and inform the recommendation. They never remove, downgrade or reroute a finding, and a judgement check cleared by a precedent sends the submission to a person and shows the precedent.

### Runtime

See [ADR 0003](docs/adr/0003-lambda-and-fifo-queue-runtime.md).

- Webhooks check a timestamped HMAC signature (`X-Signature`), record the submission idempotently on the sender's own identity, enqueue a `start` message and return 202. A retry with the same body returns the existing ID; a changed body gets 409.
- Every graph advance (`start` or `resume`) is a message on an SQS FIFO queue grouped by submission, consumed by a worker Lambda. A Postgres checkpointer lets a redelivered message continue from the last completed step, a status check turns stale or duplicate messages into no-ops, and a message that fails three times goes to a dead-letter queue with an email alarm.
- A sweeper runs every five minutes and re-enqueues submissions stuck longer than a threshold set above the queue's visibility timeout, which heals a crash between recording a decision and enqueueing it.

### Privacy and tracing

- Redaction runs before every prompt and again on traces before they leave. Untrusted submission fields reach the model inside labelled data tags. Logs are JSON lines carrying IDs only.
- Each graph advance is one Langfuse trace, grouped into a session by submission ID and flushed before the Lambda returns. Guardrail events (redactions, hygiene changes, verification drops) are recorded on the trace as metadata.
- **OpenTelemetry**: the Langfuse Python SDK pinned in `uv.lock` (5.2.0) is built on OpenTelemetry. It exports spans over OTLP and accepts a supplied tracer provider or span exporter, so the traces are OpenTelemetry-compatible. Masking happens at export: the SDK's `mask_otel_spans` hook redacts every string attribute, and a wrapping exporter redacts span status descriptions and exception events, which that hook cannot change. Tests check the exported spans, including those of a failed advance, carry no card numbers, emails or phone numbers.

## Stack

| Concern | Local and tests | Deployed |
| --- | --- | --- |
| Language and packaging | Python 3.12, uv, `src/` layout | Same, one container image |
| Database | Postgres 16 with pgvector in compose | Supabase through the transaction pooler |
| Queue | ElasticMQ in compose; an in-memory fake in unit tests | SQS FIFO with a dead-letter queue |
| Infrastructure | none | One AWS SAM stack: API, worker, sweeper and preparation Lambdas, SQS, SNS alarm, EventBridge Scheduler, SSM parameters, ECR |
| Chat models | Fake models | Claude, GPT OSS 120B and Nova through Bedrock; Gemini through Google GenAI for the eval only |
| Embeddings | Fake embedder | Titan Text Embeddings V2 through Bedrock |
| Tracing | none | Langfuse Cloud, masked before sending |

Model IDs are configuration, never code: each model's ID is a `CHAT_MODELS__<KEY>__MODEL_ID` setting in `.env.example`, and changing a model is a configuration change. GPT OSS 120B is OpenAI's open-weight model, not frontier GPT.

## Running locally

You need Docker, uv and make. Nothing in this section calls a model or AWS, apart from the demo with real models at its end.

```sh
make install        # uv sync --frozen
make up             # shared compose stack: Postgres with pgvector, ElasticMQ
cp .env.example .env
make migrate        # creates this branch's database and applies migrations
make check          # ruff, mypy --strict, all tests
```

The tests drive the whole flow (signed webhooks, the graph, the worker, approver decisions, the mock ERP and precedents) against the compose Postgres, the in-memory queue or ElasticMQ, and fake models.

To serve the API and call it by hand:

```sh
make seed                           # load the Halden seed data
make issue-token APPROVER=EMP-001   # prints a bearer token once
make run-api                        # serves on http://localhost:8000; leave it running
```

Then, from another terminal:

```sh
make smoke URL=http://localhost:8000    # health is 200, an unsigned webhook is 401
curl -H "Authorization: Bearer <token>" "http://localhost:8000/submissions?status=pending_review"
```

A submission posted to the local API is queued but not advanced until a worker consumes the local queue; deployed, the worker runs as a Lambda fed by SQS.

### Demo with real models

`make demo` posts three signed sample submissions taken from the frozen eval dataset (a clean small claim, a claim whose purpose is unclear, and a claim with injected instructions in its business purpose), waits for each to be processed, lists the pending reviews and prints one finding with its clause and quotes. The demo itself only talks to the local API. The worker and the policy index call the models configured in `.env`, so set their credentials there first, and leave `SSM_PARAMETER_PREFIX` empty: the demo refuses to start if it is set.

```sh
make seed                           # the samples refer to the Halden seed data
make index-policy                   # embeds changed policy clauses with the configured embedding model
make issue-token APPROVER=EMP-001   # prints a bearer token once
make run-api                        # terminal 1; needs WEBHOOK_SIGNING_KEY and DECISION_EVENT_SIGNING_KEY in .env
make run-worker                     # terminal 2; advances submissions with the configured models
make demo TOKEN=<token>             # terminal 3; URL=<base URL> if the API is not on http://localhost:8000
```

`make demo TOKEN=<token> DECIDE=1` goes on to decide the two claims waiting for review as that approver: it dismisses the unclear-purpose finding with a reason and approves that claim, and rejects the claim with injected instructions with a reason. It then shows each decision executed, the one record the mock ERP received for each, the precedent written from the dismissal, and the 409 a second decision on the same claim gets. It reads the mock ERP records and the precedent from this branch's local database.

`make record-demo` regenerates the video. It runs its own copy of the stack on a separate branch name and port (8020 by default), runs `python -m spendguard.demo_presentation` (the same flow as `make demo DECIDE=1`, printed for the recording) and keeps each line with the time it appeared, marks only that run's traces public in Langfuse, takes stills of the injection claim's trace with Playwright, reads each review's real duration from Langfuse, and renders the edit with [Remotion](https://www.remotion.dev) from `video/`. `make render-demo` re-renders from the last take without calling any model. When `recordings/narration/` holds the narration clips (`beat-1.m4a` to `beat-7.m4a`, with beat 4 as `beat-4a` and `beat-4b`), it also cleans them (silence trimmed, light noise reduction, loudness normalised to -16 LUFS), transcribes them locally with whisper.cpp to find each cue word, and renders a narrated cut, `docs/media/spendguard-demo-narrated.mp4`, whose captions and highlights follow the voice. That needs `brew install whisper-cpp` and its `ggml-base.en.bin` model in `~/.cache/whisper-cpp/`. It needs Node, `jq` and Playwright's Chromium (`uv run --with playwright==1.58.0 python -m playwright install chromium`), calls the configured models and sends traces to Langfuse. Logs go to the ignored `recordings/` directory and the take to `video/public/take/`.

The samples keep their eval case IDs as claim IDs, so running the demo again returns the same submissions rather than creating new ones. A message the worker fails on, or cannot read, stays queued and is retried, and after three receives it moves to the dead-letter queue. `make purge-queue` empties this branch's local queue and its dead-letter queue; it refuses to run unless `SQS_ENDPOINT_URL` points at ElasticMQ. `make reset-local` drops and rebuilds this branch's local database, seeds it and purges its queue, for a clean rerun; it refuses when `DATABASE_URL` is set, and afterwards the policy needs `make index-policy` and approvers need new tokens.

The policy index records which embedding model produced each clause's vectors, and `make index-policy` re-embeds any clause whose text or embedding model has changed.

One compose stack serves every worktree. The database and queue names derive from the sanitised git branch name, so branches share services without sharing state.

To tear down, stop `make run-api` with Ctrl-C, then:

```sh
make down                     # stop the compose stack; the database volume is kept
docker compose down --volumes # stop it and delete every branch's data
```

`make help` lists every target.

## Deploying

Every push to main deploys through `.github/workflows/deploy.yml`, which assumes an AWS role through GitHub's OIDC provider, so the repository holds no AWS keys. One-off setup, with admin credentials for the account, Docker and SAM CLI:

```sh
make bootstrap      # pipeline role trusted only by this repository's main branch
make secrets        # SecureString parameters the functions read at start-up
```

`make secrets` asks for the Supabase transaction pooler URL and the signing keys, generating the keys if you press Enter. `make bootstrap` prints the repository variables `deploy.yml` reads. `make deploy ALARM_EMAIL=you@example.com` runs the same script from a laptop: it builds and pushes the arm64 image tagged with the commit, deploys the stack, runs migrations and indexes the policy through the preparation function, then smoke-tests the API URL. Confirm the SNS subscription email to receive dead-letter alarms.

The deployed database starts with no approvers. Seed it and issue an approver token from a laptop, pointing the local commands at Supabase's session pooler URI (port 5432; `DATABASE_URL` in the environment overrides `.env`):

```sh
DATABASE_URL='<session pooler URI>' make seed
DATABASE_URL='<session pooler URI>' make issue-token APPROVER=EMP-001
```

To tear down, `make destroy` deletes the stack and its images, and `make destroy-bootstrap` deletes the pipeline resources, warning first if that would also delete the account's GitHub OIDC provider.

Images live in the stack's own repository, which keeps the five newest. The bootstrap is pointed at that repository rather than creating a second one.

### What the bootstrap grants

Read from the bootstrap template that SAM CLI 1.172.1 ships (`samcli/lib/pipeline/bootstrap/stage_resources.yaml`):

- **Pipeline execution role**, assumed by GitHub Actions. Trust: `sts:AssumeRoleWithWebIdentity` from `token.actions.githubusercontent.com` with audience `sts.amazonaws.com` and subject `repo:<owner>/<repo>:ref:refs/heads/main`, so pull requests, other branches and jobs that set a GitHub `environment` cannot assume it. Permissions: create, describe and execute CloudFormation change sets and delete stacks on any stack; pass the CloudFormation execution role; read and write the artifacts bucket; push and pull images in the stack's repository. The stack adds `lambda:InvokeFunction` on the preparation function only.
- **CloudFormation execution role**, assumed only by the CloudFormation service. Permissions: `Action: *` on `Resource: *`, full administrator. This is broad: anyone who can push to main can deploy a template that does anything in the account. It is accepted for a single-owner demo account; narrowing it means passing your own role to `sam pipeline bootstrap --cloudformation-execution-role` with the IAM, Lambda, SQS, SNS, CloudWatch, ECR, EventBridge Scheduler and SSM actions this template uses.
- **Artifacts bucket** and its logging bucket, both retained when the bootstrap stack is deleted.

## Evaluation

### Method

`make eval` runs every configured row against the frozen `eval-v1` dataset in [evals/cases/](evals/cases/) and refuses to report if the dataset no longer hashes to [evals/dataset.lock.json](evals/dataset.lock.json). Only the test split is reported; the dev split exists so prompt work never touches the test set. The eval calls real models, so it runs from a laptop. `.github/workflows/eval.yml` can also run it when dispatched by hand, but it needs a role that can invoke Bedrock models in the repository variable `AWS_EVAL_ROLE_ARN`, which this repository does not create, and it passes no Langfuse keys, so a CI run makes no Langfuse dataset runs. `make eval` refuses to start when `DATABASE_URL` is set, because it deletes every precedent in the database it runs against, or when any row's model has no price in `MODEL_PRICES`.

The rows are set by `EVAL_ROWS` in `.env.example`: Claude Haiku (the primary), Claude Sonnet, GPT OSS 120B (OpenAI's open-weight model), Nova Pro (AWS-native), a Gemini Pro-class model through Google's own API, and Claude Haiku given the whole policy instead of retrieved clauses, which measures what retrieval contributes. Each row runs every case once through the real graph, at temperature 0, with memory cleared at the start of the row. The first case of each memory pair runs before its follow-up, and the pair's scripted dispositions are applied as an approver's decision.

Per row:

- **Judgement precision and recall**, pooled over the four judgement codes, with 95% Wilson intervals and per-code counts. Injection cases are left out of the pool and scored only as injection survival.
- **Route and recommendation accuracy**: what the approver experiences.
- **Verify drop rate**: the share of model findings verification removed, so a model that paraphrases instead of quoting is not mistaken for one with poor judgement.
- **Injection survival**: the share of expected findings in injection cases still found despite the injected text.
- **Retrieval recall (top k per query)**: the share of gold clauses for expected judgement findings that retrieval returned, where each of the planner's queries returns its top k clauses. It does not apply to the whole-policy row.
- **p50 latency and cost per submission**: cost is token counts times the prices in `MODEL_PRICES`, each recorded with the date it was priced.

Rule-check precision and recall are reported once, not per row, and only as a consistency check: the rule labels were computed by the same rule functions the graph runs, so agreement shows the checks are wired into the graph correctly, not that they are accurate.

A memory on versus off comparison on the primary model reports the judgement false-positive rate on memory follow-ups and how often the recommendation matches the approver, scored against labels written for memory on, so a follow-up with any judgement finding counts as a false positive. A follow-up whose check got no verdict, or whose clearance verification dropped, is counted as forced review rather than as a silent clearance. It is in the dated results file, not the table below.

Each run writes a results file in `evals/results/`, named with the date and the first 8 characters of the run's identifier (`eval-v1-<date>-<run>.md`), with the exact parameters, prices and per-code counts, and replaces the table below, which names the exact model IDs measured. A results file is never overwritten, so a second run on the same day keeps the first. As each row finishes, its per-case results, token totals and run identifier are saved as JSON in a folder named like the results file, so a run that stops part-way keeps what it has paid for. A case that raises is recorded as failed and the row carries on: its expected judgement findings count as misses, its route and recommendation as wrong, and it adds no findings, latency or cost. All model clients are built before the first row, so a configuration error stops the run before any spend. `make eval CASES_PER_ROW=2` is a pre-flight: it runs the first two cases of every row and writes a results file marked as a pre-flight, leaving this README alone.

With both Langfuse keys set, `make eval` also uploads the reported cases to a Langfuse dataset named `eval-v1-<first 8 characters of the dataset hash>` and records each row as one dataset run, named with the date, the row's label and the run's identifier, with every case traced. A pre-flight's dataset runs also carry "pre-flight" in their names. The memory comparison is not uploaded.

### Case provenance

The cases were drafted by models, under constraints that keep labels honest:

- **Code-generated briefs.** For each case, code generated a brief from the slice plan ([evals/slice_plan.yaml](evals/slice_plan.yaml)), the rendered policy and the seed data: the slice, submission type, intended findings, subject, dates and which history rows the drafter could see. Drafters never saw the implementation.
- **Rotated across three model families on Bedrock.** Claude Sonnet, GPT OSS 120B and Nova Pro were each assigned about a third of every slice. When an assigned drafter could not produce a valid case, another drafted it, and each case's `drafted_by` records the provider and model that actually wrote it.
- **Labels computed by code where possible.** Drafters wrote the submission, its judgement findings and any scripted review. Rule findings, route and recommendation were computed by the rule, route and recommendation functions, and a draft whose rule findings differed from its brief, or that raised a judgement finding outside it, was rejected and redrafted.
- **Reviewed before the freeze.** The project owner reviewed every case's judgement labels and realism, then froze the dataset in the commit tagged `eval-v1`.

Possible own-family bias: the Claude, GPT OSS and Nova Pro rows are evaluated partly on cases their own family drafted, which may favour them. Gemini drafted no cases. `drafted_by` lets anyone split the results by drafter.

### Results

GPT OSS 120B is OpenAI's open-weight model, not frontier GPT.

Overlapping 95% intervals are never claimed as a win.

Production default: Haiku unless the eval shows it worse. With 16 positives the eval can only detect large gaps, so this is a cost-led default, not a demonstrated equivalence.

<!-- eval-results:start -->
No results yet: the eval has not been run against real models.
<!-- eval-results:end -->

## Tests

- `make test` runs every test, including database and queue tests. Database and queue fixtures raise when compose is down; they never skip.
- A run with any skipped test fails.
- Tests build settings with `tests/settings_factory.py`. No test reads `.env` or needs credentials, and no test calls a real model or AWS.

CI runs the same checks on every pull request and on main against Postgres and ElasticMQ service containers, validates the SAM template with `sam validate` and cfn-lint, and builds the wheel and the Docker image.

## Future improvements

- **Tiered auto-approve thresholds by category**, for example a higher one for travel and accommodation. Today one threshold covers every category.
- **A model reflection pass in verification**, deferred in ADR 0001 because it costs an extra call and is not deterministic; worth adding only if measured precision is poor.
