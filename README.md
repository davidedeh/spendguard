# SpendGuard

SpendGuard is an AI agent that reviews expense claims and supplier invoices for the fictional Halden Group against a written policy, with a human approver in the loop. Code runs every rule check and verifies every model finding, so clean, low-value claims are approved automatically and everything else reaches a person with evidence.

Built with Python, LangGraph with a Postgres checkpointer, Claude Haiku on AWS Bedrock, pgvector retrieval, Langfuse tracing, and SQS and Lambda deployed with AWS SAM.

https://github.com/user-attachments/assets/aaadc2ae-d7d6-4cf7-8318-11405eaf0040

A narrated walkthrough, about 2 minutes: a clean claim approved automatically, a vague purpose flagged with its evidence, an injection attempt sent to a human, the approver's decisions, and the Langfuse trace behind the injection case. The terminal panes are the real output of a recorded run with real models.

## Architecture

```text
intake ─┬─ run_rule_checks ─────────────────────────────────────┬─ verify ─ route ─┬─ execute ─ finalize
        └─ plan ─ retrieve_policy ─ run_judgement_checks ───────┘                  └─ human_review ─ execute ─ finalize
```

How a claim flows:

1. A signed webhook records the claim once and queues it.
2. Intake converts currency and redacts personal data before any prompt.
3. Code runs the rule checks; a model judges four policy questions.
4. Verify drops findings whose quote doesn't match, or whose clause or precedent wasn't supplied.
5. Clean claims of £100 or less auto-approve; the rest await an approver.

Detail: [how it works](docs/architecture.md), [configuration](docs/configuration.md), the [policy](docs/halden-policy.md) and the [glossary](GLOSSARY.md).

## Results

- **Recall 0.86, precision 0.62.** Claude Haiku found 18 of 21 expected findings, but 11 of its 29 were wrong; a wrong finding only sends the claim to a person, never rejects it.
- **Retrieval cuts cost, not errors.** It matched the whole policy in the prompt on recall and routing, at 29% less cost.
- **Precedent fit is unchecked.** With memory on, Haiku applied precedents to both control cases they did not fit.
- **GPT OSS 120B failed 21 of 49 cases**, every one by running to its output limit.
- **The dataset is blind-audited and versioned.** Audits that saw no labels or results produced eval-v2, frozen by hash.

Method and caveats: [docs/evaluation.md](docs/evaluation.md). Every dataset change: [evals/CHANGELOG.md](evals/CHANGELOG.md).

<!-- eval-results:start -->
Measured on 2026-11-18 against eval-v2 (968c6684); test cases: 49.

| Row | Model ID | Judgement precision (95% CI) | Judgement recall (95% CI) | Route accuracy | Recommendation accuracy | Verify drop rate | Injection survival | Retrieval recall (top 5 per query) | p50 latency | Cost per submission | Failed cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Claude Haiku (primary) | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 18/29 = 0.62 (0.44 to 0.77) | 18/21 = 0.86 (0.65 to 0.95) | 46/49 = 0.94 | 40/49 = 0.82 | 0/41 = 0.00 | 4/4 = 1.00 | 23/25 = 0.92 | 30.0 s | $0.0193 | 0 failed |
| Claude Sonnet | `us.anthropic.claude-sonnet-4-6` | 20/34 = 0.59 (0.42 to 0.74) | 20/21 = 0.95 (0.77 to 0.99) | 44/49 = 0.90 | 36/49 = 0.73 | 2/46 = 0.04 | 4/4 = 1.00 | 24/25 = 0.96 | 30.7 s | $0.0700 | 0 failed |
| GPT OSS 120B (OpenAI's open-weight model) | `openai.gpt-oss-120b-1:0` | 9/13 = 0.69 (0.42 to 0.87) | 9/21 = 0.43 (0.24 to 0.63) | 27/49 = 0.55 | 23/49 = 0.47 | 1/20 = 0.05 | 3/4 = 0.75 | 15/25 = 0.60 | 15.9 s | $0.0025 | 21 failed |
| Nova Pro (AWS-native) | `us.amazon.nova-pro-v1:0` | 12/21 = 0.57 (0.37 to 0.76) | 12/21 = 0.57 (0.37 to 0.76) | 42/49 = 0.86 | 36/49 = 0.73 | 2/31 = 0.06 | 2/4 = 0.50 | 25/25 = 1.00 | 14.2 s | $0.0121 | 0 failed |
| Claude Haiku, whole policy (no retrieval) | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 18/30 = 0.60 (0.42 to 0.75) | 18/21 = 0.86 (0.65 to 0.95) | 46/49 = 0.94 | 40/49 = 0.82 | 0/43 = 0.00 | 4/4 = 1.00 | not applicable | 29.8 s | $0.0271 | 0 failed |

Rule checks, the same for every row: precision 8/8 = 1.00, recall 8/8 = 1.00.
<!-- eval-results:end -->

GPT OSS 120B is OpenAI's open-weight model, not frontier GPT. Overlapping 95% intervals are never claimed as a win. Production default: Haiku unless the eval shows it worse. With 20 positives the eval can only detect large gaps, so this is a cost-led default, not a demonstrated equivalence.

## Key design decisions

- **Rules run in code.** Every applicable rule check runs; a model cannot skip, downgrade or remove one ([ADR 0001](docs/adr/0001-model-confined-to-judgement-checks.md)).
- **The model only judges.** Four judgement checks, with tools that see only the submission under review.
- **Verify is deterministic.** Plain Python, with every dropped finding recorded.
- **Fail closed.** A check with no usable verdict, or one verify drops, sends the claim to a person.
- **Durable human review.** Decisions execute exactly once, and precedents inform an approver but never decide ([ADR 0002](docs/adr/0002-precedents-annotate-never-suppress.md), [ADR 0003](docs/adr/0003-lambda-and-fifo-queue-runtime.md)).

## Run it locally

```sh
make install
make up
cp .env.example .env
make migrate
make check
```

The demo with real models, recording the video and teardown are in [docs/development.md](docs/development.md).

## How it's deployed

As of 2026-11-19, one AWS SAM stack in `us-east-1` runs SpendGuard, deployed since 2026-11-18. Every push to `main` deploys through GitHub Actions over OIDC, with no stored AWS keys, then migrates, indexes the policy and smoke-tests the API. Data is in Supabase Postgres; the runbook is [docs/deployment.md](docs/deployment.md).

## Limitations and next steps

- One run per row cannot show run-to-run variation; growing the dev split is worth exploring.
- A policy too large for the prompt is untested.
- One auto-approve threshold covers every category; tiered thresholds are worth exploring.
- The deploy's CloudFormation role has full administrator rights, accepted for a single-owner demo account.
- Approvers decide through the API, with no UI yet; the finance system is a mock.

The full list is in [docs/limitations-and-next-steps.md](docs/limitations-and-next-steps.md).
