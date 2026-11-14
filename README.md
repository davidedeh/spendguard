# SpendGuard

An agent that reviews expense claims and supplier invoices for the fictional Halden Group against a written policy, with a human approver in the loop. Everything here is fictional: no real companies, people, data or credentials.

Status: walking skeleton. Most modules are stubs that later tickets fill in.

## How it works

- **Rule checks** (limits, receipts, duplicates, split purchases, approved suppliers, purchase orders, budgets, weekend or holiday spend, currency rates) are pure Python and always run for their submission type.
- **Judgement checks** (unclear purpose, receipt mismatch, alcohol, and the justification half of weekend spend) are the only work a model does. Every model finding is verified by code before an approver sees it. See [ADR 0001](docs/adr/0001-model-confined-to-judgement-checks.md).
- **Precedents** are remembered approver dismissals. They annotate findings and never let a submission skip human review. See [ADR 0002](docs/adr/0002-precedents-annotate-never-suppress.md).
- **Runtime**: a LangGraph graph with a Postgres checkpointer. Every graph advance is a message on an SQS FIFO queue consumed by a Lambda worker; the API is a FastAPI Lambda behind a Function URL; an EventBridge schedule runs a sweeper for stuck submissions. See [ADR 0003](docs/adr/0003-lambda-and-fifo-queue-runtime.md).

Domain terms (submission, requester, finding, disposition, precedent) are defined in [GLOSSARY.md](GLOSSARY.md). Decisions live in [docs/adr/](docs/adr/). `docs/draft-spec.md` is superseded and is not a source.

## Stack

| Concern | Local and tests | Deployed |
| --- | --- | --- |
| Language and packaging | Python 3.12, uv, `src/` layout | Same, one container image |
| Database | Postgres 16 with pgvector in compose | Supabase through the transaction pooler |
| Queue | ElasticMQ in compose; an in-memory fake in unit tests | SQS FIFO with a dead-letter queue |
| Infrastructure | none | One AWS SAM stack: API and worker Lambdas, SQS, SNS alarm, EventBridge Scheduler, SSM parameters, ECR |
| Chat models | Fake models | Claude, GPT OSS 120B and Nova through Bedrock; Gemini through Google GenAI for the eval only |
| Embeddings | Fake embedder | Titan Text Embeddings V2 through Bedrock |
| Tracing | none | Langfuse Cloud, masked before sending |

Model IDs are configuration (`.env.example`), never code. GPT OSS 120B is OpenAI's open-weight model, not frontier GPT.

## Running locally

```sh
make install        # uv sync --frozen
make up             # shared compose stack: Postgres with pgvector, ElasticMQ
cp .env.example .env
make migrate        # creates this branch's database and applies migrations
make check          # ruff, mypy --strict, all tests
```

One compose stack serves every worktree. The database and queue names derive from the sanitised git branch name, so branches share services without sharing state.

`make help` lists every target. Targets for work not built yet exit with an error naming the ticket that builds them.

## Tests

- `make test` runs every test, including database and queue tests. Database and queue fixtures raise when compose is down; they never skip.
- A run with any skipped test fails.
- Tests build settings with `tests/settings_factory.py`. No test reads `.env` or needs credentials, and no test calls a real model or AWS.

CI runs the same checks on every pull request and on main against Postgres and ElasticMQ service containers, then builds the wheel and the Docker image.
