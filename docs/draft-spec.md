> **Superseded.** This draft is kept for history only. Do not build from it. The design is defined by the published specs (GitHub issues labelled `ready-for-agent`), `GLOSSARY.md` and `docs/adr/`. Many decisions here have been reversed, among them the model selecting checks, model-set severity and confidence, `request_info`, ECS Fargate, background-task execution and the 8–12 page policy.

# SpendGuard — policy-aware approval agent

## What this is

A production-shaped portfolio project: a LangGraph agent that receives expense claims and supplier invoices via webhook, checks them against a written company policy and company data, flags violations with severity and policy citations, routes anything risky to a human approver, and records the decision through an API.

The point of the project is to demonstrate, end to end and in Python:

- multi-step agent orchestration with LangGraph (explicit graph, state, conditional routing)
- native human-in-the-loop (`interrupt` / `Command(resume=...)`) and durable state (Postgres checkpointer)
- RAG over policy documents (chunking, embeddings, pgvector)
- long-term memory of approver precedents across requests (LangGraph Store)
- tool use against real-shaped systems (APIs, webhooks)
- a measured eval harness comparing models (precision/recall on seeded violations)
- tracing, guardrails, testing, packaging, Docker, CI/CD, AWS deployment

Everything is fictional. The company is **Halden Group**, a mid-sized private-equity-backed services business. No real company names, data or credentials anywhere in the repo.

## Stack

| Concern | Choice |
| --- | --- |
| Language | Python 3.12, `uv` for env and packaging, `src/` layout |
| Agent | `langgraph`, `langchain-core`; models via `init_chat_model` |
| Models | Claude via AWS Bedrock (`langchain-aws`) as primary; one GPT and one Gemini model for eval comparison. Claude Haiku is the default during development. All model IDs come from env/config via `init_chat_model`, never hard-coded |
| API | FastAPI + Uvicorn, async throughout |
| Storage | Postgres 16 + pgvector: docker compose locally; Supabase (free tier) when deployed |
| Checkpointing | `langgraph-checkpoint-postgres` (`AsyncPostgresSaver`) |
| Observability | Langfuse (LangChain callback handler) + structured JSON logging |
| Quality | `pytest`, `pytest-asyncio`, `ruff`, `mypy --strict` on `src/` |
| Delivery | Dockerfile (multi-stage, slim), GitHub Actions, AWS ECS Fargate via ECR. Database is Supabase, not RDS and not a container on Fargate (no persistent storage there) |

## Domain

### Inputs

Two request types arrive on the webhook:

- **Expense claim**: employee, date, amount, currency, category, merchant, business purpose, receipt text (OCR'd text is provided as a string; no image processing).
- **Supplier invoice**: supplier, invoice number, line items, amount, currency, cost centre, PO number (optional), due date.

### Policy

`data/policy/` holds a fictional *Halden Group Expense & Procurement Policy* as Markdown, roughly 8–12 pages, with numbered clauses (e.g. `§3.2 Meals`). Write it in step 1. It must contain clear, testable rules: per-category limits, receipt thresholds, approval thresholds by amount, alcohol rules, duplicate submission, weekend/holiday spend, preferred suppliers, PO requirements above a threshold, split-purchase rules, foreign-currency handling.

### Company data (seeded in Postgres)

Employees (grade, cost centre, manager), cost centres with budgets and spend to date, suppliers (approved / not approved), historical claims and invoices (for duplicate detection), FX rates table.

### Finding codes

A closed set, used everywhere including evals:

`OVER_CATEGORY_LIMIT`, `MISSING_RECEIPT`, `DUPLICATE_SUBMISSION`, `SPLIT_PURCHASE`, `NON_APPROVED_SUPPLIER`, `MISSING_PO`, `BUDGET_EXCEEDED`, `ALCOHOL_NOT_PERMITTED`, `WEEKEND_SPEND_UNJUSTIFIED`, `PURPOSE_UNCLEAR`, `RECEIPT_MISMATCH`, `CURRENCY_ANOMALY`.

Each finding: `code`, `severity` (`low` | `medium` | `high`), `clause_ref` (e.g. `§3.2`), `evidence` (short string), `source` (`rule` | `llm`), `confidence` (0–1).

## Design principle: rules where possible, LLM where necessary

Numeric and lookup checks (limits, budgets, duplicates, PO thresholds, approved suppliers, FX) are **deterministic Python tools**. The LLM handles judgement only: is the business purpose plausible, does the receipt text match the claim, is this a split purchase in disguise, which policy clauses apply. Document this split in the README; it is a key talking point.

## Graph

State is a Pydantic model (or `TypedDict` with reducers) holding: `request`, `request_type`, `plan`, `policy_context` (retrieved chunks with clause refs), `findings` (append reducer), `verified_findings`, `route`, `decision`, `approver_note`, `precedents` (retrieved long-term memory), `audit` (append reducer), `messages` (`add_messages`).

Nodes (all `async`):

1. **`intake`**: validate against the request schema, normalise currency, apply input guardrails (strip control/zero-width characters, cap field lengths, neutralise instruction-like text in free-text fields). Reject malformed requests with a clear error.
2. **`plan`**: LLM with structured output decides which checks are relevant for this request (list of finding codes to test) and what policy topics to retrieve.
3. **`retrieve_policy`**: pgvector similarity search over policy chunks; keep clause refs for citations.
4. **`run_rule_checks`**: runs the deterministic tools for the planned codes; emits findings with `source="rule"`.
5. **`run_judgement_checks`**: a small tool-calling sub-agent (its own subgraph) with read-only tools (`get_employee`, `get_cost_centre`, `get_past_requests`, `get_supplier`, `convert_currency`); emits findings with `source="llm"`. Bounded to a max step count.
6. **`verify`**: reflection pass. Drops any LLM finding without a valid `clause_ref` and concrete `evidence`, de-duplicates, calibrates severity against the policy text. Output guardrail: findings must parse into the schema and codes must be from the closed set.
7. **`route`** (conditional edge): no findings and amount under the auto-approve threshold → `execute`; any `high` severity, or amount over the approval threshold → `human_review`; otherwise → `human_review` with a recommendation.
8. **`human_review`**: `interrupt()` with a payload of request summary, findings, recommendation. Resumed via `Command(resume={"decision": "approve" | "reject" | "request_info", "note": ...})`.
9. **`execute`**: write the decision to the mock ERP endpoint (internal FastAPI route simulating an external system) and fire an outbound notification webhook. Idempotent on request id.
10. **`finalize`**: write the audit record.

Checkpointer: `AsyncPostgresSaver`, `thread_id` = request id. A killed process must resume from the last completed node.

Expose a factory `build_graph(config: AgentConfig) -> CompiledGraph` so the graph is reusable as a template (different policy, data access and tools injected via config).

## Long-term memory

The checkpointer gives short-term, per-thread memory. Long-term, cross-thread memory uses LangGraph's Postgres-backed Store.

- Namespaces: `("suppliers", supplier_id)`, `("employees", employee_id)`.
- Written in `finalize`: when an approver's decision overrides a finding (e.g. approves a non-approved supplier with a reason), store that as a precedent.
- Read in `run_judgement_checks` and `verify`: relevant precedents are added to context, with a citation to the precedent, so repeat false positives drop.
- Eval: run with memory off and on over a sequence that includes repeat cases; report the false-positive delta.

## API

- `POST /webhooks/requests`: HMAC-SHA256 signature check (`X-Signature` header, secret from env); enqueue a graph run as a background task; return `202` with the request id.
- `GET /requests/{id}`: current state: status, findings, pending interrupt payload if any.
- `POST /requests/{id}/decision`: resume the interrupted run with the approver's decision.
- `POST /mock-erp/decisions`: the simulated downstream system.
- `GET /health`.

## Evals

- `evals/dataset.jsonl`: 30 synthetic requests (mix of claims and invoices), each labelled with the expected set of finding codes and expected route. At least 8 clean requests (no findings) to measure false positives. At least 2 cases per finding code.
- `evals/run.py`: runs the full graph (human review auto-resolved) per model configuration and computes per-code precision/recall/F1, micro-averaged precision/recall, route accuracy, mean latency, mean cost (token usage × configured prices).
- Model configurations: Claude (via Bedrock), one GPT model and one Gemini model; Claude Sonnet vs Haiku as an extra row. Same prompts, same temperature. Report cost per request alongside quality.
- Push runs to Langfuse as dataset runs; also write `evals/results/<date>.md` with a comparison table. The latest table goes in the README.
- `make eval` runs it. Not part of normal CI; a manual `workflow_dispatch` job runs it with secrets.

## Testing

- Unit tests for every deterministic tool and check (pure functions, no DB where avoidable).
- Graph tests with a fake chat model (`GenericFakeChatModel` or equivalent) covering: clean auto-approve path, high-severity path to interrupt, resume with approve and with reject, verify dropping an uncited finding.
- Durability test: run until interrupt, rebuild the graph from a fresh checkpointer connection, resume by thread id, assert completion.
- API tests with `httpx.AsyncClient`: bad signature → 401, valid webhook → 202, decision endpoint resumes the run.
- Tests that need Postgres use the docker compose service; mark them `@pytest.mark.db`.

## Repo layout

```
src/spendguard/
  config.py          # AgentConfig, settings from env
  graph/             # state.py, nodes/, routing.py, build.py
  tools/             # deterministic checks + read-only data tools
  rag/               # chunking, embedding, ingest, retrieve
  guardrails.py
  api/               # FastAPI app, routes, signature verification
  observability.py   # logging + Langfuse wiring
data/policy/         # policy markdown
data/seed/           # seed SQL / JSON
evals/
tests/
docker-compose.yml   # postgres+pgvector, app
Dockerfile
.github/workflows/ci.yml, deploy.yml, eval.yml
.env.example
Makefile             # setup, seed, ingest, test, eval, run
```

## Build order and done criteria

**Milestone 1: core graph working locally**
1. Scaffold the repo, tooling, docker compose, Makefile, CI running lint + types + tests.
2. Write the policy document and seed data.
3. RAG ingest and retrieval with clause refs.
4. Deterministic tools + unit tests.
5. Graph nodes 1–7 and 10 with the fake model in tests; then a real run via a CLI script against sample requests.
Done when: `make run-sample` processes 5 sample requests end to end with real models and prints findings with citations.

**Milestone 2: human-in-the-loop, API, delivery**
6. `human_review` interrupt + resume, Postgres checkpointer, durability test.
7. FastAPI routes, signature verification, mock ERP, outbound webhook.
8. Langfuse tracing and structured logs.
9. Dockerfile, deploy workflow to ECS Fargate.
Done when: a signed webhook to the deployed service produces a pending review, the decision endpoint resumes it, and the trace is visible in Langfuse.

**Milestone 3: evals and polish**
10. Eval dataset, harness, three-model comparison, results table.
11. README: architecture diagram (Mermaid), the rules-vs-LLM design note, how to run, eval results, trade-offs and what I'd do next.
Done when: README shows a results table with real numbers.

## How to work in this repo

- Work milestone by milestone; commit after each numbered step with a clear message.
- Write tests for deterministic logic before the logic.
- Keep nodes small and single-purpose; no business logic in FastAPI route handlers.
- Every LLM call uses structured output with a Pydantic schema; no free-text parsing.
- No secrets in code or commits; add every new env var to `.env.example`.
- When a design choice has a real trade-off, add one line to `docs/decisions.md`.
- Do not add frameworks or services not listed here without asking.

## Cost discipline

Keep cloud and model costs minimal. Avoid resources that keep billing unattended; anything that does must be cheap and quick to tear down.

- Use the fake chat model in all automated tests; never call a real model in CI.
- Default to Claude Haiku for development runs; use Sonnet only in evals.
- Cache embeddings; only re-embed the policy when it changes.
- Run the full eval only when numbers are needed (expect 3–4 runs total); support `--limit N` for quick partial runs.
- No RDS, NAT gateways, load balancers or other always-on paid AWS resources unless asked. Document how to scale the ECS service to zero in the README.
- Flag any step that would create an AWS resource with ongoing cost before doing it.

## Before starting (manual, done by me)

- AWS account with a $10 budget alert; Bedrock model access enabled for the chosen models and region.
- Langfuse Cloud account (free tier) and keys.
- Supabase project (free tier) with the pgvector extension enabled.
- OpenAI and Gemini API keys.
- Skills from `langchain-ai/langchain-skills` installed locally: `langgraph-fundamentals`, `langgraph-persistence`, `langgraph-human-in-the-loop`, `langchain-fundamentals`, `langchain-rag`, `langchain-dependencies`. Prefer these and the live docs over memory for any LangChain/LangGraph API.

## Out of scope

Front-end UI, auth beyond webhook signatures, image/OCR processing, multi-tenancy, real ERP integrations.