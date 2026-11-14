# Agent guide

Read `CLAUDE.md`, `GLOSSARY.md` and `docs/adr/` first. The specs are GitHub issues #1, #2 and #3; read the ticket and its parent spec before starting. `docs/draft-spec.md` is superseded.

## Stack

Python 3.12 with uv. LangGraph with an async Postgres checkpointer. FastAPI behind the Lambda Web Adapter. Postgres 16 with pgvector (compose locally, Supabase deployed). SQS FIFO (ElasticMQ locally, an in-memory fake in unit tests). One AWS SAM stack with EventBridge Scheduler and SSM parameters. Bedrock for Claude, GPT OSS, Nova and Titan embeddings; Google GenAI for Gemini in the eval only. Langfuse Cloud for traces.

## Layout

- `src/spendguard/settings.py`: the one `Settings` model. Every key in `.env.example` is a field, and a test keeps them in step. Add a field and its `.env.example` key together.
- `src/spendguard/branch_names.py`: branch to database and queue names.
- `src/spendguard/db/`: `ensure_branch_database`, the migrator and `migrations/`. Migration files are `YYYYMMDDHHMMSS_description.sql`; the migrator refuses any other name.
- `src/spendguard/structured_logging.py`: `log_event(event, **identifiers)`. Values must be IDs; free text raises.
- `src/spendguard/graph/nodes/`: one module per graph node, named as in spec #2.
- `src/spendguard/api/routes/`: one router per route group; `api/app.py` mounts them.
- `src/spendguard/store/`: one module per table.
- Other packages (`policy`, `seed`, `models`, `eval`, `rules`, `retrieval`, `privacy`, `security`, `observability`, `queue`) hold stubs with their final names. Fill the stub rather than creating a parallel module.

## Tests

- Build settings with `make_test_settings(**overrides)` from `tests/settings_factory.py`. Calling `load_settings` inside a test raises.
- Fixtures in `tests/conftest.py`: `database_connection` (branch database, migrated, rolled back after each test) and `branch_queue` (branch FIFO queue on ElasticMQ, purged before each test). Both raise when compose is down.
- Any skipped test fails the run.
- Never call a real model or AWS service.

## Commands

`make up`, `make migrate`, `make check` (ruff, `mypy --strict`, pytest). `make help` lists the rest.
