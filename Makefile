.DEFAULT_GOAL := help
.PHONY: help install up down migrate lint format typecheck test check build-wheel build-image validate-template smoke run-api run-worker purge-queue reset-local index-policy demo record-demo render-demo seed smoke-providers draft-cases freeze issue-token eval eval-rerun bootstrap secrets deploy destroy destroy-bootstrap

not_built_yet = @echo "make $@ is not built yet ($(1))" >&2; exit 1

help: ## List targets
	@grep -E '^[a-z][a-z0-9-]*:.*## ' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "%-20s %s\n", $$1, $$2}'

install: ## Install dependencies from uv.lock
	uv sync --frozen

up: ## Start the shared compose stack (Postgres with pgvector, ElasticMQ)
	docker compose up -d --wait

down: ## Stop the shared compose stack
	docker compose down

migrate: ## Create the branch database if needed and apply migrations
	uv run python -m spendguard.db

lint: ## Run ruff
	uv run ruff check .
	uv run ruff format --check .

format: ## Format with ruff
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Run mypy --strict
	uv run mypy --strict

test: ## Run every test, including database and queue tests
	uv run pytest

check: lint typecheck test ## Run lint, typecheck and tests

build-wheel: ## Build the wheel
	uv build --wheel

build-image: ## Build the container image
	docker build -t spendguard:local .

run-api: ## Serve the API locally
	uv run uvicorn spendguard.api.app:create_production_app --factory --reload

run-worker: ## Advance submissions from the local queue (calls the configured models)
	uv run python -m spendguard.worker

purge-queue: ## Empty this branch's local queue and its dead-letter queue (ElasticMQ only)
	uv run python -m spendguard.queue.sqs

reset-local: ## Drop and rebuild this branch's local database, seed it and empty its local queue; then run make index-policy
	uv run python -m spendguard.db --drop-first
	$(MAKE) seed purge-queue

index-policy: ## Embed changed policy clauses into this branch's index (calls the configured embedding model)
	uv run python -m spendguard.deployment.preparation

demo: ## Post signed sample submissions to a running API and show a pending finding (APPROVER_TOKEN=approver token in the environment, URL=base URL, DECIDE=1 to also decide them)
	@uv run python -m spendguard.demo $(or $(URL),http://localhost:8000) $(if $(DECIDE),--decide)

record-demo: ## Record a demo take on a stack of its own and render the demo videos in recordings/ (calls the configured models and Langfuse)
	scripts/record-demo/record-demo.sh

render-demo: ## Re-render the demo videos in recordings/ from the last take, narrated too when recordings/narration has clips (no model calls)
	scripts/record-demo/render-demo.sh

validate-template: ## Validate the SAM template with sam validate and cfn-lint (offline)
	sam validate --lint --template template.yaml --region us-east-1
	uv run cfn-lint template.yaml

seed: ## Load the seed data into the branch database (SEED_DIRECTORY=path to override)
	uv run python -m spendguard.seed $(SEED_DIRECTORY)

smoke-providers: ## Check structured output, tool calling and embeddings against real providers (not in CI)
	uv run python -m spendguard.models.smoke

draft-cases: ## Draft eval cases with provider rotation (calls the real drafting models)
	uv run python -m spendguard.eval.drafting

freeze: ## Validate the eval dataset and write the lock file
	uv run python -m spendguard.eval.freeze

issue-token: ## Issue an approver bearer token and print it once (APPROVER=employee reference)
	uv run python -m spendguard.security.approver_tokens $(APPROVER)

eval: ## Run every eval row against the frozen dataset (calls the real models; CASES_PER_ROW=n for a pre-flight sampling every slice that leaves the README alone)
	uv run python -m spendguard.eval.runner $(if $(CASES_PER_ROW),--cases-per-row $(CASES_PER_ROW))

eval-rerun: ## Rerun the cases of a finished run that failed on the provider's side, then rewrite its results (calls the real models; RUN=<results file name without .md>)
	uv run python -m spendguard.eval.rerun --run $(RUN)

bootstrap: ## Set up the GitHub OIDC pipeline role for this repository's main branch
	scripts/bootstrap.sh

secrets: ## Write SecureString parameters
	scripts/secrets.sh

deploy: ## Build, push and deploy the SAM stack from a laptop (ALARM_EMAIL=address)
	scripts/deploy.sh

destroy: ## Delete the stack and empty the image repository
	scripts/destroy.sh

smoke: ## Check a running API: health is 200 and an unsigned webhook is 401 (URL=base URL)
	uv run python -m spendguard.deployment.smoke $(URL)

destroy-bootstrap: ## Tear down the bootstrap stack, warning first about the shared OIDC provider
	scripts/destroy-bootstrap.sh
