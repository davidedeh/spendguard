.DEFAULT_GOAL := help
.PHONY: help install up down migrate lint format typecheck test check build-wheel build-image validate-template run-api seed smoke-providers draft-cases freeze issue-token eval bootstrap secrets deploy destroy destroy-bootstrap

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
	uv run uvicorn spendguard.api.app:create_app --factory --reload

validate-template: ## Validate the SAM template with sam validate and cfn-lint
	$(call not_built_yet,Spec 2 SAM stack)

seed: ## Load the seed data into the branch database (SEED_DIRECTORY=path to override)
	uv run python -m spendguard.seed $(SEED_DIRECTORY)

smoke-providers: ## Check structured output, tool calling and embeddings against real providers (not in CI)
	uv run python -m spendguard.models.smoke

draft-cases: ## Draft eval cases with provider rotation (calls the real drafting models)
	uv run python -m spendguard.eval.drafting

freeze: ## Validate the eval dataset and write the lock file
	uv run python -m spendguard.eval.freeze

issue-token: ## Issue an approver bearer token and print it once
	$(call not_built_yet,Spec 2 approver auth)

eval: ## Run the eval against the frozen dataset
	$(call not_built_yet,Spec 2 eval runner)

bootstrap: ## Set up GitHub OIDC trust and the image repository
	$(call not_built_yet,Spec 2 OIDC pipeline deploy)

secrets: ## Write SecureString parameters
	$(call not_built_yet,Spec 2 SAM stack)

deploy: ## Build, push and deploy the SAM stack from a laptop
	$(call not_built_yet,Spec 2 SAM stack)

destroy: ## Delete the stack and empty the image repository
	$(call not_built_yet,Spec 2 SAM stack)

destroy-bootstrap: ## Tear down the bootstrap stack
	$(call not_built_yet,Spec 2 OIDC pipeline deploy)
