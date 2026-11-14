FROM public.ecr.aws/docker/library/python:3.12-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM public.ecr.aws/docker/library/python:3.12-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.1 /lambda-adapter /opt/extensions/lambda-adapter
COPY --from=build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" PORT=8000
WORKDIR /app

CMD ["uvicorn", "spendguard.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
