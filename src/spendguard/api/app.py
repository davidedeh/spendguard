from fastapi import FastAPI

from spendguard import settings
from spendguard.api.dependencies import ApiDependencies
from spendguard.api.routes import health, mock_erp, submissions, webhooks
from spendguard.db.database import (
    Lifespan,
    async_connection_pool,
    branch_connection_info,
    pool_lifespan,
)
from spendguard.policy.loader import POLICY_PATH, load_policy
from spendguard.queue.sqs import SqsQueue, ensure_branch_queue, sqs_client
from spendguard.seed.loader import SEED_DIRECTORY, read_seed_directory
from spendguard.structured_logging import configure_logging


def create_app(dependencies: ApiDependencies, lifespan: Lifespan | None = None) -> FastAPI:
    app = FastAPI(title='SpendGuard', lifespan=lifespan)
    app.state.dependencies = dependencies
    for module in (health, webhooks, submissions, mock_erp):
        app.include_router(module.router)
    return app


def create_production_app() -> FastAPI:
    loaded_settings = settings.load_settings()
    configure_logging(loaded_settings.log_level)
    connection_pool = async_connection_pool(branch_connection_info(loaded_settings))
    queue = SqsQueue(
        client=sqs_client(loaded_settings), queue_url=ensure_branch_queue(loaded_settings)
    )

    return create_app(
        ApiDependencies(
            settings=loaded_settings,
            policy=load_policy(POLICY_PATH),
            seed_data=read_seed_directory(SEED_DIRECTORY),
            connection_pool=connection_pool,
            queue=queue,
        ),
        lifespan=pool_lifespan(connection_pool),
    )
