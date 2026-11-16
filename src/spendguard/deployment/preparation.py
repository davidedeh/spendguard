import asyncio

from fastapi import FastAPI
from langchain_core.embeddings import Embeddings
from psycopg import AsyncConnection
from psycopg.rows import dict_row

from spendguard import settings
from spendguard.db.database import ensure_branch_database
from spendguard.models.factory import build_embedder
from spendguard.policy.loader import POLICY_PATH, Policy, load_policy
from spendguard.retrieval.policy_index import index_policy
from spendguard.settings import Settings
from spendguard.structured_logging import configure_logging, log_event


async def prepare_database(loaded_settings: Settings, policy: Policy, embedder: Embeddings) -> None:
    connection_info = await asyncio.to_thread(ensure_branch_database, loaded_settings)
    async with await AsyncConnection.connect(
        connection_info, autocommit=True, prepare_threshold=None, row_factory=dict_row
    ) as connection:
        await index_policy(connection, policy, embedder)


def create_production_preparation_app() -> FastAPI:
    loaded_settings = settings.load_settings()
    configure_logging(loaded_settings.log_level)
    app = FastAPI(title='SpendGuard database preparation')

    @app.post('/events')
    async def events() -> dict[str, str]:
        policy = load_policy(POLICY_PATH)
        await prepare_database(loaded_settings, policy, build_embedder(loaded_settings))
        log_event('database_prepared', policy_version=policy.version)
        return {'status': 'prepared'}

    return app
