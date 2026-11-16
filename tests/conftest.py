import io
import logging
from collections.abc import AsyncIterator, Iterator

import psycopg
import pytest

from settings_factory import make_test_settings
from spendguard.db.database import ConnectionPool, async_connection_pool, ensure_branch_database
from spendguard.queue.sqs import ensure_branch_queue, sqs_client
from spendguard.settings import Settings
from spendguard.structured_logging import configure_logging


@pytest.fixture(scope='session')
def test_settings() -> Settings:
    return make_test_settings()


@pytest.fixture(scope='session')
def branch_database(test_settings: Settings) -> str:
    return ensure_branch_database(test_settings)


@pytest.fixture(scope='session')
def _branch_queue_url(test_settings: Settings) -> str:
    return ensure_branch_queue(test_settings)


@pytest.fixture
def branch_queue(test_settings: Settings, _branch_queue_url: str) -> str:
    sqs_client(test_settings).purge_queue(QueueUrl=_branch_queue_url)
    return _branch_queue_url


@pytest.fixture
def database_connection(branch_database: str) -> Iterator[psycopg.Connection]:
    with (
        psycopg.connect(branch_database) as connection,
        connection.transaction(force_rollback=True),
    ):
        yield connection


@pytest.fixture
def log_stream() -> Iterator[io.StringIO]:
    root = logging.getLogger()
    original_handlers, original_level = root.handlers[:], root.level
    stream = io.StringIO()
    configure_logging('INFO', stream)
    yield stream
    root.handlers, root.level = original_handlers, original_level


@pytest.fixture(scope='session')
async def connection_pool(branch_database: str) -> AsyncIterator[ConnectionPool]:
    pool = async_connection_pool(branch_database)
    await pool.open(wait=True)
    yield pool
    await pool.close()
