import contextlib

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import AsyncConnection, sql
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool

from spendguard.branch_names import database_name_for_branch
from spendguard.db.migrator import apply_migrations
from spendguard.settings import LocalPostgres, Settings

MAINTENANCE_DATABASE = 'postgres'
CONNECT_TIMEOUT_SECONDS = 3

type ConnectionPool = AsyncConnectionPool[AsyncConnection[DictRow]]


class DatabaseUnreachableError(RuntimeError):
    pass


class NotALocalDatabaseError(RuntimeError):
    def __init__(self) -> None:
        super().__init__('dropping only removes a local branch database: DATABASE_URL is set')


def _local_postgres(settings: Settings) -> LocalPostgres:
    local_postgres = settings.local_postgres()
    if local_postgres is None:
        raise ValueError('Every POSTGRES_ setting is needed when DATABASE_URL is not set')
    return local_postgres


def _connection_info(settings: Settings, database_name: str) -> str:
    local_postgres = _local_postgres(settings)
    return psycopg.conninfo.make_conninfo(
        host=local_postgres.host,
        port=local_postgres.port,
        user=local_postgres.user,
        password=local_postgres.password.get_secret_value(),
        dbname=database_name,
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
    )


def branch_connection_info(settings: Settings) -> str:
    if settings.database_url is not None:
        return settings.database_url.get_secret_value()
    return _connection_info(settings, database_name_for_branch(settings.branch_name))


def _create_branch_database_if_missing(settings: Settings) -> None:
    database_name = database_name_for_branch(settings.branch_name)
    try:
        maintenance_connection = psycopg.connect(
            _connection_info(settings, MAINTENANCE_DATABASE), autocommit=True
        )
    except psycopg.OperationalError as error:
        raise DatabaseUnreachableError(
            f'Postgres at {settings.postgres_host}:{settings.postgres_port} is unreachable;'
            ' start it with make up'
        ) from error
    with maintenance_connection:
        exists = maintenance_connection.execute(
            'select 1 from pg_database where datname = %s', (database_name,)
        ).fetchone()
        if exists is None:
            with contextlib.suppress(psycopg.errors.DuplicateDatabase):
                maintenance_connection.execute(
                    sql.SQL('create database {}').format(sql.Identifier(database_name))
                )


def drop_branch_database(settings: Settings) -> None:
    if settings.database_url is not None:
        raise NotALocalDatabaseError
    with psycopg.connect(
        _connection_info(settings, MAINTENANCE_DATABASE), autocommit=True
    ) as maintenance_connection:
        maintenance_connection.execute(
            sql.SQL('drop database if exists {} with (force)').format(
                sql.Identifier(database_name_for_branch(settings.branch_name))
            )
        )


def ensure_branch_database(settings: Settings) -> str:
    if settings.database_url is None:
        _create_branch_database_if_missing(settings)
    connection_info = branch_connection_info(settings)
    with psycopg.connect(connection_info, prepare_threshold=None) as connection:
        apply_migrations(connection)
    with psycopg.connect(
        connection_info, autocommit=True, prepare_threshold=None, row_factory=dict_row
    ) as connection:
        PostgresSaver(connection).setup()
    return connection_info


def async_connection_pool(connection_info: str) -> ConnectionPool:
    return AsyncConnectionPool(
        connection_info,
        connection_class=AsyncConnection[DictRow],
        kwargs={'autocommit': True, 'prepare_threshold': None, 'row_factory': dict_row},
        open=False,
    )
