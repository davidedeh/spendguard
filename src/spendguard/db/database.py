import contextlib

import psycopg
from psycopg import sql

from spendguard.branch_names import database_name_for_branch
from spendguard.db.migrator import apply_migrations
from spendguard.settings import Settings

MAINTENANCE_DATABASE = 'postgres'
CONNECT_TIMEOUT_SECONDS = 3


class DatabaseUnreachableError(RuntimeError):
    pass


def _connection_info(settings: Settings, database_name: str) -> str:
    return psycopg.conninfo.make_conninfo(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password.get_secret_value(),
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


def ensure_branch_database(settings: Settings) -> str:
    if settings.database_url is None:
        _create_branch_database_if_missing(settings)
    connection_info = branch_connection_info(settings)
    with psycopg.connect(connection_info) as connection:
        apply_migrations(connection)
    return connection_info
