import psycopg
import pytest

from network import closed_port
from settings_factory import make_test_settings
from spendguard.db.database import (
    DatabaseUnreachableError,
    NotALocalDatabaseError,
    drop_branch_database,
    ensure_branch_database,
)


def test_the_branch_database_is_migrated_with_pgvector(
    database_connection: psycopg.Connection,
) -> None:
    row = database_connection.execute("select '[1,2,3]'::vector <-> '[1,2,5]'::vector").fetchone()

    assert row == (2.0,)


def test_an_unreachable_postgres_raises_instead_of_skipping() -> None:
    settings = make_test_settings(postgres_port=closed_port())

    with pytest.raises(DatabaseUnreachableError, match='make up'):
        ensure_branch_database(settings)


def test_dropping_a_branch_database_removes_everything_in_it() -> None:
    settings = make_test_settings(branch_name=f'{make_test_settings().branch_name}-drop')
    with psycopg.connect(ensure_branch_database(settings), autocommit=True) as connection:
        connection.execute('create table left_behind (id integer)')

    drop_branch_database(settings)

    with psycopg.connect(ensure_branch_database(settings)) as connection:
        assert connection.execute("select to_regclass('left_behind')").fetchone() == (None,)


def test_dropping_refuses_a_database_given_by_url() -> None:
    settings = make_test_settings(database_url='postgresql://someone@db.example/halden')

    with pytest.raises(NotALocalDatabaseError):
        drop_branch_database(settings)
