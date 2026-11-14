import psycopg
import pytest

from network import closed_port
from settings_factory import make_test_settings
from spendguard.db.database import DatabaseUnreachableError, ensure_branch_database


def test_the_branch_database_is_migrated_with_pgvector(
    database_connection: psycopg.Connection,
) -> None:
    row = database_connection.execute("select '[1,2,3]'::vector <-> '[1,2,5]'::vector").fetchone()

    assert row == (2.0,)


def test_an_unreachable_postgres_raises_instead_of_skipping() -> None:
    settings = make_test_settings(postgres_port=closed_port())

    with pytest.raises(DatabaseUnreachableError, match='make up'):
        ensure_branch_database(settings)
