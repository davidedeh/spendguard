from pathlib import Path

import psycopg
import pytest

from seeding import replace_seed_data
from spendguard.security.approver_tokens import NotAnApproverError, issue_token
from spendguard.seed.loader import read_seed_directory

SEED_DATA = read_seed_directory(Path(__file__).parent / 'fixtures' / 'seed')


@pytest.fixture
def seeded_connection(database_connection: psycopg.Connection) -> psycopg.Connection:
    replace_seed_data(database_connection, SEED_DATA)
    return database_connection


def stored_token_hash(connection: psycopg.Connection, employee: str) -> str | None:
    row = connection.execute(
        'select token_hash from approvers where employee = %s', (employee,)
    ).fetchone()
    assert row is not None
    hash_value: str | None = row[0]
    return hash_value


def test_an_issued_token_is_stored_only_as_a_hash(seeded_connection: psycopg.Connection) -> None:
    hash_before = stored_token_hash(seeded_connection, 'EMP-002')

    token = issue_token(seeded_connection, 'EMP-002')

    hash_after = stored_token_hash(seeded_connection, 'EMP-002')
    assert hash_after is not None
    assert hash_after != hash_before
    rows = seeded_connection.execute(
        'select employee, approval_limit_gbp::text, token_hash from approvers'
    ).fetchall()
    stored_values = [value for row in rows for value in row if value is not None]
    assert len(token) >= 32
    assert not any(token in value for value in stored_values)


def test_issuing_a_token_for_an_employee_who_is_not_an_approver_is_refused(
    seeded_connection: psycopg.Connection,
) -> None:
    with pytest.raises(NotAnApproverError, match='EMP-003'):
        issue_token(seeded_connection, 'EMP-003')
