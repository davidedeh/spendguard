import hashlib
import secrets
import sys

import psycopg

from spendguard import settings
from spendguard.db.database import ensure_branch_database

TOKEN_BYTES = 32


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class NotAnApproverError(LookupError):
    def __init__(self, employee: str) -> None:
        super().__init__(f'{employee} is not in the approvers table')


def issue_token(connection: psycopg.Connection, employee: str) -> str:
    token = secrets.token_urlsafe(TOKEN_BYTES)
    updated = connection.execute(
        'update approvers set token_hash = %s where employee = %s returning employee',
        (token_hash(token), employee),
    ).fetchone()
    if updated is None:
        raise NotAnApproverError(employee)
    return token


def main(arguments: list[str]) -> None:
    if len(arguments) != 1:
        raise SystemExit('usage: make issue-token APPROVER=<employee reference>')
    loaded_settings = settings.load_settings()
    with psycopg.connect(ensure_branch_database(loaded_settings)) as connection:
        try:
            token = issue_token(connection, arguments[0])
        except NotAnApproverError as error:
            raise SystemExit(str(error)) from error
    print(token)


if __name__ == '__main__':
    main(sys.argv[1:])
