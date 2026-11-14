import re
from pathlib import Path

import psycopg

MIGRATIONS_DIRECTORY = Path(__file__).parent / 'migrations'
MIGRATION_LOCK_KEY = 4_004_004
migration_file_pattern = re.compile(r'\d{14}_[a-z0-9_]+\.sql')


class InvalidMigrationNameError(ValueError):
    pass


def _migration_files(migrations_directory: Path) -> list[Path]:
    files = sorted(migrations_directory.glob('*.sql'))
    for file in files:
        if not migration_file_pattern.fullmatch(file.name):
            raise InvalidMigrationNameError(file.name)
    return files


def apply_migrations(
    connection: psycopg.Connection,
    migrations_directory: Path = MIGRATIONS_DIRECTORY,
) -> list[str]:
    files = _migration_files(migrations_directory)
    applied_now: list[str] = []
    with connection.transaction():
        connection.execute('select pg_advisory_xact_lock(%s)', (MIGRATION_LOCK_KEY,))
        connection.execute(
            'create table if not exists schema_migrations ('
            ' name text primary key,'
            ' applied_at timestamptz not null default now())'
        )
        already_applied = {
            name for (name,) in connection.execute('select name from schema_migrations')
        }
        for file in files:
            name = file.stem
            if name in already_applied:
                continue
            connection.execute(file.read_text())
            connection.execute('insert into schema_migrations (name) values (%s)', (name,))
            applied_now.append(name)
    return applied_now
