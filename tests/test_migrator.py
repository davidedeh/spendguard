from pathlib import Path

import psycopg
import pytest

from spendguard.db.migrator import InvalidMigrationNameError, apply_migrations


def test_pending_migrations_apply_in_timestamp_order_and_only_once(
    database_connection: psycopg.Connection, tmp_path: Path
) -> None:
    (tmp_path / '20260102000000_add_size.sql').write_text(
        'alter table widgets add column size integer'
    )
    (tmp_path / '20260101000000_create_widgets.sql').write_text('create table widgets (id int)')

    first_run = apply_migrations(database_connection, tmp_path)
    second_run = apply_migrations(database_connection, tmp_path)

    assert first_run == ['20260101000000_create_widgets', '20260102000000_add_size']
    assert second_run == []


def test_a_migration_without_a_timestamp_name_is_refused(
    database_connection: psycopg.Connection, tmp_path: Path
) -> None:
    (tmp_path / '0001_create_widgets.sql').write_text('create table widgets (id int)')

    with pytest.raises(InvalidMigrationNameError, match=r'0001_create_widgets\.sql'):
        apply_migrations(database_connection, tmp_path)
