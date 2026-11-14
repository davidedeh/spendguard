from pathlib import Path

import psycopg
import yaml
from psycopg import sql

from spendguard.seed.integrity import check_seed_integrity
from spendguard.seed.schema import SeedData
from spendguard.strict_model import StrictModel

SEED_DIRECTORY = Path(__file__).parent / 'data'

TABLE_KEYS = {
    'employees': ('reference',),
    'approvers': ('employee',),
    'cost_centres': ('reference',),
    'suppliers': ('reference',),
    'purchase_orders': ('reference',),
    'historical_claims': ('reference',),
    'historical_invoices': ('reference',),
    'fx_rates': ('currency', 'rate_date'),
    'public_holidays': ('holiday_on',),
}


class SeedIntegrityError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__('\n'.join(problems))
        self.problems = problems


def read_seed_directory(directory: Path) -> SeedData:
    if not directory.is_dir():
        raise FileNotFoundError(f'no seed directory at {directory}')
    files = {file.stem: yaml.safe_load(file.read_text()) for file in directory.glob('*.yaml')}
    return SeedData.model_validate(files)


def _upsert_statement(table: str, columns: list[str], key_columns: tuple[str, ...]) -> sql.Composed:
    updated_columns = [column for column in columns if column not in key_columns]
    return sql.SQL(
        'insert into {table} ({columns}) values ({placeholders})'
        ' on conflict ({key_columns}) do update set {assignments}'
    ).format(
        table=sql.Identifier(table),
        columns=sql.SQL(', ').join(sql.Identifier(column) for column in columns),
        placeholders=sql.SQL(', ').join(sql.Placeholder(column) for column in columns),
        key_columns=sql.SQL(', ').join(sql.Identifier(column) for column in key_columns),
        assignments=sql.SQL(', ').join(
            sql.SQL('{column} = excluded.{column}').format(column=sql.Identifier(column))
            for column in updated_columns
        ),
    )


def load_seed_data(connection: psycopg.Connection, seed_data: SeedData) -> None:
    problems = check_seed_integrity(seed_data)
    if problems:
        raise SeedIntegrityError(problems)
    with connection.transaction(), connection.cursor() as cursor:
        for table, key_columns in TABLE_KEYS.items():
            records: tuple[StrictModel, ...] = getattr(seed_data, table)
            if not records:
                continue
            columns = list(type(records[0]).model_fields)
            cursor.executemany(
                _upsert_statement(table, columns, key_columns),
                [record.model_dump() for record in records],
            )
        cursor.execute('set constraints all immediate')
