import psycopg
from psycopg import sql

from spendguard.seed.loader import TABLE_KEYS, load_seed_data
from spendguard.seed.schema import SeedData


def replace_seed_data(connection: psycopg.Connection, seed_data: SeedData) -> None:
    with connection.transaction():
        for table in TABLE_KEYS:
            connection.execute(sql.SQL('delete from {}').format(sql.Identifier(table)))
        load_seed_data(connection, seed_data)
