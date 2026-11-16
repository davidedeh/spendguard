from pathlib import Path

import psycopg
from psycopg import sql

from settings_factory import make_test_settings
from spendguard.branch_names import database_name_for_branch
from spendguard.db.database import branch_connection_info
from spendguard.deployment.preparation import prepare_database
from spendguard.models.factory import build_embedder
from spendguard.policy.loader import load_policy

FIXTURE_POLICY = load_policy(Path(__file__).parent / 'fixtures' / 'policy' / 'valid_policy.yaml')
PREPARATION_SETTINGS = make_test_settings(branch_name='deployment-preparation')


def drop_preparation_database() -> None:
    maintenance_connection_info = psycopg.conninfo.make_conninfo(
        branch_connection_info(PREPARATION_SETTINGS), dbname='postgres'
    )
    with psycopg.connect(maintenance_connection_info, autocommit=True) as maintenance_connection:
        maintenance_connection.execute(
            sql.SQL('drop database if exists {} with (force)').format(
                sql.Identifier(database_name_for_branch(PREPARATION_SETTINGS.branch_name))
            )
        )


async def test_preparing_an_empty_database_migrates_it_and_indexes_every_policy_clause() -> None:
    drop_preparation_database()

    await prepare_database(
        PREPARATION_SETTINGS, FIXTURE_POLICY, build_embedder(PREPARATION_SETTINGS)
    )

    with psycopg.connect(branch_connection_info(PREPARATION_SETTINGS)) as connection:
        indexed_references = {
            reference
            for (reference,) in connection.execute(
                'select distinct clause_reference from policy_chunks'
            )
        }
    assert indexed_references == {clause.reference for clause in FIXTURE_POLICY.clauses}
