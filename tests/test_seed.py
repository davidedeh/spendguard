import datetime
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg
import pydantic
import pytest
import yaml
from psycopg import sql

from spendguard.seed.integrity import check_seed_integrity
from spendguard.seed.loader import SeedIntegrityError, load_seed_data, read_seed_directory
from spendguard.seed.schema import SeedData

FIXTURE_SEED_DIRECTORY = Path(__file__).parent / 'fixtures' / 'seed'


def fixture_copy(directory: Path, **replaced_files: list[dict[str, Any]]) -> Path:
    shutil.copytree(FIXTURE_SEED_DIRECTORY, directory)
    for stem, records in replaced_files.items():
        (directory / f'{stem}.yaml').write_text(yaml.safe_dump(records))
    return directory


def test_the_fixture_seed_files_read_into_typed_records() -> None:
    seed_data = read_seed_directory(FIXTURE_SEED_DIRECTORY)

    claim = seed_data.historical_claims[1]
    assert (claim.reference, claim.claimant, claim.currency) == ('CLM-0002', 'EMP-002', 'EUR')
    assert claim.amount == Decimal('120.00')
    assert claim.spent_on == datetime.date(2026, 10, 3)
    assert seed_data.cost_centres[2].owner is None
    assert seed_data.fx_rates[0].gbp_per_unit == Decimal('0.86')


def test_a_missing_seed_directory_is_named_in_the_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='no-seed-here'):
        read_seed_directory(tmp_path / 'no-seed-here')


@pytest.mark.parametrize(
    ('employee', 'rejected_field'),
    [
        (
            {
                'reference': 'EMP-001',
                'name': 'Avery Quinn',
                'grade': 'G5',
                'cost_centre': 'CC-100',
                'manager': None,
            },
            'employees.0.name',
        ),
        (
            {'reference': 'Avery Quinn', 'grade': 'G5', 'cost_centre': 'CC-100', 'manager': None},
            'employees.0.reference',
        ),
        (
            {'reference': 'EMP-001', 'grade': 'G5', 'cost_centre': 'CC-100', 'manager': 'A Quinn'},
            'employees.0.manager',
        ),
    ],
)
def test_a_seed_record_cannot_hold_a_persons_name(
    tmp_path: Path, employee: dict[str, Any], rejected_field: str
) -> None:
    directory = fixture_copy(tmp_path / 'seed', employees=[employee])

    with pytest.raises(pydantic.ValidationError, match=rejected_field):
        read_seed_directory(directory)


def test_an_invoice_against_a_po_takes_its_cost_centre_from_the_po(tmp_path: Path) -> None:
    invoices = yaml.safe_load((FIXTURE_SEED_DIRECTORY / 'historical_invoices.yaml').read_text())
    invoices[0]['cost_centre'] = 'CC-200'
    directory = fixture_copy(tmp_path / 'seed', historical_invoices=invoices)

    with pytest.raises(pydantic.ValidationError, match=r'historical_invoices\.0\n'):
        read_seed_directory(directory)


@pytest.mark.parametrize(
    ('stem', 'field', 'value'),
    [
        ('purchase_orders', 'amount', 0),
        ('historical_claims', 'amount', -42.5),
        ('historical_invoices', 'amount', 0),
        ('approvers', 'approval_limit_gbp', 0),
        ('cost_centres', 'budget_gbp', 0),
        ('cost_centres', 'spend_to_date_gbp', -0.01),
        ('fx_rates', 'gbp_per_unit', 0),
    ],
)
def test_an_amount_the_product_could_not_record_is_rejected(
    tmp_path: Path, stem: str, field: str, value: float
) -> None:
    records = yaml.safe_load((FIXTURE_SEED_DIRECTORY / f'{stem}.yaml').read_text())
    records[0][field] = value
    directory = fixture_copy(tmp_path / 'seed', **{stem: records})

    with pytest.raises(pydantic.ValidationError, match=rf'{stem}\.0\.{field}'):
        read_seed_directory(directory)


def with_field_replaced(
    seed_data: SeedData, collection: str, index: int, field: str, value: object
) -> SeedData:
    records = list(getattr(seed_data, collection))
    records[index] = records[index].model_copy(update={field: value})
    return seed_data.model_copy(update={collection: tuple(records)})


@pytest.mark.parametrize(
    ('purchase_order', 'cost_centre', 'requester'),
    [
        ('PO-0001', None, 'EMP-003'),
        (None, 'CC-200', 'EMP-002'),
        (None, 'CC-300', None),
        (None, None, None),
    ],
)
def test_an_invoice_requester_is_the_po_raiser_else_the_cost_centre_owner(
    purchase_order: str | None, cost_centre: str | None, requester: str | None
) -> None:
    seed_data = read_seed_directory(FIXTURE_SEED_DIRECTORY)

    assert seed_data.invoice_requester(purchase_order, cost_centre) == requester


def test_the_fixture_seed_data_has_no_integrity_problems() -> None:
    assert check_seed_integrity(read_seed_directory(FIXTURE_SEED_DIRECTORY)) == []


@pytest.mark.parametrize(
    ('collection', 'index', 'field', 'value', 'problem'),
    [
        (
            'purchase_orders',
            0,
            'raised_by',
            'EMP-999',
            'purchase order PO-0001: raised_by EMP-999 is not a known employee',
        ),
        (
            'purchase_orders',
            0,
            'supplier',
            'SUP-999',
            'purchase order PO-0001: supplier SUP-999 is not a known supplier',
        ),
        (
            'purchase_orders',
            0,
            'cost_centre',
            'CC-999',
            'purchase order PO-0001: cost_centre CC-999 is not a known cost centre',
        ),
        (
            'employees',
            1,
            'cost_centre',
            'CC-999',
            'employee EMP-002: cost_centre CC-999 is not a known cost centre',
        ),
        (
            'employees',
            2,
            'manager',
            'EMP-999',
            'employee EMP-003: manager EMP-999 is not a known employee',
        ),
        (
            'approvers',
            1,
            'employee',
            'EMP-999',
            'approver EMP-999: employee EMP-999 is not a known employee',
        ),
        (
            'cost_centres',
            1,
            'owner',
            'EMP-999',
            'cost centre CC-200: owner EMP-999 is not a known employee',
        ),
        (
            'historical_claims',
            1,
            'claimant',
            'EMP-999',
            'historical claim CLM-0002: claimant EMP-999 is not a known employee',
        ),
        (
            'historical_invoices',
            1,
            'supplier',
            'SUP-999',
            'historical invoice INV-0002: supplier SUP-999 is not a known supplier',
        ),
        (
            'historical_invoices',
            0,
            'purchase_order',
            'PO-9999',
            'historical invoice INV-0001: purchase_order PO-9999 is not a known purchase order',
        ),
        (
            'historical_invoices',
            1,
            'cost_centre',
            'CC-999',
            'historical invoice INV-0002: cost_centre CC-999 is not a known cost centre',
        ),
    ],
)
def test_a_dangling_reference_is_reported(
    collection: str, index: int, field: str, value: str, problem: str
) -> None:
    seed_data = read_seed_directory(FIXTURE_SEED_DIRECTORY)

    broken = with_field_replaced(seed_data, collection, index, field, value)

    assert check_seed_integrity(broken) == [problem]


@pytest.mark.parametrize(
    ('collection', 'index', 'problem'),
    [
        ('employees', 1, 'employee EMP-002 appears more than once'),
        ('approvers', 0, 'approver EMP-001 appears more than once'),
        ('suppliers', 1, 'supplier SUP-002 appears more than once'),
        ('historical_invoices', 0, 'historical invoice INV-0001 appears more than once'),
        ('fx_rates', 1, 'fx rate USD 2026-10-01 appears more than once'),
        ('public_holidays', 0, 'public holiday 2026-12-25 appears more than once'),
    ],
)
def test_a_record_that_appears_twice_is_reported(collection: str, index: int, problem: str) -> None:
    seed_data = read_seed_directory(FIXTURE_SEED_DIRECTORY)
    records = getattr(seed_data, collection)

    broken = seed_data.model_copy(update={collection: (*records, records[index])})

    assert check_seed_integrity(broken) == [problem]


SEED_TABLES = (
    'employees',
    'approvers',
    'cost_centres',
    'suppliers',
    'purchase_orders',
    'historical_claims',
    'historical_invoices',
    'fx_rates',
    'public_holidays',
)


def row_counts(connection: psycopg.Connection) -> dict[str, int]:
    counts = {}
    for table in SEED_TABLES:
        row = connection.execute(
            sql.SQL('select count(*) from {}').format(sql.Identifier(table))
        ).fetchone()
        assert row is not None
        counts[table] = row[0]
    return counts


@pytest.fixture
def empty_seed_tables(database_connection: psycopg.Connection) -> psycopg.Connection:
    for table in SEED_TABLES:
        database_connection.execute(sql.SQL('delete from {}').format(sql.Identifier(table)))
    return database_connection


def test_loading_the_fixture_seed_data_twice_leaves_one_copy(
    empty_seed_tables: psycopg.Connection,
) -> None:
    database_connection = empty_seed_tables
    seed_data = read_seed_directory(FIXTURE_SEED_DIRECTORY)

    load_seed_data(database_connection, seed_data)
    load_seed_data(database_connection, seed_data)

    assert row_counts(database_connection) == {
        'employees': 3,
        'approvers': 2,
        'cost_centres': 3,
        'suppliers': 2,
        'purchase_orders': 1,
        'historical_claims': 2,
        'historical_invoices': 2,
        'fx_rates': 2,
        'public_holidays': 2,
    }


def test_reloading_after_a_seed_file_changes_updates_the_record(
    empty_seed_tables: psycopg.Connection,
) -> None:
    database_connection = empty_seed_tables
    seed_data = read_seed_directory(FIXTURE_SEED_DIRECTORY)
    load_seed_data(database_connection, seed_data)

    changed = with_field_replaced(seed_data, 'approvers', 1, 'approval_limit_gbp', Decimal('7500'))
    load_seed_data(database_connection, changed)

    rows = database_connection.execute(
        'select employee, approval_limit_gbp from approvers order by employee'
    ).fetchall()
    assert rows == [('EMP-001', Decimal('25000.00')), ('EMP-002', Decimal('7500.00'))]


def test_seed_data_with_integrity_problems_is_refused_and_nothing_is_loaded(
    database_connection: psycopg.Connection,
) -> None:
    seed_data = read_seed_directory(FIXTURE_SEED_DIRECTORY)
    broken = with_field_replaced(seed_data, 'purchase_orders', 0, 'raised_by', 'EMP-999')
    broken = with_field_replaced(broken, 'historical_claims', 0, 'claimant', 'EMP-998')
    counts_before = row_counts(database_connection)

    with pytest.raises(SeedIntegrityError) as raised:
        load_seed_data(database_connection, broken)

    assert raised.value.problems == [
        'purchase order PO-0001: raised_by EMP-999 is not a known employee',
        'historical claim CLM-0001: claimant EMP-998 is not a known employee',
    ]
    assert row_counts(database_connection) == counts_before


@pytest.mark.parametrize('cost_centre', ['CC-300', None])
def test_an_invoice_without_a_po_or_a_cost_centre_owner_has_its_missing_requester_reported(
    cost_centre: str | None,
) -> None:
    seed_data = read_seed_directory(FIXTURE_SEED_DIRECTORY)

    broken = with_field_replaced(seed_data, 'historical_invoices', 1, 'cost_centre', cost_centre)

    assert check_seed_integrity(broken) == [
        'historical invoice INV-0002: no requester, since it has no purchase order'
        ' and no cost centre owner'
    ]
