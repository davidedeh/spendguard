import dataclasses
from typing import Any

import pytest

from spendguard.policy.loader import POLICY_PATH, load_policy
from spendguard.seed.agreement import check_policy_agreement
from spendguard.seed.loader import SEED_DIRECTORY, read_seed_directory
from spendguard.seed.schema import SeedData

HALDEN_POLICY = load_policy(POLICY_PATH)


def agreeing_seed_records() -> dict[str, list[dict[str, Any]]]:
    return {
        'employees': [
            {
                'reference': 'EMP-001',
                'grade': 'executive',
                'cost_centre': 'CC-100',
                'manager': None,
            },
            {
                'reference': 'EMP-002',
                'grade': 'executive',
                'cost_centre': 'CC-100',
                'manager': 'EMP-001',
            },
            {
                'reference': 'EMP-003',
                'grade': 'manager',
                'cost_centre': 'CC-200',
                'manager': 'EMP-001',
            },
            {
                'reference': 'EMP-004',
                'grade': 'associate',
                'cost_centre': 'CC-200',
                'manager': 'EMP-003',
            },
        ],
        'approvers': [
            {'employee': 'EMP-001', 'approval_limit_gbp': '25000.00'},
            {'employee': 'EMP-002', 'approval_limit_gbp': '25000.00'},
            {'employee': 'EMP-003', 'approval_limit_gbp': '2500.00'},
        ],
        'cost_centres': [
            {
                'reference': 'CC-100',
                'owner': 'EMP-001',
                'budget_gbp': '50000.00',
                'spend_to_date_gbp': '1000.00',
            },
            {
                'reference': 'CC-200',
                'owner': 'EMP-003',
                'budget_gbp': '9000.00',
                'spend_to_date_gbp': '1500.00',
            },
        ],
        'suppliers': [{'reference': 'SUP-001', 'approved': True}],
        'purchase_orders': [
            {
                'reference': 'PO-0001',
                'raised_by': 'EMP-004',
                'supplier': 'SUP-001',
                'cost_centre': 'CC-200',
                'amount': '1200.00',
                'currency': 'GBP',
            }
        ],
        'historical_claims': [
            {
                'reference': 'CLM-0001',
                'claimant': 'EMP-004',
                'spent_on': '2026-10-05',
                'category': 'client_entertainment',
                'quantity': 3,
                'merchant': 'Canal Side Grill',
                'amount': '141.00',
                'currency': 'GBP',
            },
            {
                'reference': 'CLM-0002',
                'claimant': 'EMP-002',
                'spent_on': '2026-10-06',
                'category': 'accommodation',
                'quantity': 2,
                'merchant': 'Hotel du Parc',
                'amount': '300.00',
                'currency': 'EUR',
            },
        ],
        'historical_invoices': [
            {
                'reference': 'INV-0001',
                'supplier': 'SUP-001',
                'supplier_invoice_number': 'A-1',
                'purchase_order': 'PO-0001',
                'cost_centre': None,
                'issued_on': '2026-10-07',
                'amount': '1200.00',
                'currency': 'GBP',
            },
            {
                'reference': 'INV-0002',
                'supplier': 'SUP-001',
                'supplier_invoice_number': 'A-2',
                'purchase_order': None,
                'cost_centre': 'CC-100',
                'issued_on': '2026-10-08',
                'amount': '200.00',
                'currency': 'USD',
            },
        ],
        'fx_rates': [
            {'currency': 'EUR', 'rate_date': '2026-10-06', 'gbp_per_unit': '0.86'},
            {'currency': 'USD', 'rate_date': '2026-10-08', 'gbp_per_unit': '0.79'},
        ],
        'public_holidays': [{'holiday_on': '2026-12-25', 'description': 'Christmas Day'}],
    }


def test_the_halden_seed_data_agrees_with_the_halden_policy() -> None:
    assert check_policy_agreement(read_seed_directory(SEED_DIRECTORY), HALDEN_POLICY) == []


def test_the_halden_seed_data_has_the_variety_the_checks_need() -> None:
    seed_data = read_seed_directory(SEED_DIRECTORY)

    assert {supplier.approved for supplier in seed_data.suppliers} == {True, False}
    assert {invoice.purchase_order is None for invoice in seed_data.historical_invoices} == {
        True,
        False,
    }
    assert len({approver.approval_limit_gbp for approver in seed_data.approvers}) == 2
    assert any(centre.spend_to_date_gbp > centre.budget_gbp for centre in seed_data.cost_centres)
    multi_unit_categories = {
        claim.category for claim in seed_data.historical_claims if claim.quantity > 1
    }
    assert {'accommodation', 'client_entertainment', 'subsistence'} <= multi_unit_categories


def test_seed_data_that_agrees_with_the_policy_has_no_problems() -> None:
    seed_data = SeedData.model_validate(agreeing_seed_records())

    assert check_policy_agreement(seed_data, HALDEN_POLICY) == []


def broken_seed_data(collection: str, index: int, **fields: object) -> SeedData:
    records = agreeing_seed_records()
    records[collection][index].update(fields)
    return SeedData.model_validate(records)


@pytest.mark.parametrize(
    ('collection', 'index', 'fields', 'problem'),
    [
        (
            'employees',
            3,
            {'grade': 'senior associate'},
            "employee EMP-004: grade 'senior associate' is not a policy grade",
        ),
        (
            'approvers',
            2,
            {'approval_limit_gbp': '5000.00'},
            'approver EMP-003: approval limit 5000.00 is not the manager limit of 2500',
        ),
        (
            'approvers',
            2,
            {'employee': 'EMP-004'},
            "approver EMP-004: grade 'associate' does not approve submissions",
        ),
        (
            'historical_claims',
            0,
            {'currency': 'JPY'},
            'historical claim CLM-0001: currency JPY is not an accepted currency',
        ),
        (
            'historical_invoices',
            1,
            {'currency': 'CHF'},
            'historical invoice INV-0002: currency CHF is not an accepted currency',
        ),
        (
            'purchase_orders',
            0,
            {'currency': 'SEK'},
            'purchase order PO-0001: currency SEK is not an accepted currency',
        ),
        (
            'historical_claims',
            1,
            {'spent_on': '2026-10-05'},
            'historical claim CLM-0002: no company rate for EUR on 2026-10-05',
        ),
        (
            'historical_invoices',
            0,
            {'currency': 'USD'},
            'historical invoice INV-0001: no company rate for USD on 2026-10-07',
        ),
        (
            'historical_claims',
            1,
            {'category': 'hotel'},
            "historical claim CLM-0002: category 'hotel' is not a policy category",
        ),
        (
            'historical_claims',
            0,
            {'quantity': 1},
            'historical claim CLM-0001: a client entertainment quantity counts the claimant and'
            ' every guest, so it is at least 2, not 1',
        ),
        (
            'purchase_orders',
            0,
            {'raised_by': 'EMP-999'},
            'purchase order PO-0001: raised_by EMP-999 is not a known employee',
        ),
        (
            'cost_centres',
            1,
            {'spend_to_date_gbp': '1300.00'},
            'cost centre CC-200: spend to date 1300.00 is less than the 1341.00'
            ' of history charged to it since 2026-04-01',
        ),
        (
            'cost_centres',
            0,
            {'spend_to_date_gbp': '400.00'},
            'cost centre CC-100: spend to date 400.00 is less than the 416.00'
            ' of history charged to it since 2026-04-01',
        ),
    ],
)
def test_a_seed_record_that_disagrees_with_the_policy_is_reported(
    collection: str, index: int, fields: dict[str, object], problem: str
) -> None:
    seed_data = broken_seed_data(collection, index, **fields)

    assert check_policy_agreement(seed_data, HALDEN_POLICY) == [problem]


TOP_BAND = 'over 2500 up to 25000'


@pytest.mark.parametrize(
    ('removed_approvers', 'problems'),
    [
        (
            ['EMP-002'],
            [f'employee EMP-001: no approver other than themselves can decide {TOP_BAND}'],
        ),
        (
            ['EMP-001', 'EMP-002'],
            [
                f'employee EMP-001: no approver other than themselves can decide {TOP_BAND}',
                f'employee EMP-002: no approver other than themselves can decide {TOP_BAND}',
                'employee EMP-003: no approver other than themselves can decide up to 2500',
                f'employee EMP-003: no approver other than themselves can decide {TOP_BAND}',
                f'employee EMP-004: no approver other than themselves can decide {TOP_BAND}',
            ],
        ),
        (
            ['EMP-003'],
            [],
        ),
    ],
)
def test_an_amount_band_with_no_approver_other_than_the_requester_is_reported(
    removed_approvers: list[str], problems: list[str]
) -> None:
    records = agreeing_seed_records()
    records['approvers'] = [
        approver
        for approver in records['approvers']
        if approver['employee'] not in removed_approvers
    ]

    assert check_policy_agreement(SeedData.model_validate(records), HALDEN_POLICY) == problems


def test_history_from_before_the_budget_year_does_not_count_towards_spend_to_date() -> None:
    records = agreeing_seed_records()
    records['historical_claims'][0]['spent_on'] = '2026-03-31'
    records['cost_centres'][1]['spend_to_date_gbp'] = '1200.00'

    assert check_policy_agreement(SeedData.model_validate(records), HALDEN_POLICY) == []


def test_an_accepted_currency_missing_from_the_fx_table_is_reported() -> None:
    records = agreeing_seed_records()
    records['fx_rates'] = [rate for rate in records['fx_rates'] if rate['currency'] != 'USD']
    records['historical_invoices'][1]['currency'] = 'GBP'

    assert check_policy_agreement(SeedData.model_validate(records), HALDEN_POLICY) == [
        'the fx table has no rate for USD, a currency the policy accepts'
    ]


def test_an_employee_grade_with_no_limit_in_a_category_is_reported() -> None:
    accommodation = HALDEN_POLICY.expense_categories['accommodation']
    limits_without_manager = {
        grade: limit for grade, limit in accommodation.limits.items() if grade != 'manager'
    }
    policy = dataclasses.replace(
        HALDEN_POLICY,
        expense_categories={
            **HALDEN_POLICY.expense_categories,
            'accommodation': dataclasses.replace(accommodation, limits=limits_without_manager),
        },
    )
    seed_data = SeedData.model_validate(agreeing_seed_records())

    assert check_policy_agreement(seed_data, policy) == [
        "employee EMP-003: grade 'manager' has no accommodation limit"
    ]
