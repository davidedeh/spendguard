import dataclasses
import datetime
from pathlib import Path
from typing import Any

import pytest

from spendguard.policy.loader import POLICY_PATH, FindingCode, load_policy
from spendguard.rules.checks import (
    EXPENSE_CLAIM_CHECKS,
    EXPENSE_CLAIM_JUDGEMENT_GATES,
    SUPPLIER_INVOICE_CHECKS,
    Finding,
    falls_on_weekend_or_holiday,
    judgement_checks,
    rule_findings,
)
from spendguard.seed.schema import SeedData
from spendguard.submission import ExpenseClaim, SubmissionType, SupplierInvoice

FIXTURES = Path(__file__).parent / 'fixtures'
POLICY = load_policy(FIXTURES / 'policy' / 'valid_policy.yaml')
POLICY_CHECKING_CLAIM_BUDGETS = dataclasses.replace(
    POLICY,
    applicability={
        'expense_claim': (*POLICY.applicability['expense_claim'], 'BUDGET_EXCEEDED'),
        'supplier_invoice': POLICY.applicability['supplier_invoice'],
    },
)
RECEIPT = 'Harbour Kitchen\n6 Oct 2026\n1 x Soup 6.50\n1 x Sandwich 8.00\nTOTAL 14.50'


def seed_data(**overrides: Any) -> SeedData:
    records: dict[str, Any] = {
        'employees': [
            {'reference': 'EMP-001', 'grade': 'junior', 'cost_centre': 'CC-100', 'manager': None},
            {'reference': 'EMP-002', 'grade': 'senior', 'cost_centre': 'CC-200', 'manager': None},
        ],
        'approvers': [],
        'cost_centres': [
            {
                'reference': 'CC-100',
                'owner': 'EMP-001',
                'budget_gbp': '5000.00',
                'spend_to_date_gbp': '1000.00',
            },
            {
                'reference': 'CC-200',
                'owner': 'EMP-002',
                'budget_gbp': '3000.00',
                'spend_to_date_gbp': '2900.00',
            },
        ],
        'suppliers': [
            {'reference': 'SUP-001', 'approved': True},
            {'reference': 'SUP-002', 'approved': False},
        ],
        'purchase_orders': [
            {
                'reference': 'PO-0001',
                'raised_by': 'EMP-001',
                'supplier': 'SUP-001',
                'cost_centre': 'CC-200',
                'amount': '1200.00',
                'currency': 'GBP',
            }
        ],
        'historical_claims': [],
        'historical_invoices': [],
        'fx_rates': [{'currency': 'EUR', 'rate_date': '2026-10-06', 'gbp_per_unit': '0.80'}],
        'public_holidays': [{'holiday_on': '2026-10-12', 'description': 'Founders day'}],
    }
    records.update(overrides)
    return SeedData.model_validate(records)


def claim(**overrides: Any) -> ExpenseClaim:
    fields: dict[str, Any] = {
        'type': 'expense_claim',
        'claimant': 'EMP-001',
        'spent_on': '2026-10-06',
        'category': 'subsistence',
        'quantity': 1,
        'merchant': 'Harbour Kitchen',
        'amount': '14.50',
        'currency': 'GBP',
        'claimed_exchange_rate': None,
        'business_purpose': 'Lunch with the audit team at the client site.',
        'receipt_text': RECEIPT,
    }
    fields.update(overrides)
    return ExpenseClaim.model_validate(fields)


def invoice(**overrides: Any) -> SupplierInvoice:
    fields: dict[str, Any] = {
        'type': 'supplier_invoice',
        'supplier': 'SUP-001',
        'supplier_invoice_number': 'A-100',
        'purchase_order': None,
        'cost_centre': 'CC-100',
        'issued_on': '2026-10-06',
        'amount': '300.00',
        'currency': 'GBP',
        'claimed_exchange_rate': None,
        'description': 'Printer toner and paper for the finance office.',
    }
    fields.update(overrides)
    return SupplierInvoice.model_validate(fields)


def codes(findings: tuple[Finding, ...]) -> tuple[FindingCode, ...]:
    return tuple(finding.code for finding in findings)


@pytest.mark.parametrize(
    ('overrides', 'raised'),
    [
        ({'amount': '39.99'}, False),
        ({'amount': '40.00'}, False),
        ({'amount': '40.01'}, True),
        ({'claimant': 'EMP-002', 'amount': '55.00'}, False),
        ({'claimant': 'EMP-002', 'amount': '55.01'}, True),
        ({'quantity': 3, 'amount': '120.00'}, False),
        ({'quantity': 3, 'amount': '120.03'}, True),
        ({'currency': 'EUR', 'amount': '50.00'}, False),
        ({'currency': 'EUR', 'amount': '50.02'}, True),
    ],
)
def test_over_category_limit_compares_the_base_amount_per_unit_with_the_grade_limit(
    overrides: dict[str, Any], raised: bool
) -> None:
    findings = rule_findings(claim(**overrides), seed_data(), POLICY)

    assert codes(findings) == (('OVER_CATEGORY_LIMIT',) if raised else ())


@pytest.mark.parametrize(
    ('overrides', 'raised'),
    [
        ({'amount': '24.99', 'receipt_text': None}, False),
        ({'amount': '25.00', 'receipt_text': None}, False),
        ({'amount': '25.01', 'receipt_text': None}, True),
        ({'amount': '25.01'}, False),
        ({'currency': 'EUR', 'amount': '31.25', 'receipt_text': None}, False),
        ({'currency': 'EUR', 'amount': '31.27', 'receipt_text': None}, True),
    ],
)
def test_missing_receipt_is_raised_over_the_receipt_threshold_without_a_receipt(
    overrides: dict[str, Any], raised: bool
) -> None:
    findings = rule_findings(claim(**overrides), seed_data(), POLICY)

    assert codes(findings) == (('MISSING_RECEIPT',) if raised else ())


PRIOR_CLAIM = {
    'reference': 'CLM-0001',
    'claimant': 'EMP-001',
    'spent_on': '2026-10-06',
    'category': 'subsistence',
    'quantity': 1,
    'merchant': 'Harbour Kitchen',
    'amount': '14.50',
    'currency': 'GBP',
}


@pytest.mark.parametrize(
    ('prior_overrides', 'raised'),
    [
        ({}, True),
        ({'merchant': 'Harbour Kitchen Ltd'}, True),
        ({'amount': '14.49'}, False),
        ({'spent_on': '2026-10-05'}, False),
        ({'claimant': 'EMP-002'}, False),
        ({'currency': 'EUR'}, False),
    ],
)
def test_duplicate_submission_matches_a_prior_claim_on_claimant_date_amount_and_currency(
    prior_overrides: dict[str, Any], raised: bool
) -> None:
    seed = seed_data(historical_claims=[{**PRIOR_CLAIM, **prior_overrides}])

    findings = rule_findings(claim(), seed, POLICY)

    assert codes(findings) == (('DUPLICATE_SUBMISSION',) if raised else ())


@pytest.mark.parametrize(
    ('overrides', 'raised'),
    [
        ({'claimant': 'EMP-002', 'amount': '20.00'}, False),
        ({'claimant': 'EMP-002', 'amount': '100.00', 'quantity': 2}, False),
        ({'claimant': 'EMP-002', 'amount': '100.01', 'quantity': 2}, True),
        ({'claimant': 'EMP-002', 'currency': 'EUR', 'amount': '125.00', 'quantity': 3}, False),
        ({'claimant': 'EMP-002', 'currency': 'EUR', 'amount': '125.02', 'quantity': 3}, True),
    ],
)
def test_budget_exceeded_adds_the_base_amount_to_the_claimants_cost_centre_spend(
    overrides: dict[str, Any], raised: bool
) -> None:
    findings = rule_findings(claim(**overrides), seed_data(), POLICY_CHECKING_CLAIM_BUDGETS)

    assert codes(findings) == (('BUDGET_EXCEEDED',) if raised else ())


@pytest.mark.parametrize(
    ('claimed_exchange_rate', 'raised'),
    [
        (None, False),
        ('0.824', False),
        ('0.8241', True),
        ('0.776', False),
        ('0.7759', True),
    ],
)
def test_currency_anomaly_compares_the_claimed_rate_with_the_company_rate(
    claimed_exchange_rate: str | None, raised: bool
) -> None:
    submission = claim(currency='EUR', amount='14.50', claimed_exchange_rate=claimed_exchange_rate)

    findings = rule_findings(submission, seed_data(), POLICY)

    assert codes(findings) == (('CURRENCY_ANOMALY',) if raised else ())


def test_a_finding_takes_its_clause_and_severity_from_the_policy() -> None:
    severities_swapped = dataclasses.replace(
        POLICY,
        finding_codes={
            **POLICY.finding_codes,
            'OVER_CATEGORY_LIMIT': dataclasses.replace(
                POLICY.finding_codes['OVER_CATEGORY_LIMIT'], severity='high'
            ),
        },
    )

    [finding] = rule_findings(claim(amount='40.01'), seed_data(), severities_swapped)

    assert finding == Finding(
        code='OVER_CATEGORY_LIMIT',
        clause_reference=POLICY.finding_codes['OVER_CATEGORY_LIMIT'].clause.reference,
        severity='high',
    )


def test_only_the_codes_the_applicability_table_lists_for_claims_run() -> None:
    over_budget = claim(claimant='EMP-002', amount='100.01', quantity=2)

    assert codes(rule_findings(over_budget, seed_data(), POLICY)) == ()
    assert codes(rule_findings(over_budget, seed_data(), POLICY_CHECKING_CLAIM_BUDGETS)) == (
        'BUDGET_EXCEEDED',
    )


@pytest.mark.parametrize(
    ('spent_on', 'gated'),
    [
        (datetime.date(2026, 10, 9), False),
        (datetime.date(2026, 10, 10), True),
        (datetime.date(2026, 10, 11), True),
        (datetime.date(2026, 10, 12), True),
        (datetime.date(2026, 10, 13), False),
    ],
)
def test_the_weekend_gate_opens_on_a_saturday_a_sunday_or_a_public_holiday(
    spent_on: datetime.date, gated: bool
) -> None:
    assert falls_on_weekend_or_holiday(spent_on, seed_data()) is gated


@pytest.mark.parametrize(
    ('overrides', 'expected'),
    [
        ({}, ()),
        ({'supplier': 'SUP-002'}, ('NON_APPROVED_SUPPLIER',)),
    ],
)
def test_non_approved_supplier_is_raised_for_a_supplier_not_on_the_approved_list(
    overrides: dict[str, Any], expected: tuple[FindingCode, ...]
) -> None:
    assert codes(rule_findings(invoice(**overrides), seed_data(), POLICY)) == expected


@pytest.mark.parametrize(
    ('overrides', 'raised'),
    [
        ({'amount': '499.99'}, False),
        ({'amount': '500.00'}, False),
        ({'amount': '500.01'}, True),
        ({'currency': 'EUR', 'amount': '625.00'}, False),
        ({'currency': 'EUR', 'amount': '625.02'}, True),
        ({'purchase_order': 'PO-0001', 'cost_centre': None, 'amount': '100.00'}, False),
    ],
)
def test_missing_po_is_raised_over_the_purchase_order_threshold_without_a_purchase_order(
    overrides: dict[str, Any], raised: bool
) -> None:
    findings = rule_findings(invoice(**overrides), seed_data(), POLICY)

    assert codes(findings) == (('MISSING_PO',) if raised else ())


@pytest.mark.parametrize(
    ('overrides', 'raised'),
    [
        ({'purchase_order': 'PO-0001', 'cost_centre': None, 'amount': '99.99'}, False),
        ({'purchase_order': 'PO-0001', 'cost_centre': None, 'amount': '100.00'}, False),
        ({'purchase_order': 'PO-0001', 'cost_centre': None, 'amount': '100.01'}, True),
        ({'cost_centre': 'CC-200', 'amount': '100.01'}, True),
        ({'cost_centre': 'CC-100', 'amount': '100.01'}, False),
    ],
)
def test_budget_exceeded_uses_the_purchase_orders_cost_centre_or_the_invoices_own(
    overrides: dict[str, Any], raised: bool
) -> None:
    findings = rule_findings(invoice(**overrides), seed_data(), POLICY)

    assert codes(findings) == (('BUDGET_EXCEEDED',) if raised else ())


PRIOR_INVOICE = {
    'reference': 'INV-0001',
    'supplier': 'SUP-001',
    'supplier_invoice_number': 'A-090',
    'purchase_order': None,
    'cost_centre': 'CC-100',
    'issued_on': '2026-09-20',
    'amount': '300.00',
    'currency': 'GBP',
}


@pytest.mark.parametrize(
    ('prior_overrides', 'amount', 'raised'),
    [
        ({}, '200.00', False),
        ({}, '200.01', True),
        ({'issued_on': '2026-09-15'}, '200.01', True),
        ({'issued_on': '2026-09-14'}, '200.01', False),
        ({'issued_on': '2026-10-27'}, '200.01', True),
        ({'issued_on': '2026-10-28'}, '200.01', False),
        ({'cost_centre': 'CC-200'}, '200.01', False),
        ({'supplier': 'SUP-002'}, '200.01', False),
        ({'purchase_order': 'PO-0001', 'cost_centre': None}, '200.01', False),
    ],
)
def test_split_purchase_combines_prior_invoices_from_the_supplier_to_the_cost_centre(
    prior_overrides: dict[str, Any], amount: str, raised: bool
) -> None:
    seed = seed_data(historical_invoices=[{**PRIOR_INVOICE, **prior_overrides}])

    findings = rule_findings(invoice(amount=amount), seed, POLICY)

    assert codes(findings) == (('SPLIT_PURCHASE',) if raised else ())


def test_an_invoice_over_the_threshold_on_its_own_is_missing_a_po_rather_than_split() -> None:
    seed = seed_data(historical_invoices=[PRIOR_INVOICE])

    findings = rule_findings(invoice(amount='500.01'), seed, POLICY)

    assert codes(findings) == ('MISSING_PO',)


PRIOR_PO_INVOICE = {
    'reference': 'INV-0002',
    'supplier': 'SUP-001',
    'supplier_invoice_number': 'A-100',
    'purchase_order': 'PO-0001',
    'cost_centre': None,
    'issued_on': '2026-10-06',
    'amount': '50.00',
    'currency': 'GBP',
}


@pytest.mark.parametrize(
    ('prior_overrides', 'raised'),
    [
        ({}, True),
        ({'amount': '49.00'}, True),
        ({'supplier_invoice_number': 'A-099'}, True),
        ({'supplier_invoice_number': 'A-099', 'amount': '49.00'}, False),
        ({'supplier_invoice_number': 'A-099', 'issued_on': '2026-10-05'}, False),
        ({'supplier': 'SUP-002'}, False),
    ],
)
def test_duplicate_submission_matches_a_prior_invoice_on_number_or_date_and_amount(
    prior_overrides: dict[str, Any], raised: bool
) -> None:
    seed = seed_data(historical_invoices=[{**PRIOR_PO_INVOICE, **prior_overrides}])
    submission = invoice(purchase_order='PO-0001', cost_centre=None, amount='50.00')

    findings = rule_findings(submission, seed, POLICY)

    assert codes(findings) == (('DUPLICATE_SUBMISSION',) if raised else ())


def test_currency_anomaly_runs_for_invoices_too() -> None:
    submission = invoice(currency='EUR', claimed_exchange_rate='0.8241')

    assert codes(rule_findings(submission, seed_data(), POLICY)) == ('CURRENCY_ANOMALY',)


def test_only_the_codes_the_applicability_table_lists_for_invoices_run() -> None:
    policy_without_supplier_check = dataclasses.replace(
        POLICY,
        applicability={
            'expense_claim': POLICY.applicability['expense_claim'],
            'supplier_invoice': tuple(
                code
                for code in POLICY.applicability['supplier_invoice']
                if code != 'NON_APPROVED_SUPPLIER'
            ),
        },
    )
    unapproved = invoice(supplier='SUP-002')

    assert codes(rule_findings(unapproved, seed_data(), policy_without_supplier_check)) == ()


def test_judgement_checks_run_only_where_the_applicability_table_lists_them() -> None:
    policy_judging_alcohol_on_invoices = dataclasses.replace(
        POLICY,
        applicability={
            'expense_claim': (*POLICY.applicability['expense_claim'], 'PURPOSE_UNCLEAR'),
            'supplier_invoice': (
                *POLICY.applicability['supplier_invoice'],
                'ALCOHOL_NOT_PERMITTED',
            ),
        },
    )
    saturday_claim = claim(spent_on='2026-10-10')

    assert judgement_checks(invoice(), seed_data(), policy_judging_alcohol_on_invoices) == [
        'ALCOHOL_NOT_PERMITTED'
    ]
    assert judgement_checks(saturday_claim, seed_data(), policy_judging_alcohol_on_invoices) == [
        'WEEKEND_SPEND_UNJUSTIFIED',
        'PURPOSE_UNCLEAR',
    ]


@pytest.mark.parametrize(
    ('submission_type', 'registered_codes'),
    [
        ('expense_claim', set(EXPENSE_CLAIM_CHECKS)),
        ('supplier_invoice', set(SUPPLIER_INVOICE_CHECKS)),
    ],
)
def test_every_rule_code_the_halden_policy_applies_has_a_registered_rule_function(
    submission_type: SubmissionType, registered_codes: set[FindingCode]
) -> None:
    halden_policy = load_policy(POLICY_PATH)
    applicable_rule_codes = {
        code
        for code in halden_policy.applicability[submission_type]
        if halden_policy.finding_codes[code].check == 'rule'
    }

    assert applicable_rule_codes - registered_codes == set()


def test_the_weekend_gate_is_registered_for_weekend_spend_on_claims() -> None:
    gate = EXPENSE_CLAIM_JUDGEMENT_GATES['WEEKEND_SPEND_UNJUSTIFIED']

    assert gate(claim(spent_on='2026-10-10'), seed_data(), POLICY) is True
    assert gate(claim(spent_on='2026-10-09'), seed_data(), POLICY) is False
