import datetime
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal

from spendguard.policy.loader import FindingCode, Policy, Severity
from spendguard.seed.schema import (
    Employee,
    ExpenseClaimFields,
    HistoricalInvoice,
    SeedData,
    SupplierInvoiceFields,
)
from spendguard.submission import ExpenseClaim, Submission, SupplierInvoice

SATURDAY = 5
WEEKEND_CODE: FindingCode = 'WEEKEND_SPEND_UNJUSTIFIED'


class MissingCompanyRateError(ValueError):
    pass


@dataclass(frozen=True)
class Finding:
    code: FindingCode
    clause_reference: str
    severity: Severity


def company_rate(currency: str, on: datetime.date, seed_data: SeedData, policy: Policy) -> Decimal:
    if currency == policy.base_currency:
        return Decimal(1)
    rate = seed_data.company_rate(currency, on)
    if rate is None:
        raise MissingCompanyRateError(f'no company rate for {currency} on {on.isoformat()}')
    return rate


def base_currency_amount(
    submission: ExpenseClaimFields | SupplierInvoiceFields, seed_data: SeedData, policy: Policy
) -> Decimal:
    rate = company_rate(submission.currency, submission.dated_on, seed_data, policy)
    return submission.amount * rate


def falls_on_weekend_or_holiday(on: datetime.date, seed_data: SeedData) -> bool:
    return on.weekday() >= SATURDAY or any(
        holiday.holiday_on == on for holiday in seed_data.public_holidays
    )


def _claimant(claim: ExpenseClaim, seed_data: SeedData) -> Employee | None:
    return next(
        (employee for employee in seed_data.employees if employee.reference == claim.claimant),
        None,
    )


def _over_category_limit(claim: ExpenseClaim, seed_data: SeedData, policy: Policy) -> bool:
    category = policy.expense_categories.get(claim.category)
    claimant = _claimant(claim, seed_data)
    if category is None or claimant is None or claimant.grade not in category.limits:
        return False
    per_unit = base_currency_amount(claim, seed_data, policy) / claim.quantity
    return per_unit > category.limits[claimant.grade]


def _missing_receipt(claim: ExpenseClaim, seed_data: SeedData, policy: Policy) -> bool:
    return (
        claim.receipt_text is None
        and base_currency_amount(claim, seed_data, policy) > policy.receipt_threshold
    )


def _duplicate_claim(claim: ExpenseClaim, seed_data: SeedData, policy: Policy) -> bool:
    return any(
        (prior.claimant, prior.spent_on, prior.amount, prior.currency)
        == (claim.claimant, claim.spent_on, claim.amount, claim.currency)
        for prior in seed_data.historical_claims
    )


def _claim_budget_exceeded(claim: ExpenseClaim, seed_data: SeedData, policy: Policy) -> bool:
    claimant = _claimant(claim, seed_data)
    cost_centre = None if claimant is None else claimant.cost_centre
    return _takes_cost_centre_over_budget(cost_centre, claim, seed_data, policy)


def _takes_cost_centre_over_budget(
    cost_centre_reference: str | None,
    submission: ExpenseClaim | SupplierInvoice,
    seed_data: SeedData,
    policy: Policy,
) -> bool:
    cost_centre = next(
        (centre for centre in seed_data.cost_centres if centre.reference == cost_centre_reference),
        None,
    )
    if cost_centre is None:
        return False
    spend = cost_centre.spend_to_date_gbp + base_currency_amount(submission, seed_data, policy)
    return spend > cost_centre.budget_gbp


def _currency_anomaly(
    submission: ExpenseClaim | SupplierInvoice, seed_data: SeedData, policy: Policy
) -> bool:
    if submission.claimed_exchange_rate is None or submission.currency == policy.base_currency:
        return False
    rate = company_rate(submission.currency, submission.dated_on, seed_data, policy)
    deviation = abs(submission.claimed_exchange_rate - rate) / rate
    return deviation * 100 > policy.currency_deviation_percent


def _non_approved_supplier(invoice: SupplierInvoice, seed_data: SeedData, policy: Policy) -> bool:
    return not any(
        supplier.approved
        for supplier in seed_data.suppliers
        if supplier.reference == invoice.supplier
    )


def _missing_po(invoice: SupplierInvoice, seed_data: SeedData, policy: Policy) -> bool:
    return (
        invoice.purchase_order is None
        and base_currency_amount(invoice, seed_data, policy) > policy.purchase_order_threshold
    )


def charged_cost_centre(invoice: SupplierInvoice, seed_data: SeedData) -> str | None:
    if invoice.purchase_order is None:
        return invoice.cost_centre
    return next(
        (
            order.cost_centre
            for order in seed_data.purchase_orders
            if order.reference == invoice.purchase_order
        ),
        None,
    )


def _invoice_budget_exceeded(invoice: SupplierInvoice, seed_data: SeedData, policy: Policy) -> bool:
    return _takes_cost_centre_over_budget(
        charged_cost_centre(invoice, seed_data), invoice, seed_data, policy
    )


def _split_purchase(invoice: SupplierInvoice, seed_data: SeedData, policy: Policy) -> bool:
    if invoice.purchase_order is not None:
        return False
    own_amount = base_currency_amount(invoice, seed_data, policy)
    if own_amount > policy.purchase_order_threshold:
        return False
    if invoice.cost_centre is None:
        return False
    priors = prior_unordered_invoices(
        invoice.supplier, invoice.cost_centre, invoice.issued_on, seed_data, policy
    )
    combined = own_amount + sum(
        (base_currency_amount(prior, seed_data, policy) for prior in priors), Decimal(0)
    )
    return combined > policy.purchase_order_threshold


def prior_unordered_invoices(
    supplier: str, cost_centre: str, on: datetime.date, seed_data: SeedData, policy: Policy
) -> tuple[HistoricalInvoice, ...]:
    window = datetime.timedelta(days=policy.split_purchase_window_days)
    return tuple(
        prior
        for prior in seed_data.historical_invoices
        if prior.purchase_order is None
        and prior.supplier == supplier
        and prior.cost_centre == cost_centre
        and abs(prior.issued_on - on) <= window
    )


def _duplicate_invoice(invoice: SupplierInvoice, seed_data: SeedData, policy: Policy) -> bool:
    return any(
        prior.supplier == invoice.supplier
        and (
            prior.supplier_invoice_number == invoice.supplier_invoice_number
            or (prior.issued_on, prior.amount, prior.currency)
            == (invoice.issued_on, invoice.amount, invoice.currency)
        )
        for prior in seed_data.historical_invoices
    )


def _weekend_gate(claim: ExpenseClaim, seed_data: SeedData, policy: Policy) -> bool:
    return falls_on_weekend_or_holiday(claim.spent_on, seed_data)


ClaimCheck = Callable[[ExpenseClaim, SeedData, Policy], bool]
InvoiceCheck = Callable[[SupplierInvoice, SeedData, Policy], bool]

EXPENSE_CLAIM_CHECKS: Mapping[FindingCode, ClaimCheck] = {
    'OVER_CATEGORY_LIMIT': _over_category_limit,
    'MISSING_RECEIPT': _missing_receipt,
    'DUPLICATE_SUBMISSION': _duplicate_claim,
    'BUDGET_EXCEEDED': _claim_budget_exceeded,
    'CURRENCY_ANOMALY': _currency_anomaly,
}
EXPENSE_CLAIM_JUDGEMENT_GATES: Mapping[FindingCode, ClaimCheck] = {
    WEEKEND_CODE: _weekend_gate,
}
SUPPLIER_INVOICE_JUDGEMENT_CHECKS: tuple[FindingCode, ...] = ('PURPOSE_UNCLEAR',)
SUPPLIER_INVOICE_CHECKS: Mapping[FindingCode, InvoiceCheck] = {
    'DUPLICATE_SUBMISSION': _duplicate_invoice,
    'SPLIT_PURCHASE': _split_purchase,
    'NON_APPROVED_SUPPLIER': _non_approved_supplier,
    'MISSING_PO': _missing_po,
    'BUDGET_EXCEEDED': _invoice_budget_exceeded,
    'CURRENCY_ANOMALY': _currency_anomaly,
}


def rule_findings(
    submission: Submission, seed_data: SeedData, policy: Policy
) -> tuple[Finding, ...]:
    if isinstance(submission, ExpenseClaim):
        claim = submission
        raised = [
            code
            for code in policy.applicability['expense_claim']
            if code in EXPENSE_CLAIM_CHECKS and EXPENSE_CLAIM_CHECKS[code](claim, seed_data, policy)
        ]
    else:
        invoice = submission
        raised = [
            code
            for code in policy.applicability['supplier_invoice']
            if code in SUPPLIER_INVOICE_CHECKS
            and SUPPLIER_INVOICE_CHECKS[code](invoice, seed_data, policy)
        ]
    return tuple(finding_for(code, policy) for code in raised)


def judgement_checks(
    submission: Submission, seed_data: SeedData, policy: Policy
) -> list[FindingCode]:
    if isinstance(submission, ExpenseClaim):
        claim = submission
        return [
            code
            for code in policy.codes_with_check('judgement')
            if code not in EXPENSE_CLAIM_JUDGEMENT_GATES
            or EXPENSE_CLAIM_JUDGEMENT_GATES[code](claim, seed_data, policy)
        ]
    return [
        code
        for code in policy.codes_with_check('judgement')
        if code in SUPPLIER_INVOICE_JUDGEMENT_CHECKS
    ]


def raised_rule_codes(
    submission: Submission, seed_data: SeedData, policy: Policy
) -> list[FindingCode]:
    return sorted(finding.code for finding in rule_findings(submission, seed_data, policy))


def finding_for(code: FindingCode, policy: Policy) -> Finding:
    definition = policy.finding_codes[code]
    return Finding(
        code=code, clause_reference=definition.clause.reference, severity=definition.severity
    )
