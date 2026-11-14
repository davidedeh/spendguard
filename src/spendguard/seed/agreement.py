import datetime
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal

from spendguard.policy.loader import Policy
from spendguard.seed.integrity import check_seed_integrity
from spendguard.seed.schema import SeedData

CLIENT_ENTERTAINMENT = 'client_entertainment'


def _string_list(policy: Policy, name: str) -> list[str]:
    value = policy.parameters[name]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f'policy parameter {name} is not a list of strings: {value!r}')
    return [item for item in value if isinstance(item, str)]


def _grade_problems(seed_data: SeedData, policy: Policy) -> Iterator[str]:
    grades = _string_list(policy, 'grades')
    for employee in seed_data.employees:
        if employee.grade not in grades:
            yield f'employee {employee.reference}: grade {employee.grade!r} is not a policy grade'
            continue
        for category in policy.expense_categories.values():
            if employee.grade not in category.limits:
                yield (
                    f'employee {employee.reference}: grade {employee.grade!r}'
                    f' has no {category.key} limit'
                )


def _approval_limits(policy: Policy) -> dict[str, Decimal]:
    value = policy.parameters['approval_limits']
    if not isinstance(value, dict):
        raise TypeError(f'policy parameter approval_limits is not a mapping: {value!r}')
    limits = {}
    for grade, limit in value.items():
        if not isinstance(limit, int | Decimal):
            raise TypeError(f'policy approval limit for {grade} is not a number: {limit!r}')
        limits[grade] = Decimal(limit)
    return limits


def _approver_problems(seed_data: SeedData, policy: Policy) -> Iterator[str]:
    approval_limits = _approval_limits(policy)
    grades = {employee.reference: employee.grade for employee in seed_data.employees}
    for approver in seed_data.approvers:
        grade = grades.get(approver.employee)
        if grade is None:
            continue
        if grade not in approval_limits:
            yield f'approver {approver.employee}: grade {grade!r} does not approve submissions'
        elif approver.approval_limit_gbp != approval_limits[grade]:
            yield (
                f'approver {approver.employee}: approval limit {approver.approval_limit_gbp}'
                f' is not the {grade} limit of {approval_limits[grade]}'
            )


def _amount_bands(policy: Policy) -> list[tuple[str, Decimal]]:
    band_tops = sorted(set(_approval_limits(policy).values()))
    return [
        (f'up to {top}' if bottom is None else f'over {bottom} up to {top}', top)
        for bottom, top in zip([None, *band_tops], band_tops, strict=False)
    ]


def _amount_band_problems(seed_data: SeedData, policy: Policy) -> Iterator[str]:
    bands = _amount_bands(policy)
    for employee in seed_data.employees:
        for band, top in bands:
            if not any(
                approver.employee != employee.reference and approver.approval_limit_gbp >= top
                for approver in seed_data.approvers
            ):
                yield (
                    f'employee {employee.reference}: no approver other than themselves'
                    f' can decide {band}'
                )


@dataclass(frozen=True)
class _PricedRecord:
    label: str
    amount: Decimal
    currency: str
    dated_on: datetime.date | None
    charged_to: str | None


def _priced_records(seed_data: SeedData) -> Iterator[_PricedRecord]:
    employee_cost_centres = {
        employee.reference: employee.cost_centre for employee in seed_data.employees
    }
    order_cost_centres = {order.reference: order.cost_centre for order in seed_data.purchase_orders}
    for order in seed_data.purchase_orders:
        yield _PricedRecord(
            label=f'purchase order {order.reference}',
            amount=order.amount,
            currency=order.currency,
            dated_on=None,
            charged_to=None,
        )
    for claim in seed_data.historical_claims:
        yield _PricedRecord(
            label=f'historical claim {claim.reference}',
            amount=claim.amount,
            currency=claim.currency,
            dated_on=claim.spent_on,
            charged_to=employee_cost_centres.get(claim.claimant),
        )
    for invoice in seed_data.historical_invoices:
        yield _PricedRecord(
            label=f'historical invoice {invoice.reference}',
            amount=invoice.amount,
            currency=invoice.currency,
            dated_on=invoice.issued_on,
            charged_to=(
                invoice.cost_centre
                if invoice.purchase_order is None
                else order_cost_centres.get(invoice.purchase_order)
            ),
        )


def _budget_year_start(policy: Policy, on: datetime.date) -> datetime.date:
    value = policy.parameters['budget_year_start']
    if not isinstance(value, str):
        raise TypeError(f'policy parameter budget_year_start is not a string: {value!r}')
    day_and_month = datetime.datetime.strptime(value, '%d %B').date()
    start = day_and_month.replace(year=on.year)
    return start if start <= on else start.replace(year=on.year - 1)


def _budget_problems(seed_data: SeedData, policy: Policy) -> Iterator[str]:
    spend = [record for record in _priced_records(seed_data) if record.charged_to is not None]
    dates = [record.dated_on for record in spend if record.dated_on is not None]
    if not dates:
        return
    year_start = _budget_year_start(policy, max(dates))
    history_totals: dict[str, Decimal] = {}
    for record in spend:
        if record.charged_to is None or record.dated_on is None or record.dated_on < year_start:
            continue
        rate = (
            Decimal(1)
            if record.currency == policy.base_currency
            else seed_data.company_rate(record.currency, record.dated_on)
        )
        if rate is None:
            continue
        history_totals[record.charged_to] = (
            history_totals.get(record.charged_to, Decimal(0)) + record.amount * rate
        )
    for centre in seed_data.cost_centres:
        history_total = history_totals.get(centre.reference, Decimal(0)).quantize(Decimal('0.01'))
        if centre.spend_to_date_gbp < history_total:
            yield (
                f'cost centre {centre.reference}: spend to date {centre.spend_to_date_gbp}'
                f' is less than the {history_total} of history charged to it'
                f' since {year_start.isoformat()}'
            )


def _currency_problems(seed_data: SeedData, policy: Policy) -> Iterator[str]:
    base_currency = policy.base_currency
    accepted_currencies = _string_list(policy, 'accepted_currencies')
    rate_currencies = {rate.currency for rate in seed_data.fx_rates}
    for currency in accepted_currencies:
        if currency != base_currency and currency not in rate_currencies:
            yield f'the fx table has no rate for {currency}, a currency the policy accepts'
    for record in _priced_records(seed_data):
        if record.currency not in accepted_currencies:
            yield f'{record.label}: currency {record.currency} is not an accepted currency'
        elif (
            record.currency != base_currency
            and record.dated_on is not None
            and seed_data.company_rate(record.currency, record.dated_on) is None
        ):
            yield (
                f'{record.label}: no company rate for {record.currency}'
                f' on {record.dated_on.isoformat()}'
            )


def _claim_problems(seed_data: SeedData, policy: Policy) -> Iterator[str]:
    for claim in seed_data.historical_claims:
        if claim.category not in policy.expense_categories:
            yield (
                f'historical claim {claim.reference}: category {claim.category!r}'
                ' is not a policy category'
            )
        elif claim.category == CLIENT_ENTERTAINMENT and claim.quantity < 2:
            yield (
                f'historical claim {claim.reference}: a client entertainment quantity of'
                f' {claim.quantity} leaves no guest besides the claimant'
            )


def check_policy_agreement(seed_data: SeedData, policy: Policy) -> list[str]:
    return [
        *check_seed_integrity(seed_data),
        *_grade_problems(seed_data, policy),
        *_approver_problems(seed_data, policy),
        *_amount_band_problems(seed_data, policy),
        *_currency_problems(seed_data, policy),
        *_claim_problems(seed_data, policy),
        *_budget_problems(seed_data, policy),
    ]
