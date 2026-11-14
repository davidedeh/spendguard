from collections import Counter
from collections.abc import Iterable, Iterator
from typing import assert_never

from spendguard.seed.schema import (
    Approver,
    CostCentre,
    Employee,
    FxRate,
    HistoricalClaim,
    HistoricalInvoice,
    PublicHoliday,
    PurchaseOrder,
    SeedData,
    Supplier,
)

COLLECTION_KINDS = {
    'employees': 'employee',
    'approvers': 'approver',
    'cost_centres': 'cost centre',
    'suppliers': 'supplier',
    'purchase_orders': 'purchase order',
    'historical_claims': 'historical claim',
    'historical_invoices': 'historical invoice',
    'fx_rates': 'fx rate',
    'public_holidays': 'public holiday',
}

REFERENCE_FIELDS = (
    ('employees', 'cost_centre', 'cost_centres'),
    ('employees', 'manager', 'employees'),
    ('approvers', 'employee', 'employees'),
    ('cost_centres', 'owner', 'employees'),
    ('purchase_orders', 'raised_by', 'employees'),
    ('purchase_orders', 'supplier', 'suppliers'),
    ('purchase_orders', 'cost_centre', 'cost_centres'),
    ('historical_claims', 'claimant', 'employees'),
    ('historical_invoices', 'supplier', 'suppliers'),
    ('historical_invoices', 'purchase_order', 'purchase_orders'),
    ('historical_invoices', 'cost_centre', 'cost_centres'),
)


AnySeedRecord = (
    Employee
    | Approver
    | CostCentre
    | Supplier
    | PurchaseOrder
    | HistoricalClaim
    | HistoricalInvoice
    | FxRate
    | PublicHoliday
)


def _records(seed_data: SeedData, collection: str) -> Iterable[AnySeedRecord]:
    records: Iterable[AnySeedRecord] = getattr(seed_data, collection)
    return records


def _identity(record: AnySeedRecord) -> str:
    match record:
        case Approver():
            return record.employee
        case FxRate():
            return f'{record.currency} {record.rate_date.isoformat()}'
        case PublicHoliday():
            return record.holiday_on.isoformat()
        case (
            Employee()
            | CostCentre()
            | Supplier()
            | PurchaseOrder()
            | HistoricalClaim()
            | HistoricalInvoice()
        ):
            return record.reference
        case _:
            assert_never(record)


def _duplicates(seed_data: SeedData) -> Iterator[str]:
    for collection, kind in COLLECTION_KINDS.items():
        counts = Counter(_identity(record) for record in _records(seed_data, collection))
        for identity, count in counts.items():
            if count > 1:
                yield f'{kind} {identity} appears more than once'


def _dangling_references(seed_data: SeedData) -> Iterator[str]:
    for collection, field, target_collection in REFERENCE_FIELDS:
        known = {_identity(target) for target in _records(seed_data, target_collection)}
        for record in _records(seed_data, collection):
            value = getattr(record, field)
            if value is not None and value not in known:
                yield (
                    f'{COLLECTION_KINDS[collection]} {_identity(record)}: {field} {value}'
                    f' is not a known {COLLECTION_KINDS[target_collection]}'
                )


def _missing_invoice_requesters(seed_data: SeedData) -> Iterator[str]:
    known_cost_centres = {centre.reference for centre in seed_data.cost_centres}
    for invoice in seed_data.historical_invoices:
        if invoice.purchase_order is not None:
            continue
        if invoice.cost_centre is not None and invoice.cost_centre not in known_cost_centres:
            continue
        if seed_data.invoice_requester(None, invoice.cost_centre) is None:
            yield (
                f'historical invoice {invoice.reference}: no requester, since it has no'
                ' purchase order and no cost centre owner'
            )


def check_seed_integrity(seed_data: SeedData) -> list[str]:
    return [
        *_duplicates(seed_data),
        *_dangling_references(seed_data),
        *_missing_invoice_requesters(seed_data),
    ]
