import datetime
from decimal import Decimal
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

EmployeeReference = Annotated[str, StringConstraints(pattern=r'^EMP-\d{3,}$')]
CostCentreReference = Annotated[str, StringConstraints(pattern=r'^CC-\d{3,}$')]
SupplierReference = Annotated[str, StringConstraints(pattern=r'^SUP-\d{3,}$')]
PurchaseOrderReference = Annotated[str, StringConstraints(pattern=r'^PO-\d{4,}$')]
ClaimReference = Annotated[str, StringConstraints(pattern=r'^CLM-\d{4,}$')]
InvoiceReference = Annotated[str, StringConstraints(pattern=r'^INV-\d{4,}$')]
CurrencyCode = Annotated[str, StringConstraints(pattern=r'^[A-Z]{3}$')]
Amount = Annotated[Decimal, Field(gt=0, decimal_places=2)]
NonNegativeAmount = Annotated[Decimal, Field(ge=0, decimal_places=2)]
Quantity = Annotated[int, Field(ge=1)]


class SeedRecord(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class Employee(SeedRecord):
    reference: EmployeeReference
    grade: str
    cost_centre: CostCentreReference
    manager: EmployeeReference | None


class Approver(SeedRecord):
    employee: EmployeeReference
    approval_limit_gbp: Amount


class CostCentre(SeedRecord):
    reference: CostCentreReference
    owner: EmployeeReference | None
    budget_gbp: Amount
    spend_to_date_gbp: NonNegativeAmount


class Supplier(SeedRecord):
    reference: SupplierReference
    approved: bool


class PurchaseOrder(SeedRecord):
    reference: PurchaseOrderReference
    raised_by: EmployeeReference
    supplier: SupplierReference
    cost_centre: CostCentreReference
    amount: Amount
    currency: CurrencyCode


class HistoricalClaim(SeedRecord):
    reference: ClaimReference
    claimant: EmployeeReference
    spent_on: datetime.date
    category: str
    quantity: Quantity
    merchant: str
    amount: Amount
    currency: CurrencyCode


class HistoricalInvoice(SeedRecord):
    reference: InvoiceReference
    supplier: SupplierReference
    supplier_invoice_number: str
    purchase_order: PurchaseOrderReference | None
    cost_centre: CostCentreReference | None
    issued_on: datetime.date
    amount: Amount
    currency: CurrencyCode

    @model_validator(mode='after')
    def _cost_centre_comes_from_the_purchase_order(self) -> Self:
        if self.purchase_order is not None and self.cost_centre is not None:
            raise ValueError('an invoice against a purchase order takes its cost centre from it')
        return self


class FxRate(SeedRecord):
    currency: CurrencyCode
    rate_date: datetime.date
    gbp_per_unit: Annotated[Decimal, Field(gt=0)]


class PublicHoliday(SeedRecord):
    holiday_on: datetime.date
    description: str


class SeedData(SeedRecord):
    employees: tuple[Employee, ...]
    approvers: tuple[Approver, ...]
    cost_centres: tuple[CostCentre, ...]
    suppliers: tuple[Supplier, ...]
    purchase_orders: tuple[PurchaseOrder, ...]
    historical_claims: tuple[HistoricalClaim, ...]
    historical_invoices: tuple[HistoricalInvoice, ...]
    fx_rates: tuple[FxRate, ...]
    public_holidays: tuple[PublicHoliday, ...]

    def company_rate(self, currency: str, on: datetime.date) -> Decimal | None:
        return next(
            (
                rate.gbp_per_unit
                for rate in self.fx_rates
                if rate.currency == currency and rate.rate_date == on
            ),
            None,
        )

    def invoice_requester(
        self, purchase_order: str | None, cost_centre: str | None
    ) -> EmployeeReference | None:
        if purchase_order is not None:
            return next(
                (
                    order.raised_by
                    for order in self.purchase_orders
                    if order.reference == purchase_order
                ),
                None,
            )
        return next(
            (centre.owner for centre in self.cost_centres if centre.reference == cost_centre),
            None,
        )
