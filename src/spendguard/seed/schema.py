import datetime
from decimal import Decimal
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from spendguard.strict_model import StrictModel

EmployeeReference = Annotated[str, StringConstraints(pattern=r'^EMP-\d{3,}$')]
CostCentreReference = Annotated[str, StringConstraints(pattern=r'^CC-\d{3,}$')]
SUPPLIER_REFERENCE_PATTERN = r'SUP-\d{3,}'
SupplierReference = Annotated[str, StringConstraints(pattern=rf'^{SUPPLIER_REFERENCE_PATTERN}$')]
PurchaseOrderReference = Annotated[str, StringConstraints(pattern=r'^PO-\d{4,}$')]
ClaimReference = Annotated[str, StringConstraints(pattern=r'^CLM-\d{4,}$')]
InvoiceReference = Annotated[str, StringConstraints(pattern=r'^INV-\d{4,}$')]
CurrencyCode = Annotated[str, StringConstraints(pattern=r'^[A-Z]{3}$')]
Amount = Annotated[Decimal, Field(gt=0, decimal_places=2)]
NonNegativeAmount = Annotated[Decimal, Field(ge=0, decimal_places=2)]
Quantity = Annotated[int, Field(ge=1)]


class Employee(StrictModel):
    reference: EmployeeReference
    grade: str
    cost_centre: CostCentreReference
    manager: EmployeeReference | None


class Approver(StrictModel):
    employee: EmployeeReference
    approval_limit_gbp: Amount


class CostCentre(StrictModel):
    reference: CostCentreReference
    owner: EmployeeReference | None
    budget_gbp: Amount
    spend_to_date_gbp: NonNegativeAmount


class Supplier(StrictModel):
    reference: SupplierReference
    approved: bool


class PurchaseOrder(StrictModel):
    reference: PurchaseOrderReference
    raised_by: EmployeeReference
    supplier: SupplierReference
    cost_centre: CostCentreReference
    amount: Amount
    currency: CurrencyCode


class ExpenseClaimFields(StrictModel):
    claimant: EmployeeReference
    spent_on: datetime.date
    category: str
    quantity: Quantity
    merchant: str
    amount: Amount
    currency: CurrencyCode

    @property
    def dated_on(self) -> datetime.date:
        return self.spent_on


class SupplierInvoiceFields(StrictModel):
    supplier: SupplierReference
    supplier_invoice_number: str
    purchase_order: PurchaseOrderReference | None
    cost_centre: CostCentreReference | None
    issued_on: datetime.date
    amount: Amount
    currency: CurrencyCode

    @property
    def dated_on(self) -> datetime.date:
        return self.issued_on

    @model_validator(mode='after')
    def _cost_centre_comes_from_the_purchase_order(self) -> Self:
        if self.purchase_order is not None and self.cost_centre is not None:
            raise ValueError(
                'give purchase_order or cost_centre, not both:'
                ' an invoice against a purchase order takes its cost centre from it'
            )
        return self


class HistoricalClaim(ExpenseClaimFields):
    reference: ClaimReference


class HistoricalInvoice(SupplierInvoiceFields):
    reference: InvoiceReference


class FxRate(StrictModel):
    currency: CurrencyCode
    rate_date: datetime.date
    gbp_per_unit: Annotated[Decimal, Field(gt=0)]


class PublicHoliday(StrictModel):
    holiday_on: datetime.date
    description: str


class SeedData(StrictModel):
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

    def company_rate_problem(
        self, currency: str, on: datetime.date, base_currency: str
    ) -> str | None:
        if currency == base_currency or self.company_rate(currency, on) is not None:
            return None
        return f'no company rate for {currency} on {on.isoformat()}'

    def invoice_requester(self, invoice: SupplierInvoiceFields) -> EmployeeReference | None:
        if invoice.purchase_order is not None:
            return next(
                (
                    order.raised_by
                    for order in self.purchase_orders
                    if order.reference == invoice.purchase_order
                ),
                None,
            )
        return next(
            (
                centre.owner
                for centre in self.cost_centres
                if centre.reference == invoice.cost_centre
            ),
            None,
        )
