import datetime
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spendguard.seed.schema import (
    Amount,
    CostCentreReference,
    CurrencyCode,
    EmployeeReference,
    PurchaseOrderReference,
    Quantity,
    SupplierReference,
)

ExchangeRate = Annotated[Decimal, Field(gt=0)]


class _SubmissionModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class ExpenseClaim(_SubmissionModel):
    type: Literal['expense_claim']
    claimant: EmployeeReference
    spent_on: datetime.date
    category: str
    quantity: Quantity
    merchant: str
    amount: Amount
    currency: CurrencyCode
    claimed_gbp_per_unit: ExchangeRate | None
    business_purpose: str
    receipt_text: str | None


class SupplierInvoice(_SubmissionModel):
    type: Literal['supplier_invoice']
    supplier: SupplierReference
    supplier_invoice_number: str
    purchase_order: PurchaseOrderReference | None
    cost_centre: CostCentreReference | None
    issued_on: datetime.date
    amount: Amount
    currency: CurrencyCode
    claimed_gbp_per_unit: ExchangeRate | None
    description: str

    @model_validator(mode='after')
    def _cost_centre_comes_from_the_purchase_order(self) -> Self:
        if self.purchase_order is not None and self.cost_centre is not None:
            raise ValueError('an invoice against a purchase order takes its cost centre from it')
        return self


Submission = Annotated[ExpenseClaim | SupplierInvoice, Field(discriminator='type')]
