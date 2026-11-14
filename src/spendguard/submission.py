from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field

from spendguard.seed.schema import ExpenseClaimFields, SupplierInvoiceFields

ExchangeRate = Annotated[Decimal, Field(gt=0)]
ExpenseClaimType = Literal['expense_claim']
SupplierInvoiceType = Literal['supplier_invoice']
SubmissionType = Literal[ExpenseClaimType, SupplierInvoiceType]


class ExpenseClaim(ExpenseClaimFields):
    type: ExpenseClaimType
    claimed_gbp_per_unit: ExchangeRate | None
    business_purpose: str
    receipt_text: str | None


class SupplierInvoice(SupplierInvoiceFields):
    type: SupplierInvoiceType
    claimed_gbp_per_unit: ExchangeRate | None
    description: str


Submission = Annotated[ExpenseClaim | SupplierInvoice, Field(discriminator='type')]
