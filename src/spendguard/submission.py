from decimal import Decimal
from typing import Annotated, Literal

from pydantic import Field

from spendguard.seed.schema import ExpenseClaimFields, SupplierInvoiceFields

ExchangeRate = Annotated[Decimal, Field(gt=0)]
ClaimedExchangeRate = Annotated[
    ExchangeRate | None,
    Field(
        description=(
            "Base-currency units per one unit of the submission's currency, as the requester"
            ' states it; null when the submission is in the base currency'
        )
    ),
]
ExpenseClaimType = Literal['expense_claim']
SupplierInvoiceType = Literal['supplier_invoice']
SubmissionType = Literal[ExpenseClaimType, SupplierInvoiceType]


class ExpenseClaim(ExpenseClaimFields):
    type: ExpenseClaimType
    claimed_exchange_rate: ClaimedExchangeRate
    business_purpose: str
    receipt_text: str | None


class SupplierInvoice(SupplierInvoiceFields):
    type: SupplierInvoiceType
    claimed_exchange_rate: ClaimedExchangeRate
    description: str


Submission = Annotated[ExpenseClaim | SupplierInvoice, Field(discriminator='type')]
