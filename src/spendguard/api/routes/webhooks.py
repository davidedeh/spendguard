import hashlib
import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError

from spendguard.api.dependencies import ApiDependencies, api_dependencies, signed_body
from spendguard.queue.messages import QueueMessage
from spendguard.store import submissions
from spendguard.strict_model import StrictModel
from spendguard.submission import (
    ExpenseClaim,
    ExpenseClaimDelivery,
    SupplierInvoice,
    SupplierInvoiceDelivery,
)

router = APIRouter()


class Accepted(StrictModel):
    submission_id: str


@router.post('/webhooks/expense-claims', status_code=status.HTTP_202_ACCEPTED)
async def receive_expense_claim(
    request: Request, dependencies: Annotated[ApiDependencies, Depends(api_dependencies)]
) -> Accepted:
    delivery = await _signed_delivery(request, dependencies, ExpenseClaimDelivery)
    category = delivery.claim.category
    if category not in dependencies.policy.expense_categories:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            [
                {
                    'type': 'unknown_category',
                    'loc': ['claim', 'category'],
                    'msg': (
                        'Category must be one of the policy categories: '
                        + ', '.join(dependencies.policy.expense_categories)
                    ),
                    'input': category,
                }
            ],
        )
    return await _accept(
        dependencies,
        idempotency_key=delivery.idempotency_key,
        submission=delivery.claim,
        canonical_body=delivery.model_dump_json(),
    )


@router.post('/webhooks/supplier-invoices', status_code=status.HTTP_202_ACCEPTED)
async def receive_supplier_invoice(
    request: Request, dependencies: Annotated[ApiDependencies, Depends(api_dependencies)]
) -> Accepted:
    delivery = await _signed_delivery(request, dependencies, SupplierInvoiceDelivery)
    if dependencies.seed_data.invoice_requester(delivery.invoice) is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            [
                {
                    'type': 'no_requester',
                    'loc': ['invoice'],
                    'msg': (
                        'Give a known purchase_order, or a cost_centre with an owner,'
                        ' so the invoice has a requester'
                    ),
                    'input': {
                        'purchase_order': delivery.invoice.purchase_order,
                        'cost_centre': delivery.invoice.cost_centre,
                    },
                }
            ],
        )
    return await _accept(
        dependencies,
        idempotency_key=delivery.idempotency_key,
        submission=delivery.invoice,
        canonical_body=delivery.model_dump_json(),
    )


async def _signed_delivery[Delivery: StrictModel](
    request: Request, dependencies: ApiDependencies, delivery_model: type[Delivery]
) -> Delivery:
    settings = dependencies.settings
    raw_body = await signed_body(
        request, settings.webhook_signing_key, settings.signature_tolerance_seconds
    )
    try:
        return delivery_model.model_validate_json(raw_body)
    except ValidationError as error:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, json.loads(error.json(include_url=False))
        ) from error


async def _accept(
    dependencies: ApiDependencies,
    *,
    idempotency_key: str,
    submission: ExpenseClaim | SupplierInvoice,
    canonical_body: str,
) -> Accepted:
    recording = await submissions.record_submission(
        dependencies.connection_pool,
        idempotency_key=idempotency_key,
        submission_type=submission.type,
        payload=submission.model_dump(mode='json'),
        payload_hash=hashlib.sha256(canonical_body.encode()).hexdigest(),
    )
    if recording.outcome == 'conflicting':
        raise HTTPException(
            status.HTTP_409_CONFLICT, 'this idempotency key was accepted with a different body'
        )
    if recording.outcome == 'created':
        dependencies.queue.send(
            QueueMessage(
                type='start',
                submission_id=recording.submission_id,
                thread_id=idempotency_key,
            )
        )
    return Accepted(submission_id=str(recording.submission_id))
