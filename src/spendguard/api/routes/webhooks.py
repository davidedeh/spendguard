import hashlib
import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError

from spendguard.api.dependencies import ApiDependencies, api_dependencies, signed_body
from spendguard.queue.messages import QueueMessage
from spendguard.store import submissions
from spendguard.strict_model import StrictModel
from spendguard.submission import ExpenseClaimDelivery

router = APIRouter()


class Accepted(StrictModel):
    submission_id: str


@router.post('/webhooks/expense-claims', status_code=status.HTTP_202_ACCEPTED)
async def receive_expense_claim(
    request: Request, dependencies: Annotated[ApiDependencies, Depends(api_dependencies)]
) -> Accepted:
    settings = dependencies.settings
    raw_body = await signed_body(
        request, settings.webhook_signing_key, settings.signature_tolerance_seconds
    )
    try:
        delivery = ExpenseClaimDelivery.model_validate_json(raw_body)
    except ValidationError as error:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, json.loads(error.json(include_url=False))
        ) from error
    canonical_body = delivery.model_dump_json()
    recording = await submissions.record_submission(
        dependencies.connection_pool,
        idempotency_key=delivery.idempotency_key,
        submission_type='expense_claim',
        payload=delivery.claim.model_dump(mode='json'),
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
                thread_id=delivery.idempotency_key,
            )
        )
    return Accepted(submission_id=str(recording.submission_id))


@router.post('/webhooks/supplier-invoices')
async def receive_supplier_invoice() -> None:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, 'not implemented')
