import uuid
from decimal import Decimal
from typing import Annotated, Any, Literal, Self, TypedDict

from fastapi import APIRouter, Depends, HTTPException, Query, status
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import Field, model_validator

from spendguard.api.dependencies import ApiDependencies, api_dependencies, authenticated_approver
from spendguard.graph.state import AuditEntry, Decision, DecisionOutcome, DispositionVerdict
from spendguard.queue.messages import QueueMessage
from spendguard.store import submissions
from spendguard.store.approvers import AuthenticatedApprover
from spendguard.store.submissions import SubmissionStatus
from spendguard.strict_model import StrictModel

router = APIRouter()

Approver = Annotated[AuthenticatedApprover, Depends(authenticated_approver)]
Dependencies = Annotated[ApiDependencies, Depends(api_dependencies)]


@router.get('/submissions')
async def list_submissions(
    approver: Approver,
    dependencies: Dependencies,
    status_filter: Annotated[Literal['pending_review'], Query(alias='status')],
) -> dict[str, list[dict[str, Any]]]:
    return {'submissions': await submissions.pending_reviews(dependencies.connection_pool)}


class SubmissionDetail(TypedDict):
    submission_id: str
    status: SubmissionStatus
    review: dict[str, Any] | None
    audit: list[AuditEntry]


@router.get('/submissions/{submission_id}')
async def read_submission(
    submission_id: uuid.UUID, approver: Approver, dependencies: Dependencies
) -> SubmissionDetail:
    record = await submissions.find_submission(dependencies.connection_pool, submission_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'no such submission')
    checkpoint = await AsyncPostgresSaver(dependencies.connection_pool).aget_tuple(
        {'configurable': {'thread_id': record.idempotency_key}}
    )
    audit: list[AuditEntry] = (
        [] if checkpoint is None else checkpoint.checkpoint['channel_values'].get('audit', [])
    )
    return {
        'submission_id': str(record.id),
        'status': record.status,
        'review': record.review,
        'audit': audit,
    }


class DispositionRequest(StrictModel):
    finding_id: str
    verdict: DispositionVerdict
    reason: str = Field(min_length=1)


class DecisionRequest(StrictModel):
    outcome: DecisionOutcome
    reason: str | None
    dispositions: list[DispositionRequest]

    @model_validator(mode='after')
    def _rejection_has_a_reason(self) -> Self:
        if self.outcome == 'reject' and not self.reason:
            raise ValueError('a rejection needs a reason')
        return self


class DecisionAccepted(TypedDict):
    submission_id: str
    status: SubmissionStatus


@router.post('/submissions/{submission_id}/decision', status_code=status.HTTP_202_ACCEPTED)
async def decide_submission(
    submission_id: uuid.UUID,
    decision_request: DecisionRequest,
    approver: Approver,
    dependencies: Dependencies,
) -> DecisionAccepted:
    record = await submissions.find_submission(dependencies.connection_pool, submission_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, 'no such submission')
    if record.review is None:
        raise HTTPException(status.HTTP_409_CONFLICT, 'submission is not awaiting a decision')
    if record.review['requester'] == approver.employee:
        raise HTTPException(status.HTTP_403_FORBIDDEN, 'approvers may not decide their own spend')
    if Decimal(record.review['base_currency_amount']) > approver.approval_limit_gbp:
        raise HTTPException(status.HTTP_403_FORBIDDEN, 'amount exceeds your approval limit')
    if sorted(disposition.finding_id for disposition in decision_request.dispositions) != sorted(
        finding['finding_id'] for finding in record.review['findings']
    ):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, 'give exactly one disposition per finding'
        )
    decision: Decision = {
        'outcome': decision_request.outcome,
        'decided_by': approver.employee,
        'reason': decision_request.reason,
        'dispositions': [
            {
                'finding_id': disposition.finding_id,
                'verdict': disposition.verdict,
                'reason': disposition.reason,
            }
            for disposition in decision_request.dispositions
        ],
    }
    if not await submissions.record_decision(
        dependencies.connection_pool, submission_id, decision, approver.employee
    ):
        raise HTTPException(status.HTTP_409_CONFLICT, 'submission is not awaiting a decision')
    dependencies.queue.send(
        QueueMessage(type='resume', submission_id=submission_id, thread_id=record.idempotency_key)
    )
    return {'submission_id': str(submission_id), 'status': 'decided'}
