import uuid
from typing import Annotated, Any, Literal, TypedDict

from fastapi import APIRouter, Depends, HTTPException, Query, status
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from spendguard.api.dependencies import ApiDependencies, api_dependencies, authenticated_approver
from spendguard.graph.state import AuditEntry
from spendguard.store import submissions
from spendguard.store.approvers import AuthenticatedApprover
from spendguard.store.submissions import SubmissionStatus

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


@router.post('/submissions/{submission_id}/decision')
async def decide_submission(submission_id: uuid.UUID) -> None:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, 'not implemented')
