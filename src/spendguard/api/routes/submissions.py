import uuid

from fastapi import APIRouter, HTTPException, status

router = APIRouter()


@router.get('/submissions')
async def list_submissions() -> None:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, 'not implemented')


@router.get('/submissions/{submission_id}')
async def read_submission(submission_id: uuid.UUID) -> None:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, 'not implemented')


@router.post('/submissions/{submission_id}/decision')
async def decide_submission(submission_id: uuid.UUID) -> None:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, 'not implemented')
