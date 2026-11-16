import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from psycopg.types.json import Jsonb

from spendguard.db.database import ConnectionPool
from spendguard.submission import SubmissionType

TABLE_NAME = 'submissions'

SubmissionStatus = Literal['received', 'pending_review', 'decided', 'complete']
RecordingOutcome = Literal['created', 'repeated', 'conflicting']


@dataclass(frozen=True)
class SubmissionRecord:
    id: uuid.UUID
    idempotency_key: str
    submission_type: SubmissionType
    payload: dict[str, Any]
    status: SubmissionStatus
    review: dict[str, Any] | None


@dataclass(frozen=True)
class Recording:
    submission_id: uuid.UUID
    outcome: RecordingOutcome


async def record_submission(
    connection_pool: ConnectionPool,
    *,
    idempotency_key: str,
    submission_type: SubmissionType,
    payload: dict[str, Any],
    payload_hash: str,
) -> Recording:
    async with connection_pool.connection() as connection:
        inserted = await (
            await connection.execute(
                'insert into submissions'
                ' (idempotency_key, submission_type, payload, payload_hash, status)'
                " values (%s, %s, %s, %s, 'received')"
                ' on conflict (idempotency_key) do nothing returning id',
                (idempotency_key, submission_type, Jsonb(payload), payload_hash),
            )
        ).fetchone()
        if inserted is not None:
            return Recording(submission_id=inserted['id'], outcome='created')
        existing = await (
            await connection.execute(
                'select id, payload_hash from submissions where idempotency_key = %s',
                (idempotency_key,),
            )
        ).fetchone()
    if existing is None:
        raise RuntimeError(f'submission {idempotency_key} vanished after a conflicting insert')
    return Recording(
        submission_id=existing['id'],
        outcome='repeated' if existing['payload_hash'] == payload_hash else 'conflicting',
    )


async def find_submission(
    connection_pool: ConnectionPool, submission_id: uuid.UUID
) -> SubmissionRecord | None:
    async with connection_pool.connection() as connection:
        row = await (
            await connection.execute(
                'select id, idempotency_key, submission_type, payload, status, review'
                ' from submissions where id = %s',
                (submission_id,),
            )
        ).fetchone()
    if row is None:
        return None
    return SubmissionRecord(
        id=row['id'],
        idempotency_key=row['idempotency_key'],
        submission_type=row['submission_type'],
        payload=row['payload'],
        status=row['status'],
        review=row['review'],
    )


async def move_status(
    connection_pool: ConnectionPool,
    submission_id: uuid.UUID,
    *,
    from_status: SubmissionStatus,
    to_status: SubmissionStatus,
) -> bool:
    async with connection_pool.connection() as connection:
        moved = await (
            await connection.execute(
                'update submissions set status = %s, status_changed_at = now()'
                ' where id = %s and status = %s returning id',
                (to_status, submission_id, from_status),
            )
        ).fetchone()
    return moved is not None


async def record_review(
    connection_pool: ConnectionPool, submission_id: uuid.UUID, review: Mapping[str, Any]
) -> None:
    async with connection_pool.connection() as connection:
        await connection.execute(
            'update submissions set review = %s where id = %s', (Jsonb(review), submission_id)
        )


async def pending_reviews(connection_pool: ConnectionPool) -> list[dict[str, Any]]:
    async with connection_pool.connection() as connection:
        rows = await (
            await connection.execute(
                "select review from submissions where status = 'pending_review'"
                ' order by status_changed_at, id'
            )
        ).fetchall()
    return [row['review'] for row in rows]
