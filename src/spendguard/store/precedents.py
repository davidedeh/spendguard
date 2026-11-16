import uuid
from dataclasses import dataclass

from spendguard.db.database import ConnectionPool
from spendguard.graph.state import SuppliedPrecedent
from spendguard.policy.loader import FindingCode, Policy

TABLE_NAME = 'precedents'
RECENT_PRECEDENT_LIMIT = 3


@dataclass(frozen=True)
class NewPrecedent:
    subject: str
    code: FindingCode
    clause_reference: str
    clause_hash: str
    reason: str
    approver: str
    submission_id: uuid.UUID


async def record_precedent(connection_pool: ConnectionPool, precedent: NewPrecedent) -> None:
    async with connection_pool.connection() as connection:
        await connection.execute(
            'insert into precedents (subject, code, clause_reference, clause_hash, reason,'
            ' approver, submission_id) values (%s, %s, %s, %s, %s, %s, %s)'
            ' on conflict (submission_id, code, clause_reference) do nothing',
            (
                precedent.subject,
                precedent.code,
                precedent.clause_reference,
                precedent.clause_hash,
                precedent.reason,
                precedent.approver,
                precedent.submission_id,
            ),
        )


async def recent_precedents(
    connection_pool: ConnectionPool, subject: str, code: FindingCode, policy: Policy
) -> list[SuppliedPrecedent]:
    async with connection_pool.connection() as connection:
        rows = await (
            await connection.execute(
                'select precedents.id, code, precedents.clause_reference, reason'
                ' from precedents'
                ' join unnest(%s::text[], %s::text[]) as current_clauses (clause_reference, hash)'
                ' on current_clauses.clause_reference = precedents.clause_reference'
                ' and current_clauses.hash = precedents.clause_hash'
                ' where subject = %s and code = %s'
                ' order by recorded_at desc, precedents.id limit %s',
                (
                    [clause.reference for clause in policy.clauses],
                    [clause.hash for clause in policy.clauses],
                    subject,
                    code,
                    RECENT_PRECEDENT_LIMIT,
                ),
            )
        ).fetchall()
    return [
        {
            'precedent_id': str(row['id']),
            'code': row['code'],
            'clause_reference': row['clause_reference'],
            'reason': row['reason'],
        }
        for row in rows
    ]


async def clear_precedents(connection_pool: ConnectionPool) -> None:
    async with connection_pool.connection() as connection:
        await connection.execute('delete from precedents')
