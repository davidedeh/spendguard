from dataclasses import dataclass
from decimal import Decimal

from spendguard.db.database import ConnectionPool

TABLE_NAME = 'approvers'


@dataclass(frozen=True)
class AuthenticatedApprover:
    employee: str
    approval_limit_gbp: Decimal


async def find_by_token_hash(
    connection_pool: ConnectionPool, token_hash: str
) -> AuthenticatedApprover | None:
    async with connection_pool.connection() as connection:
        row = await (
            await connection.execute(
                'select employee, approval_limit_gbp from approvers where token_hash = %s',
                (token_hash,),
            )
        ).fetchone()
    if row is None:
        return None
    return AuthenticatedApprover(
        employee=row['employee'], approval_limit_gbp=row['approval_limit_gbp']
    )
