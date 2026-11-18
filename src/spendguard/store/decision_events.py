from typing import Any

from psycopg.types.json import Jsonb

from spendguard.db.database import ConnectionPool

TABLE_NAME = 'decision_events'


async def record_event_once(
    connection_pool: ConnectionPool, idempotency_key: str, payload: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    async with connection_pool.connection() as connection:
        inserted = await (
            await connection.execute(
                'insert into decision_events (idempotency_key, payload) values (%s, %s)'
                ' on conflict (idempotency_key) do nothing returning payload',
                (idempotency_key, Jsonb(payload)),
            )
        ).fetchone()
        if inserted is not None:
            return inserted['payload'], True
        existing = await (
            await connection.execute(
                'select payload from decision_events where idempotency_key = %s',
                (idempotency_key,),
            )
        ).fetchone()
    if existing is None:
        raise RuntimeError(f'decision event {idempotency_key} vanished after a conflicting insert')
    return existing['payload'], False


async def events_for(connection_pool: ConnectionPool, idempotency_key: str) -> list[dict[str, Any]]:
    async with connection_pool.connection() as connection:
        rows = await (
            await connection.execute(
                'select payload from decision_events where idempotency_key = %s',
                (idempotency_key,),
            )
        ).fetchall()
    return [row['payload'] for row in rows]
