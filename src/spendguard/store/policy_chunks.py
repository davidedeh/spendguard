from collections.abc import Sequence
from dataclasses import dataclass

from psycopg import AsyncConnection
from psycopg.rows import DictRow

TABLE_NAME = 'policy_chunks'


@dataclass(frozen=True)
class PolicyChunk:
    clause_reference: str
    chunk_index: int
    heading_path: tuple[str, ...]
    text: str
    clause_hash: str
    policy_version: str


@dataclass(frozen=True)
class StoredClause:
    clause_hash: str
    embedding_model_id: str


async def stored_clauses(connection: AsyncConnection[DictRow]) -> dict[str, StoredClause]:
    rows = await (
        await connection.execute(
            'select distinct clause_reference, clause_hash, embedding_model_id from policy_chunks'
        )
    ).fetchall()
    return {
        row['clause_reference']: StoredClause(
            clause_hash=row['clause_hash'], embedding_model_id=row['embedding_model_id']
        )
        for row in rows
    }


async def delete_clauses(
    connection: AsyncConnection[DictRow], clause_references: Sequence[str]
) -> None:
    await connection.execute(
        'delete from policy_chunks where clause_reference = any(%s)', (list(clause_references),)
    )


async def insert_chunks(
    connection: AsyncConnection[DictRow],
    chunks: Sequence[PolicyChunk],
    embeddings: Sequence[Sequence[float]],
    embedding_model_id: str,
) -> None:
    async with connection.cursor() as cursor:
        await cursor.executemany(
            'insert into policy_chunks (clause_reference, chunk_index, heading_path, text,'
            ' clause_hash, policy_version, embedding, embedding_model_id)'
            ' values (%s, %s, %s, %s, %s, %s, %s::vector, %s)',
            [
                (
                    chunk.clause_reference,
                    chunk.chunk_index,
                    list(chunk.heading_path),
                    chunk.text,
                    chunk.clause_hash,
                    chunk.policy_version,
                    _vector_literal(embedding),
                    embedding_model_id,
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ],
        )


async def set_policy_version(connection: AsyncConnection[DictRow], policy_version: str) -> None:
    await connection.execute('update policy_chunks set policy_version = %s', (policy_version,))


async def nearest_clause_references(
    connection: AsyncConnection[DictRow], embedding: Sequence[float], limit: int
) -> list[str]:
    rows = await (
        await connection.execute(
            'select clause_reference from policy_chunks order by embedding <=> %s::vector limit %s',
            (_vector_literal(embedding), limit),
        )
    ).fetchall()
    return [row['clause_reference'] for row in rows]


def _vector_literal(embedding: Sequence[float]) -> str:
    return '[' + ','.join(repr(float(value)) for value in embedding) + ']'
