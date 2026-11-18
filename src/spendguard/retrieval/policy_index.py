from langchain_core.embeddings import Embeddings
from psycopg import AsyncConnection
from psycopg.rows import DictRow

from spendguard.graph.state import RetrievedClause
from spendguard.policy.loader import Clause, Policy
from spendguard.store import policy_chunks
from spendguard.store.policy_chunks import PolicyChunk, StoredClause

CHUNK_CHARACTER_CAP = 1000
PARAGRAPH_SEPARATOR = '\n\n'


def clause_chunks(clause: Clause, policy_version: str) -> list[PolicyChunk]:
    return [
        PolicyChunk(
            clause_reference=clause.reference,
            chunk_index=chunk_index,
            heading_path=clause.heading_path,
            text=text,
            clause_hash=clause.hash,
            policy_version=policy_version,
        )
        for chunk_index, text in enumerate(_paragraph_groups(clause.text))
    ]


def _paragraph_groups(text: str) -> list[str]:
    groups: list[str] = []
    for paragraph in text.split(PARAGRAPH_SEPARATOR):
        if groups and len(groups[-1]) + len(PARAGRAPH_SEPARATOR) + len(paragraph) <= (
            CHUNK_CHARACTER_CAP
        ):
            groups[-1] = f'{groups[-1]}{PARAGRAPH_SEPARATOR}{paragraph}'
        else:
            groups.append(paragraph)
    return groups


def embedded_text(chunk: PolicyChunk) -> str:
    return f'{" > ".join(chunk.heading_path)}\n\n{chunk.text}'


async def index_policy(
    connection: AsyncConnection[DictRow],
    policy: Policy,
    embedder: Embeddings,
    embedding_model_id: str,
) -> None:
    stored = await policy_chunks.stored_clauses(connection)
    changed_clauses = [
        clause
        for clause in policy.clauses
        if stored.get(clause.reference)
        != StoredClause(clause_hash=clause.hash, embedding_model_id=embedding_model_id)
    ]
    current_references = {clause.reference for clause in policy.clauses}
    chunks = [
        chunk for clause in changed_clauses for chunk in clause_chunks(clause, policy.version)
    ]
    embeddings = (
        await embedder.aembed_documents([embedded_text(chunk) for chunk in chunks])
        if chunks
        else []
    )
    async with connection.transaction():
        await policy_chunks.delete_clauses(
            connection,
            [clause.reference for clause in changed_clauses]
            + [reference for reference in stored if reference not in current_references],
        )
        await policy_chunks.insert_chunks(connection, chunks, embeddings, embedding_model_id)
        await policy_chunks.set_policy_version(connection, policy.version)


async def retrieve_clauses(
    connection: AsyncConnection[DictRow],
    policy: Policy,
    embedder: Embeddings,
    queries: list[str],
    top_k: int,
) -> list[RetrievedClause]:
    clauses_by_reference = {clause.reference: clause for clause in policy.clauses}
    retrieved_references: list[str] = []
    for query in queries:
        embedding = await embedder.aembed_query(query)
        for reference in await policy_chunks.nearest_clause_references(
            connection, embedding, top_k
        ):
            if reference not in retrieved_references:
                retrieved_references.append(reference)
    return [supplied_clause(clauses_by_reference[reference]) for reference in retrieved_references]


def supplied_clause(clause: Clause) -> RetrievedClause:
    return {'clause_reference': clause.reference, 'clause_hash': clause.hash, 'text': clause.text}
