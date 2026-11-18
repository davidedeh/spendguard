import dataclasses
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from psycopg import AsyncConnection
from psycopg.rows import DictRow

from settings_factory import make_test_settings
from spendguard.db.database import ConnectionPool
from spendguard.models.factory import build_embedder
from spendguard.policy.loader import Policy, load_policy
from spendguard.retrieval.policy_index import CHUNK_CHARACTER_CAP, index_policy

EMBEDDING_MODEL_ID = 'fake-embedding-model'
FIXTURE_POLICY = load_policy(Path(__file__).parent / 'fixtures' / 'policy' / 'valid_policy.yaml')


class CountingEmbedder(Embeddings):
    def __init__(self) -> None:
        self.inner = build_embedder(make_test_settings())
        self.embedded: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)


@pytest.fixture
async def connection(connection_pool: ConnectionPool) -> AsyncIterator[AsyncConnection[DictRow]]:
    async with (
        connection_pool.connection() as pooled_connection,
        pooled_connection.transaction(force_rollback=True),
    ):
        yield pooled_connection


async def test_re_indexing_an_unchanged_policy_embeds_nothing(
    connection: AsyncConnection[DictRow],
) -> None:
    embedder = CountingEmbedder()
    await index_policy(connection, FIXTURE_POLICY, embedder, EMBEDDING_MODEL_ID)
    first_run_embeddings = len(embedder.embedded)

    await index_policy(connection, FIXTURE_POLICY, embedder, EMBEDDING_MODEL_ID)

    assert first_run_embeddings == len(FIXTURE_POLICY.clauses)
    assert len(embedder.embedded) == first_run_embeddings


def with_clause_text(policy: Policy, clause_index: int, text: str) -> Policy:
    clauses = list(policy.clauses)
    clauses[clause_index] = dataclasses.replace(
        clauses[clause_index], text=text, hash=f'hash of {text}'
    )
    return dataclasses.replace(policy, clauses=tuple(clauses))


async def test_changing_one_clause_re_embeds_only_that_clause(
    connection: AsyncConnection[DictRow],
) -> None:
    await index_policy(connection, FIXTURE_POLICY, CountingEmbedder(), EMBEDDING_MODEL_ID)
    embedder = CountingEmbedder()
    changed_policy = with_clause_text(FIXTURE_POLICY, 4, 'Alcohol is never reimbursed.')

    await index_policy(connection, changed_policy, embedder, EMBEDDING_MODEL_ID)

    assert embedder.embedded == ['Meals\n\nAlcohol is never reimbursed.']


async def test_a_clause_longer_than_the_cap_is_embedded_one_paragraph_group_per_chunk(
    connection: AsyncConnection[DictRow],
) -> None:
    first_paragraph = 'Spend ' + 'must be justified. ' * (CHUNK_CHARACTER_CAP // 25)
    second_paragraph = 'Approvers ' + 'record a reason. ' * (CHUNK_CHARACTER_CAP // 25)
    third_paragraph = 'Short closing note.'
    long_policy = with_clause_text(
        FIXTURE_POLICY, 4, f'{first_paragraph}\n\n{second_paragraph}\n\n{third_paragraph}'
    )
    embedder = CountingEmbedder()

    await index_policy(connection, long_policy, embedder, EMBEDDING_MODEL_ID)

    assert [text for text in embedder.embedded if text.startswith('Meals\n\n')] == [
        f'Meals\n\n{first_paragraph}',
        f'Meals\n\n{second_paragraph}\n\n{third_paragraph}',
    ]


async def test_indexing_with_another_embedding_model_re_embeds_every_clause(
    connection: AsyncConnection[DictRow],
) -> None:
    await index_policy(connection, FIXTURE_POLICY, CountingEmbedder(), EMBEDDING_MODEL_ID)
    embedder = CountingEmbedder()

    await index_policy(connection, FIXTURE_POLICY, embedder, 'another-embedding-model')
    await index_policy(connection, FIXTURE_POLICY, embedder, 'another-embedding-model')

    assert len(embedder.embedded) == len(FIXTURE_POLICY.clauses)
