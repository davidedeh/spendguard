import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from psycopg import AsyncConnection
from psycopg.rows import DictRow

from settings_factory import make_test_settings
from spendguard.db.database import ConnectionPool
from spendguard.graph.config import GraphConfig
from spendguard.graph.nodes.execute import DecisionEvent
from spendguard.graph.nodes.plan import PlannedGuidance, RetrievalPlan, plan
from spendguard.graph.state import ReviewState
from spendguard.models.factory import build_embedder
from spendguard.models.fake import FakeChatModel, structured_response
from spendguard.policy.loader import FindingCode, load_policy
from spendguard.privacy.prompt_data import DATA_TAG_INSTRUCTION, untrusted_data
from spendguard.retrieval.policy_index import index_policy, retrieve_clauses
from spendguard.seed.loader import read_seed_directory
from spendguard.settings import ChatModelSettings, Settings

EMBEDDING_MODEL_ID = make_test_settings().embedding_model.model_id

FIXTURES = Path(__file__).parent / 'fixtures'
FIXTURE_POLICY = load_policy(FIXTURES / 'policy' / 'valid_policy.yaml')
SEED_DATA = read_seed_directory(FIXTURES / 'seed')
JUDGEMENT_CODES = FIXTURE_POLICY.codes_with_check('judgement')
CLAIM: dict[str, Any] = {
    'type': 'expense_claim',
    'claimant': 'EMP-003',
    'spent_on': '2026-10-10',
    'category': 'subsistence',
    'quantity': 1,
    'merchant': 'Harbour Kitchen',
    'amount': '38.00',
    'currency': 'GBP',
    'claimed_exchange_rate': None,
    'business_purpose': 'Saturday dinner with the Leeds team for Halden Group',
    'receipt_text': 'Harbour Kitchen\n1 x Fish pie 24.00\n1 x Pint of lager 14.00\nTOTAL 38.00',
}


class UnusedSender:
    async def send(self, event: DecisionEvent) -> None:
        raise AssertionError('planning and retrieval send no decision events')


def graph_config(
    chat_model: FakeChatModel, connection_pool: ConnectionPool, settings: Settings | None = None
) -> GraphConfig:
    configured_settings = settings or make_test_settings()
    return GraphConfig(
        settings=configured_settings,
        policy=FIXTURE_POLICY,
        seed_data=SEED_DATA,
        chat_model=chat_model,
        embedder=build_embedder(configured_settings),
        connection_pool=connection_pool,
        checkpointer=InMemorySaver(),
        decision_event_sender=UnusedSender(),
        precedent_memory=True,
        record_precedents=True,
        policy_context='retrieved_clauses',
    )


def planning_reply(queries: list[str], guided_codes: Sequence[FindingCode]) -> AIMessage:
    return structured_response(
        RetrievalPlan(
            retrieval_queries=queries,
            check_guidance=[
                PlannedGuidance(code=code, look_for=f'what {code} should look for')
                for code in guided_codes
            ],
        )
    )


def claim_state() -> ReviewState:
    return {
        'submission_id': 'submission-1',
        'idempotency_key': 'travel-desk:CLAIM-1',
        'submission_type': 'expense_claim',
        'submission': CLAIM,
        'redacted_submission': CLAIM,
    }


async def test_planning_returns_queries_and_guidance_for_every_judgement_check_and_selects_none(
    connection_pool: ConnectionPool,
) -> None:
    guided_codes: list[FindingCode] = [JUDGEMENT_CODES[1], 'MISSING_RECEIPT']
    model = FakeChatModel(
        model_id='fake-planner',
        responses=[planning_reply(['alcohol at dinner', 'weekend meal'], guided_codes)],
    )

    update = plan(claim_state(), graph_config(model, connection_pool))

    assert set(update) == {'retrieval_queries', 'check_guidance'}
    assert update['retrieval_queries'] == [
        {'query': 'alcohol at dinner'},
        {'query': 'weekend meal'},
    ]
    assert [guidance['code'] for guidance in update['check_guidance']] == list(JUDGEMENT_CODES)
    assert [guidance['look_for'] for guidance in update['check_guidance']] == [
        f'what {JUDGEMENT_CODES[1]} should look for' if code == JUDGEMENT_CODES[1] else ''
        for code in JUDGEMENT_CODES
    ]


async def test_the_planner_sees_only_the_redacted_submission_inside_data_tags(
    connection_pool: ConnectionPool,
) -> None:
    model = FakeChatModel(
        model_id='fake-planner',
        responses=[planning_reply(['alcohol at dinner', 'weekend meal'], JUDGEMENT_CODES)],
    )
    state = claim_state()
    state['submission'] = CLAIM | {'business_purpose': 'Team dinner, ask pat@example.com'}
    state['redacted_submission'] = CLAIM | {'business_purpose': 'Team dinner, ask [EMAIL]'}

    plan(state, graph_config(model, connection_pool))

    [[system_message, human_message]] = model.received
    assert DATA_TAG_INSTRUCTION in system_message.text
    assert human_message.text == untrusted_data(
        {
            field: value if isinstance(value, str) else json.dumps(value)
            for field, value in state['redacted_submission'].items()
            if value is not None
        }
    )


async def test_a_planner_set_to_json_schema_has_a_decorated_reply_read_as_its_plan(
    connection_pool: ConnectionPool,
) -> None:
    reply = {
        'retrieval_queries': ['alcohol at dinner', 'weekend meal', 'receipt total'],
        'check_guidance': [{'code': JUDGEMENT_CODES[0], 'look_for': 'a Saturday date'}],
    }
    model = FakeChatModel(
        model_id='fake-planner', responses=[AIMessage(content=f'**{json.dumps(reply)}')]
    )
    settings = make_test_settings(
        chat_models={
            'fake_primary': ChatModelSettings(
                provider='fake',
                model_id='fake-planner',
                structured_output_method='json_schema',
                temperature=0.0,
            )
        }
    )

    update = plan(claim_state(), graph_config(model, connection_pool, settings))

    assert [method for method, _ in model.structured_output_calls] == ['json_schema']
    assert [query['query'] for query in update['retrieval_queries']] == reply['retrieval_queries']
    assert update['check_guidance'][0] == {
        'code': JUDGEMENT_CODES[0],
        'look_for': 'a Saturday date',
    }


@pytest.fixture
async def connection(connection_pool: ConnectionPool) -> AsyncIterator[AsyncConnection[DictRow]]:
    async with (
        connection_pool.connection() as pooled_connection,
        pooled_connection.transaction(force_rollback=True),
    ):
        yield pooled_connection


def clause_reference(clause_id: str) -> str:
    return next(clause.reference for clause in FIXTURE_POLICY.clauses if clause.id == clause_id)


async def test_the_planned_queries_retrieve_the_fixture_clauses_they_describe(
    connection_pool: ConnectionPool, connection: AsyncConnection[DictRow]
) -> None:
    settings = make_test_settings(retrieval_top_k=1)
    model = FakeChatModel(
        model_id='fake-planner',
        responses=[
            planning_reply(
                ['is alcohol reimbursed', 'weekend or public holiday justification'],
                JUDGEMENT_CODES,
            )
        ],
    )
    config = graph_config(model, connection_pool, settings)
    await index_policy(connection, FIXTURE_POLICY, config.embedder, EMBEDDING_MODEL_ID)
    queries = plan(claim_state(), config)['retrieval_queries']

    retrieved = await retrieve_clauses(
        connection,
        FIXTURE_POLICY,
        config.embedder,
        [query['query'] for query in queries],
        settings.retrieval_top_k,
    )

    assert [clause['clause_reference'] for clause in retrieved] == [
        clause_reference('no_alcohol'),
        clause_reference('weekend_spend'),
    ]


async def test_each_query_takes_its_top_k_clauses_and_the_union_holds_each_clause_once(
    connection: AsyncConnection[DictRow],
) -> None:
    embedder = build_embedder(make_test_settings())
    await index_policy(connection, FIXTURE_POLICY, embedder, EMBEDDING_MODEL_ID)

    single = await retrieve_clauses(
        connection, FIXTURE_POLICY, embedder, ['is alcohol reimbursed'], 3
    )
    repeated = await retrieve_clauses(
        connection,
        FIXTURE_POLICY,
        embedder,
        ['is alcohol reimbursed', 'is alcohol reimbursed'],
        3,
    )

    assert len({clause['clause_reference'] for clause in single}) == 3
    assert repeated == single
