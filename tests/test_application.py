import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from settings_factory import make_test_settings
from spendguard.api.app import create_app
from spendguard.api.dependencies import ApiDependencies
from spendguard.db.database import ConnectionPool
from spendguard.graph.build import build_graph
from spendguard.graph.config import GraphConfig
from spendguard.graph.nodes.execute import (
    DecisionEvent,
    DecisionEventSender,
    SignedDecisionEventSender,
)
from spendguard.graph.nodes.plan import RetrievalPlan
from spendguard.models.factory import build_embedder
from spendguard.models.fake import FakeChatModel, structured_response
from spendguard.policy.loader import POLICY_PATH, load_policy
from spendguard.queue.memory import InMemoryQueue
from spendguard.retrieval.policy_index import index_policy
from spendguard.security.approver_tokens import issue_token
from spendguard.security.signatures import sign
from spendguard.seed.loader import load_seed_data, read_seed_directory
from spendguard.settings import Settings
from spendguard.store import decision_events, submissions
from spendguard.worker import WorkerRuntime, process_message

SETTINGS = make_test_settings()
WEBHOOK_KEY = b'test-webhook-signing-key'
POLICY = load_policy(POLICY_PATH)
SEED_DATA = read_seed_directory(Path(__file__).parent / 'fixtures' / 'seed')


def unique_claim_id() -> str:
    return f'CLAIM-{uuid.uuid4().hex}'


def expense_claim_delivery(claim_id: str, **claim_overrides: Any) -> dict[str, Any]:
    return {
        'source': 'travel-desk',
        'claim_id': claim_id,
        'claim': {
            'type': 'expense_claim',
            'claimant': 'EMP-003',
            'spent_on': '2026-10-06',
            'category': 'subsistence',
            'quantity': 1,
            'merchant': 'Harbour Kitchen',
            'amount': '14.50',
            'currency': 'GBP',
            'claimed_exchange_rate': None,
            'business_purpose': 'Lunch while visiting the Leeds office for Halden Group',
            'receipt_text': 'Harbour Kitchen\n1 x Soup 6.50\n1 x Sandwich 8.00\nTOTAL 14.50',
        }
        | claim_overrides,
    }


def signed_headers(key: bytes, raw_body: bytes, signed_at: int | None = None) -> dict[str, str]:
    timestamp = int(time.time()) if signed_at is None else signed_at
    return {'X-Signature': sign(key, timestamp, raw_body), 'Content-Type': 'application/json'}


@dataclass
class RecordingSender:
    inner: DecisionEventSender
    sent: list[DecisionEvent] = field(default_factory=list)

    async def send(self, event: DecisionEvent) -> None:
        self.sent.append(event)
        await self.inner.send(event)


@dataclass
class Application:
    client: httpx.AsyncClient
    queue: InMemoryQueue
    worker: WorkerRuntime
    connection_pool: ConnectionPool
    sender: RecordingSender
    branch_database: str

    def issue_token(self, approver: str) -> str:
        with psycopg.connect(self.branch_database) as connection:
            return issue_token(connection, approver)

    async def pending_review(self, token: str) -> httpx.Response:
        return await self.client.get(
            '/submissions',
            params={'status': 'pending_review'},
            headers={'Authorization': f'Bearer {token}'},
        )

    async def post_claim(
        self, delivery: dict[str, Any], key: bytes = WEBHOOK_KEY, signed_at: int | None = None
    ) -> httpx.Response:
        raw_body = json.dumps(delivery).encode()
        return await self.client.post(
            '/webhooks/expense-claims',
            content=raw_body,
            headers=signed_headers(key, raw_body, signed_at),
        )

    async def run_worker(self) -> None:
        for message in self.queue.receive():
            await process_message(message, self.worker)

    async def status_of(self, submission_id: str) -> str:
        record = await submissions.find_submission(self.connection_pool, uuid.UUID(submission_id))
        assert record is not None
        return record.status

    async def recorded_events(self, submission_id: str) -> list[dict[str, Any]]:
        return await decision_events.events_for(self.connection_pool, submission_id)

    async def review_state(self, submission_id: str) -> dict[str, Any]:
        record = await submissions.find_submission(self.connection_pool, uuid.UUID(submission_id))
        assert record is not None
        snapshot = await self.worker.graph.aget_state(
            {'configurable': {'thread_id': record.idempotency_key}}
        )
        state: dict[str, Any] = snapshot.values
        return state


def clause_reference(clause_id: str) -> str:
    return next(clause.reference for clause in POLICY.clauses if clause.id == clause_id)


class PlanningChatModel(FakeChatModel):
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        reply = structured_response(
            RetrievalPlan(
                retrieval_queries=['is alcohol reimbursed with a meal', 'receipts for a lunch'],
                check_guidance=[],
            )
        )
        return ChatResult(generations=[ChatGeneration(message=reply)])


@pytest.fixture
def application_settings() -> Settings:
    return SETTINGS


@pytest.fixture
async def application(
    connection_pool: ConnectionPool, branch_database: str, application_settings: Settings
) -> AsyncIterator[Application]:
    with psycopg.connect(branch_database) as connection:
        load_seed_data(connection, SEED_DATA)
    queue = InMemoryQueue()
    app = create_app(
        ApiDependencies(
            settings=application_settings,
            policy=POLICY,
            connection_pool=connection_pool,
            queue=queue,
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://testserver'
    ) as client:
        sender = RecordingSender(
            SignedDecisionEventSender(
                client=client,
                url=application_settings.decision_event_url,
                key=b'test-decision-event-signing-key',
            )
        )
        embedder = build_embedder(application_settings)
        async with connection_pool.connection() as connection:
            await index_policy(connection, POLICY, embedder)
        graph = build_graph(
            GraphConfig(
                settings=application_settings,
                policy=POLICY,
                seed_data=SEED_DATA,
                chat_model=PlanningChatModel(model_id='fake-chat'),
                embedder=embedder,
                connection_pool=connection_pool,
                checkpointer=AsyncPostgresSaver(connection_pool),
                decision_event_sender=sender,
            )
        )
        yield Application(
            client=client,
            queue=queue,
            worker=WorkerRuntime(graph=graph, connection_pool=connection_pool),
            connection_pool=connection_pool,
            sender=sender,
            branch_database=branch_database,
        )


async def test_health_returns_200(application: Application) -> None:
    response = await application.client.get('/health')

    assert response.status_code == 200


async def test_a_claim_signed_with_the_wrong_key_is_rejected_with_401(
    application: Application,
) -> None:
    response = await application.post_claim(
        expense_claim_delivery(unique_claim_id()), key=b'another-key'
    )

    assert response.status_code == 401


async def test_a_claim_signed_more_than_the_tolerance_ago_is_rejected_with_401(
    application: Application,
) -> None:
    response = await application.post_claim(
        expense_claim_delivery(unique_claim_id()), signed_at=int(time.time()) - 301
    )

    assert response.status_code == 401


async def test_a_signed_claim_with_a_receipt_under_the_threshold_ends_complete_with_one_erp_event(
    application: Application,
) -> None:
    response = await application.post_claim(expense_claim_delivery(unique_claim_id()))
    assert response.status_code == 202
    submission_id = response.json()['submission_id']

    await application.run_worker()

    assert await application.status_of(submission_id) == 'complete'
    events = await application.recorded_events(submission_id)
    assert [(event['idempotency_key'], event['outcome']) for event in events] == [
        (submission_id, 'approved')
    ]


async def test_a_claim_missing_a_required_receipt_waits_for_review_and_sends_no_event(
    application: Application,
) -> None:
    response = await application.post_claim(
        expense_claim_delivery(unique_claim_id(), amount='30.00', receipt_text=None)
    )
    submission_id = response.json()['submission_id']

    await application.run_worker()

    assert await application.status_of(submission_id) == 'pending_review'
    assert await application.recorded_events(submission_id) == []


async def test_the_clauses_retrieved_for_a_submission_are_recorded_in_its_state_and_audit(
    application: Application,
) -> None:
    response = await application.post_claim(expense_claim_delivery(unique_claim_id()))
    submission_id = response.json()['submission_id']

    await application.run_worker()

    state = await application.review_state(submission_id)
    retrieved_references = [clause['clause_reference'] for clause in state['retrieved_clauses']]
    assert clause_reference('no_alcohol') in retrieved_references
    assert len(retrieved_references) == len(set(retrieved_references)) > 1
    assert {
        'node': 'retrieve_policy',
        'event': 'policy_retrieved',
        'detail': {'clause_references': retrieved_references},
    } in state['audit']


async def test_a_receipted_claim_repeating_an_earlier_claim_waits_for_review_and_sends_no_event(
    application: Application,
) -> None:
    response = await application.post_claim(
        expense_claim_delivery(
            unique_claim_id(),
            spent_on='2026-09-14',
            amount='42.50',
            receipt_text='Harbour Kitchen\n1 x Set menu 42.50\nTOTAL 42.50',
        )
    )
    submission_id = response.json()['submission_id']

    await application.run_worker()

    assert await application.status_of(submission_id) == 'pending_review'
    assert await application.recorded_events(submission_id) == []


async def test_a_claim_whose_category_is_not_a_policy_category_is_rejected_with_422(
    application: Application,
) -> None:
    response = await application.post_claim(
        expense_claim_delivery(unique_claim_id(), category='spa_treatments')
    )

    assert response.status_code == 422
    [error] = response.json()['detail']
    assert (error['loc'], error['input']) == (['claim', 'category'], 'spa_treatments')
    assert application.queue.receive() == []


async def test_an_identical_retry_returns_202_with_the_same_id_and_enqueues_nothing_more(
    application: Application,
) -> None:
    delivery = expense_claim_delivery(unique_claim_id())
    first = await application.post_claim(delivery)

    retry = await application.post_claim(delivery)

    assert retry.status_code == 202
    assert retry.json()['submission_id'] == first.json()['submission_id']
    assert [message.type for message in application.queue.receive()] == ['start']


async def test_the_same_claim_id_with_a_changed_body_returns_409(
    application: Application,
) -> None:
    claim_id = unique_claim_id()
    await application.post_claim(expense_claim_delivery(claim_id, amount='14.50'))

    changed = await application.post_claim(expense_claim_delivery(claim_id, amount='41.50'))

    assert changed.status_code == 409


async def test_a_duplicate_start_message_is_a_no_op(application: Application) -> None:
    response = await application.post_claim(expense_claim_delivery(unique_claim_id()))
    [start] = application.queue.receive()
    await process_message(start, application.worker)

    await process_message(start, application.worker)

    assert [event.submission_id for event in application.sender.sent] == [
        response.json()['submission_id']
    ]


async def test_the_mock_erp_stores_a_repeated_event_once(application: Application) -> None:
    submission_id = str(uuid.uuid4())
    event = DecisionEvent(
        idempotency_key=submission_id,
        submission_id=submission_id,
        outcome='rejected',
        decided_by='EMP-001',
        reason='Not a business expense',
    )

    await application.sender.inner.send(event)
    await application.sender.inner.send(event)

    assert await application.recorded_events(submission_id) == [event.model_dump(mode='json')]


async def test_the_mock_erp_rejects_an_event_signed_with_the_webhook_key(
    application: Application,
) -> None:
    submission_id = str(uuid.uuid4())
    raw_body = (
        DecisionEvent(
            idempotency_key=submission_id,
            submission_id=submission_id,
            outcome='approved',
            decided_by=None,
            reason=None,
        )
        .model_dump_json()
        .encode()
    )

    response = await application.client.post(
        '/mock-erp/decision-events',
        content=raw_body,
        headers=signed_headers(WEBHOOK_KEY, raw_body),
    )

    assert response.status_code == 401
    assert await application.recorded_events(submission_id) == []


@pytest.mark.parametrize(
    ('method', 'path'),
    [
        ('POST', '/webhooks/supplier-invoices'),
        ('POST', '/submissions/00000000-0000-0000-0000-000000000001/decision'),
    ],
)
async def test_every_endpoint_not_built_yet_returns_501(
    application: Application, method: str, path: str
) -> None:
    response = await application.client.request(method, path)

    assert response.status_code == 501


@pytest.mark.parametrize(
    'path',
    ['/submissions?status=pending_review', '/submissions/00000000-0000-0000-0000-000000000001'],
)
@pytest.mark.parametrize(
    'headers',
    [{}, {'Authorization': 'Bearer not-an-issued-token'}],
    ids=['no token', 'wrong token'],
)
async def test_reading_submissions_without_a_valid_token_returns_401(
    application: Application, path: str, headers: dict[str, str]
) -> None:
    response = await application.client.get(path, headers=headers)

    assert response.status_code == 401


async def test_a_claim_with_a_finding_is_listed_for_review_with_its_templated_payload(
    application: Application,
) -> None:
    response = await application.post_claim(
        expense_claim_delivery(unique_claim_id(), amount='30.00', receipt_text=None)
    )
    submission_id = response.json()['submission_id']
    await application.run_worker()

    listed = await application.pending_review(application.issue_token('EMP-001'))

    assert listed.status_code == 200
    [review] = [
        item for item in listed.json()['submissions'] if item['submission_id'] == submission_id
    ]
    assert review == {
        'submission_id': submission_id,
        'submission_type': 'expense_claim',
        'requester': 'EMP-003',
        'base_currency_amount': '30.00',
        'recommendation': 'none',
        'findings': [
            {
                'finding_id': 'MISSING_RECEIPT',
                'code': 'MISSING_RECEIPT',
                'severity': 'medium',
                'source': 'rule',
                'clause': {
                    'reference': '§3.1',
                    'heading_path': ['Receipts and evidence', 'Receipts'],
                    'text': POLICY.finding_codes['MISSING_RECEIPT'].clause.text,
                },
                'quotes': [],
                'precedents': [],
            }
        ],
        'clearances_by_precedent': [],
    }


async def test_reading_an_auto_approved_claim_shows_its_recommendation_status_and_audit(
    application: Application,
) -> None:
    response = await application.post_claim(expense_claim_delivery(unique_claim_id()))
    submission_id = response.json()['submission_id']
    await application.run_worker()

    read = await application.client.get(
        f'/submissions/{submission_id}',
        headers={'Authorization': f'Bearer {application.issue_token("EMP-002")}'},
    )

    assert read.status_code == 200
    body = read.json()
    assert body['status'] == 'complete'
    assert body['review']['recommendation'] == 'approve'
    assert body['review']['findings'] == []
    state = await application.review_state(submission_id)
    assert body['audit'] == [
        {
            'node': 'retrieve_policy',
            'event': 'policy_retrieved',
            'detail': {
                'clause_references': [
                    clause['clause_reference'] for clause in state['retrieved_clauses']
                ]
            },
        },
        {'node': 'execute', 'event': 'decision_event_sent', 'detail': {'outcome': 'approved'}},
        {'node': 'finalize', 'event': 'outcome_recorded', 'detail': {'outcome': 'approve'}},
    ]


@pytest.mark.parametrize(
    'application_settings', [make_test_settings(force_human_review=True)], ids=['kill switch on']
)
async def test_with_the_kill_switch_on_a_clean_small_claim_waits_for_review(
    application: Application,
) -> None:
    response = await application.post_claim(expense_claim_delivery(unique_claim_id()))
    submission_id = response.json()['submission_id']

    await application.run_worker()

    assert await application.status_of(submission_id) == 'pending_review'
    assert await application.recorded_events(submission_id) == []
    listed = await application.pending_review(application.issue_token('EMP-001'))
    [review] = [
        item for item in listed.json()['submissions'] if item['submission_id'] == submission_id
    ]
    assert (review['recommendation'], review['findings']) == ('approve', [])
