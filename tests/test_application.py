import asyncio
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
from spendguard.models.factory import build_embedder
from spendguard.models.fake import FakeChatModel
from spendguard.policy.loader import POLICY_PATH, load_policy
from spendguard.queue.memory import InMemoryQueue
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

    async def decide(
        self, submission_id: str, token: str, decision: dict[str, Any]
    ) -> httpx.Response:
        return await self.client.post(
            f'/submissions/{submission_id}/decision',
            json=decision,
            headers={'Authorization': f'Bearer {token}'},
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
        graph = build_graph(
            GraphConfig(
                settings=application_settings,
                policy=POLICY,
                seed_data=SEED_DATA,
                chat_model=FakeChatModel(model_id='fake-chat'),
                embedder=build_embedder(application_settings),
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
    assert body['audit'] == [
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


async def claim_waiting_for_review(application: Application, **claim_overrides: Any) -> str:
    response = await application.post_claim(
        expense_claim_delivery(
            unique_claim_id(), **({'amount': '30.00', 'receipt_text': None} | claim_overrides)
        )
    )
    submission_id: str = response.json()['submission_id']
    await application.run_worker()
    assert await application.status_of(submission_id) == 'pending_review'
    return submission_id


def missing_receipt_decision(outcome: str, reason: str | None) -> dict[str, Any]:
    return {
        'outcome': outcome,
        'reason': reason,
        'dispositions': [
            {'finding_id': 'MISSING_RECEIPT', 'verdict': 'upheld', 'reason': None},
        ],
    }


async def test_an_approved_claim_ends_complete_with_one_approved_erp_event(
    application: Application,
) -> None:
    submission_id = await claim_waiting_for_review(application)

    decided = await application.decide(
        submission_id,
        application.issue_token('EMP-002'),
        missing_receipt_decision('approve', None),
    )
    await application.run_worker()

    assert decided.status_code == 202
    assert await application.status_of(submission_id) == 'complete'
    events = await application.recorded_events(submission_id)
    assert [(event['outcome'], event['decided_by']) for event in events] == [
        ('approved', 'EMP-002')
    ]


async def test_a_rejected_claim_ends_complete_with_one_erp_event_carrying_the_stored_reason(
    application: Application,
) -> None:
    submission_id = await claim_waiting_for_review(application)

    await application.decide(
        submission_id,
        application.issue_token('EMP-001'),
        missing_receipt_decision('reject', 'No receipt for a 30 pound lunch'),
    )
    await application.run_worker()

    assert await application.status_of(submission_id) == 'complete'
    events = await application.recorded_events(submission_id)
    assert [(event['outcome'], event['decided_by'], event['reason']) for event in events] == [
        ('rejected', 'EMP-001', 'No receipt for a 30 pound lunch')
    ]


@pytest.mark.parametrize(
    'claim_overrides',
    [{'claimant': 'EMP-002'}, {'amount': '5000.01'}],
    ids=['the approver is the requester', 'the amount is over the approval limit'],
)
async def test_a_decision_the_approver_may_not_make_returns_403_and_changes_nothing(
    application: Application, claim_overrides: dict[str, Any]
) -> None:
    submission_id = await claim_waiting_for_review(application, **claim_overrides)

    decided = await application.decide(
        submission_id,
        application.issue_token('EMP-002'),
        missing_receipt_decision('approve', None),
    )

    assert decided.status_code == 403
    assert await application.status_of(submission_id) == 'pending_review'
    assert application.queue.receive() == []


@pytest.mark.parametrize(
    'dispositions',
    [
        [],
        [{'finding_id': 'MISSING_PO', 'verdict': 'upheld', 'reason': None}],
        [
            {'finding_id': 'MISSING_RECEIPT', 'verdict': 'upheld', 'reason': None},
            {'finding_id': 'MISSING_RECEIPT', 'verdict': 'dismissed', 'reason': 'Lost in transit'},
        ],
        [{'finding_id': 'MISSING_RECEIPT', 'verdict': 'dismissed', 'reason': None}],
    ],
    ids=[
        'no dispositions',
        'a disposition for another finding',
        'two dispositions for one finding',
        'a dismissal without a reason',
    ],
)
async def test_a_decision_without_one_disposition_per_finding_returns_422_and_changes_nothing(
    application: Application, dispositions: list[dict[str, Any]]
) -> None:
    submission_id = await claim_waiting_for_review(application)

    decided = await application.decide(
        submission_id,
        application.issue_token('EMP-001'),
        {'outcome': 'approve', 'reason': None, 'dispositions': dispositions},
    )

    assert decided.status_code == 422
    assert await application.status_of(submission_id) == 'pending_review'
    assert application.queue.receive() == []


async def decisions_waiting_on_a_row_lock(branch_database: str) -> int:
    async with await psycopg.AsyncConnection.connect(branch_database, autocommit=True) as observer:
        row = await (
            await observer.execute(
                "select count(*) from pg_stat_activity where wait_event_type = 'Lock'"
                " and query like 'update submissions set status = ''decided''%'"
            )
        ).fetchone()
    assert row is not None
    count: int = row[0]
    return count


async def test_of_two_concurrent_decisions_exactly_one_wins_and_the_other_gets_409(
    application: Application,
) -> None:
    submission_id = await claim_waiting_for_review(application)
    approvers = ['EMP-001', 'EMP-002']
    tokens = [application.issue_token(approver) for approver in approvers]
    async with await psycopg.AsyncConnection.connect(application.branch_database) as blocker:
        async with blocker.transaction():
            await blocker.execute(
                'select id from submissions where id = %s for update', (submission_id,)
            )
            racing = [
                asyncio.create_task(
                    application.decide(
                        submission_id, token, missing_receipt_decision('approve', None)
                    )
                )
                for token in tokens
            ]
            async with asyncio.timeout(5):
                while (await decisions_waiting_on_a_row_lock(application.branch_database)) < len(
                    racing
                ):
                    await asyncio.sleep(0.01)
        responses = await asyncio.gather(*racing)

    assert sorted(response.status_code for response in responses) == [202, 409]
    [winner] = [
        approver
        for approver, response in zip(approvers, responses, strict=True)
        if response.status_code == 202
    ]
    record = await submissions.find_submission(
        application.connection_pool, uuid.UUID(submission_id)
    )
    assert record is not None
    assert (record.status, record.decision and record.decision['decided_by']) == (
        'decided',
        winner,
    )
    assert [message.type for message in application.queue.receive()] == ['resume']
