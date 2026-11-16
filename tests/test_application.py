import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
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
from spendguard.security.signatures import sign
from spendguard.seed.loader import read_seed_directory
from spendguard.store import decision_events, submissions
from spendguard.submission import SupplierInvoiceDelivery
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


def unique_invoice_number() -> str:
    return f'INV-{uuid.uuid4().hex}'


def supplier_invoice_delivery(invoice_number: str, **invoice_overrides: Any) -> dict[str, Any]:
    return {
        'invoice': {
            'type': 'supplier_invoice',
            'supplier': 'SUP-001',
            'supplier_invoice_number': invoice_number,
            'purchase_order': 'PO-0001',
            'cost_centre': None,
            'issued_on': '2026-10-21',
            'amount': '85.00',
            'currency': 'GBP',
            'claimed_exchange_rate': None,
            'description': 'Printer toner cartridges for the Leeds office',
        }
        | invoice_overrides,
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

    async def post_claim(
        self, delivery: dict[str, Any], key: bytes = WEBHOOK_KEY, signed_at: int | None = None
    ) -> httpx.Response:
        return await self.post_signed('/webhooks/expense-claims', delivery, key, signed_at)

    async def post_invoice(self, delivery: dict[str, Any]) -> httpx.Response:
        return await self.post_signed('/webhooks/supplier-invoices', delivery, WEBHOOK_KEY, None)

    async def post_signed(
        self, path: str, delivery: dict[str, Any], key: bytes, signed_at: int | None
    ) -> httpx.Response:
        raw_body = json.dumps(delivery).encode()
        return await self.client.post(
            path, content=raw_body, headers=signed_headers(key, raw_body, signed_at)
        )

    async def run_worker(self) -> None:
        for message in self.queue.receive():
            await process_message(message, self.worker)

    async def status_of(self, submission_id: str) -> str:
        record = await submissions.find_submission(self.connection_pool, uuid.UUID(submission_id))
        assert record is not None
        return record.status

    async def requester_of(self, delivery: SupplierInvoiceDelivery) -> str:
        snapshot = await self.worker.graph.aget_state(
            {'configurable': {'thread_id': delivery.idempotency_key}}
        )
        return str(snapshot.values['requester'])

    async def recorded_events(self, submission_id: str) -> list[dict[str, Any]]:
        return await decision_events.events_for(self.connection_pool, submission_id)


@pytest.fixture
async def application(connection_pool: ConnectionPool) -> AsyncIterator[Application]:
    queue = InMemoryQueue()
    app = create_app(
        ApiDependencies(
            settings=SETTINGS,
            policy=POLICY,
            seed_data=SEED_DATA,
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
                url=SETTINGS.decision_event_url,
                key=b'test-decision-event-signing-key',
            )
        )
        graph = build_graph(
            GraphConfig(
                settings=SETTINGS,
                policy=POLICY,
                seed_data=SEED_DATA,
                chat_model=FakeChatModel(model_id='fake-chat'),
                embedder=build_embedder(SETTINGS),
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


async def test_a_clean_invoice_under_the_threshold_against_a_po_ends_complete_with_one_erp_event(
    application: Application,
) -> None:
    response = await application.post_invoice(supplier_invoice_delivery(unique_invoice_number()))
    assert response.status_code == 202
    submission_id = response.json()['submission_id']

    await application.run_worker()

    assert await application.status_of(submission_id) == 'complete'
    events = await application.recorded_events(submission_id)
    assert [(event['idempotency_key'], event['outcome']) for event in events] == [
        (submission_id, 'approved')
    ]


async def test_an_invoice_from_a_supplier_not_approved_waits_for_review_and_sends_no_event(
    application: Application,
) -> None:
    response = await application.post_invoice(
        supplier_invoice_delivery(
            unique_invoice_number(), supplier='SUP-002', purchase_order=None, cost_centre='CC-200'
        )
    )
    submission_id = response.json()['submission_id']

    await application.run_worker()

    assert await application.status_of(submission_id) == 'pending_review'
    assert await application.recorded_events(submission_id) == []


async def test_an_invoice_against_a_po_has_the_po_raiser_as_its_requester(
    application: Application,
) -> None:
    delivery = supplier_invoice_delivery(unique_invoice_number())
    await application.post_invoice(delivery)

    await application.run_worker()

    assert (
        await application.requester_of(SupplierInvoiceDelivery.model_validate(delivery))
        == 'EMP-003'
    )


async def test_an_invoice_without_a_po_has_the_cost_centre_owner_as_its_requester(
    application: Application,
) -> None:
    delivery = supplier_invoice_delivery(
        unique_invoice_number(), purchase_order=None, cost_centre='CC-200'
    )
    await application.post_invoice(delivery)

    await application.run_worker()

    assert (
        await application.requester_of(SupplierInvoiceDelivery.model_validate(delivery))
        == 'EMP-002'
    )


@pytest.mark.parametrize(
    'invoice_overrides',
    [
        {'purchase_order': 'PO-9999'},
        {'purchase_order': None, 'cost_centre': 'CC-300'},
        {'purchase_order': None, 'cost_centre': None},
    ],
    ids=['unknown purchase order', 'cost centre without an owner', 'neither'],
)
async def test_an_invoice_with_no_derivable_requester_is_rejected_with_422(
    application: Application, invoice_overrides: dict[str, Any]
) -> None:
    response = await application.post_invoice(
        supplier_invoice_delivery(unique_invoice_number(), **invoice_overrides)
    )

    assert response.status_code == 422
    [error] = response.json()['detail']
    assert error['type'] == 'no_requester'
    assert application.queue.receive() == []


async def test_an_identical_invoice_retry_returns_202_with_the_same_id_and_enqueues_nothing_more(
    application: Application,
) -> None:
    delivery = supplier_invoice_delivery(unique_invoice_number())
    first = await application.post_invoice(delivery)

    retry = await application.post_invoice(delivery)

    assert retry.status_code == 202
    assert retry.json()['submission_id'] == first.json()['submission_id']
    assert [message.type for message in application.queue.receive()] == ['start']


async def test_the_same_supplier_invoice_number_with_a_changed_body_returns_409(
    application: Application,
) -> None:
    invoice_number = unique_invoice_number()
    await application.post_invoice(supplier_invoice_delivery(invoice_number, amount='85.00'))

    changed = await application.post_invoice(
        supplier_invoice_delivery(invoice_number, amount='58.00')
    )

    assert changed.status_code == 409


async def test_the_same_invoice_number_from_another_supplier_is_a_separate_submission(
    application: Application,
) -> None:
    invoice_number = unique_invoice_number()
    first = await application.post_invoice(supplier_invoice_delivery(invoice_number))

    other = await application.post_invoice(
        supplier_invoice_delivery(invoice_number, supplier='SUP-002')
    )

    assert other.status_code == 202
    assert other.json()['submission_id'] != first.json()['submission_id']


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
        ('GET', '/submissions?status=pending_review'),
        ('GET', '/submissions/00000000-0000-0000-0000-000000000001'),
        ('POST', '/submissions/00000000-0000-0000-0000-000000000001/decision'),
    ],
)
async def test_every_endpoint_not_built_yet_returns_501(
    application: Application, method: str, path: str
) -> None:
    response = await application.client.request(method, path)

    assert response.status_code == 501
