import asyncio
import datetime
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import psycopg
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import Field

from settings_factory import make_test_settings
from spendguard.api.app import create_app
from spendguard.api.dependencies import ApiDependencies
from spendguard.db.database import ConnectionPool
from spendguard.deployment.smoke import smoke_failures
from spendguard.graph.build import build_graph
from spendguard.graph.config import GraphConfig
from spendguard.graph.nodes.execute import (
    DecisionEvent,
    DecisionEventSender,
    SignedDecisionEventSender,
)
from spendguard.graph.nodes.plan import RetrievalPlan
from spendguard.graph.nodes.run_judgement_checks import CheckVerdict, JudgementReport, QuotedText
from spendguard.models.factory import build_embedder
from spendguard.models.fake import FakeChatModel, structured_response
from spendguard.policy.loader import POLICY_PATH, FindingCode, load_policy
from spendguard.queue.memory import InMemoryQueue
from spendguard.queue.messages import QueueMessage
from spendguard.queue.sqs import SqsQueue, sqs_client
from spendguard.retrieval.policy_index import index_policy
from spendguard.security.approver_tokens import issue_token
from spendguard.security.signatures import sign
from spendguard.seed.loader import load_seed_data, read_seed_directory
from spendguard.settings import Settings
from spendguard.store import decision_events, submissions
from spendguard.submission import SupplierInvoiceDelivery
from spendguard.sweeper import create_sweeper_app, sweep
from spendguard.worker import WorkerRuntime, create_worker_app, handle_sqs_event, process_message

if TYPE_CHECKING:
    from mypy_boto3_sqs.client import SQSClient
    from mypy_boto3_sqs.type_defs import ReceiveMessageResultTypeDef

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
    branch_database: str
    chat_model: 'PlanningChatModel'

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

    async def decide(
        self, submission_id: str, token: str, decision: dict[str, Any]
    ) -> httpx.Response:
        return await self.client.post(
            f'/submissions/{submission_id}/decision',
            json=decision,
            headers={'Authorization': f'Bearer {token}'},
        )

    async def run_worker(self) -> None:
        for delivery in self.queue.receive():
            await process_message(delivery.message, self.worker)
            self.queue.acknowledge(delivery.receipt_handle)

    async def run_worker_for(self, submission_id: str) -> None:
        for delivery in self.queue.receive():
            if str(delivery.message.submission_id) == submission_id:
                await process_message(delivery.message, self.worker)
            self.queue.acknowledge(delivery.receipt_handle)

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
    verdicts: list[CheckVerdict] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        bound_tool_names = [tool['function']['name'] for tool in kwargs.get('tools', [])]
        if bound_tool_names == [RetrievalPlan.__name__]:
            reply = structured_response(
                RetrievalPlan(
                    retrieval_queries=['is alcohol reimbursed with a meal', 'receipts for a lunch'],
                    check_guidance=[],
                )
            )
        elif bound_tool_names == [JudgementReport.__name__]:
            reply = structured_response(JudgementReport(verdicts=self.verdicts))
        else:
            reply = AIMessage(content='Nothing to look up.')
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
    queue = InMemoryQueue(
        visibility_timeout_seconds=application_settings.queue_visibility_timeout_seconds
    )
    app = create_app(
        ApiDependencies(
            settings=application_settings,
            policy=POLICY,
            seed_data=SEED_DATA,
            connection_pool=connection_pool,
            queue=queue,
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://testserver'
    ) as client:
        assert application_settings.decision_event_url is not None
        sender = RecordingSender(
            SignedDecisionEventSender(
                client=client,
                url=application_settings.decision_event_url,
                key=b'test-decision-event-signing-key',
            )
        )
        embedder = build_embedder(application_settings)
        chat_model = PlanningChatModel(model_id='fake-chat')
        async with connection_pool.connection() as connection:
            await index_policy(connection, POLICY, embedder)
        graph = build_graph(
            GraphConfig(
                settings=application_settings,
                policy=POLICY,
                seed_data=SEED_DATA,
                chat_model=chat_model,
                embedder=embedder,
                connection_pool=connection_pool,
                checkpointer=AsyncPostgresSaver(connection_pool),
                decision_event_sender=sender,
            )
        )
        yield Application(
            client=client,
            queue=queue,
            worker=WorkerRuntime(graph=graph, connection_pool=connection_pool, tracing=None),
            connection_pool=connection_pool,
            sender=sender,
            branch_database=branch_database,
            chat_model=chat_model,
        )


async def test_health_returns_200(application: Application) -> None:
    response = await application.client.get('/health')

    assert response.status_code == 200


async def test_the_deployment_smoke_check_passes_against_the_application(
    application: Application,
) -> None:
    assert await smoke_failures(application.client) == []


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
    assert [delivery.message.type for delivery in application.queue.receive()] == ['start']


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
    assert [delivery.message.type for delivery in application.queue.receive()] == ['start']


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
    await process_message(start.message, application.worker)

    await process_message(start.message, application.worker)

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


async def test_a_rejection_without_a_reason_returns_422_and_changes_nothing(
    application: Application,
) -> None:
    submission_id = await claim_waiting_for_review(application)

    decided = await application.decide(
        submission_id,
        application.issue_token('EMP-001'),
        missing_receipt_decision('reject', None),
    )

    assert decided.status_code == 422
    assert await application.status_of(submission_id) == 'pending_review'
    assert application.queue.receive() == []


async def test_a_decision_on_a_submission_not_yet_reviewed_returns_409_and_changes_nothing(
    application: Application,
) -> None:
    response = await application.post_claim(
        expense_claim_delivery(unique_claim_id(), amount='30.00', receipt_text=None)
    )
    submission_id = response.json()['submission_id']

    decided = await application.decide(
        submission_id,
        application.issue_token('EMP-001'),
        missing_receipt_decision('approve', None),
    )

    assert decided.status_code == 409
    assert await application.status_of(submission_id) == 'received'
    assert [delivery.message.type for delivery in application.queue.receive()] == ['start']


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
    assert [delivery.message.type for delivery in application.queue.receive()] == ['resume']


def lambda_event_from(received: 'ReceiveMessageResultTypeDef') -> dict[str, Any]:
    return {
        'Records': [
            {
                'messageId': message['MessageId'],
                'receiptHandle': message['ReceiptHandle'],
                'body': message['Body'],
                'attributes': message.get('Attributes', {}),
                'eventSource': 'aws:sqs',
            }
            for message in received.get('Messages', [])
        ]
    }


@asynccontextmanager
async def queue_backed_client(
    application: Application, client: 'SQSClient', branch_queue: str
) -> AsyncIterator[httpx.AsyncClient]:
    queue_backed_app = create_app(
        ApiDependencies(
            settings=SETTINGS,
            policy=POLICY,
            seed_data=SEED_DATA,
            connection_pool=application.connection_pool,
            queue=SqsQueue(client=client, queue_url=branch_queue),
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=queue_backed_app), base_url='http://testserver'
    ) as api_client:
        yield api_client


async def post_claim_through(api_client: httpx.AsyncClient, delivery: dict[str, Any]) -> str:
    raw_body = json.dumps(delivery).encode()
    response = await api_client.post(
        '/webhooks/expense-claims', content=raw_body, headers=signed_headers(WEBHOOK_KEY, raw_body)
    )
    submission_id: str = response.json()['submission_id']
    return submission_id


async def test_the_worker_handler_runs_a_start_message_from_elasticmq_to_complete(
    application: Application, branch_queue: str
) -> None:
    client = sqs_client(SETTINGS)
    async with queue_backed_client(application, client, branch_queue) as api_client:
        submission_id = await post_claim_through(
            api_client, expense_claim_delivery(unique_claim_id())
        )
    received = client.receive_message(QueueUrl=branch_queue, MaxNumberOfMessages=10)

    outcome = await handle_sqs_event(lambda_event_from(received), application.worker)

    assert outcome == {'batchItemFailures': []}
    assert await application.status_of(submission_id) == 'complete'
    assert [event.submission_id for event in application.sender.sent] == [submission_id]


async def test_the_worker_handler_runs_a_resume_message_from_elasticmq_to_complete(
    application: Application, branch_queue: str
) -> None:
    client = sqs_client(SETTINGS)
    token = application.issue_token('EMP-001')
    async with queue_backed_client(application, client, branch_queue) as api_client:
        submission_id = await post_claim_through(
            api_client,
            expense_claim_delivery(unique_claim_id(), amount='30.00', receipt_text=None),
        )
        started = client.receive_message(QueueUrl=branch_queue, MaxNumberOfMessages=10)
        await handle_sqs_event(lambda_event_from(started), application.worker)
        for message in started.get('Messages', []):
            client.delete_message(QueueUrl=branch_queue, ReceiptHandle=message['ReceiptHandle'])
        await api_client.post(
            f'/submissions/{submission_id}/decision',
            json=missing_receipt_decision('reject', 'Receipt never supplied'),
            headers={'Authorization': f'Bearer {token}'},
        )
    resumed = client.receive_message(QueueUrl=branch_queue, MaxNumberOfMessages=10)

    outcome = await handle_sqs_event(lambda_event_from(resumed), application.worker)

    assert outcome == {'batchItemFailures': []}
    assert await application.status_of(submission_id) == 'complete'
    assert [(event.submission_id, event.outcome) for event in application.sender.sent] == [
        (submission_id, 'rejected')
    ]


def lambda_record(message_id: str, body: str) -> dict[str, Any]:
    return {'messageId': message_id, 'body': body, 'eventSource': 'aws:sqs'}


async def test_a_failing_record_and_every_record_after_it_are_reported_as_batch_failures(
    application: Application,
) -> None:
    first = await application.post_claim(expense_claim_delivery(unique_claim_id()))
    last = await application.post_claim(expense_claim_delivery(unique_claim_id()))
    first_start, last_start = application.queue.receive()
    event = {
        'Records': [
            lambda_record('message-1', first_start.message.model_dump_json()),
            lambda_record('message-2', '{"type": "unknown"}'),
            lambda_record('message-3', last_start.message.model_dump_json()),
        ]
    }

    outcome = await handle_sqs_event(event, application.worker)

    assert outcome == {
        'batchItemFailures': [{'itemIdentifier': 'message-2'}, {'itemIdentifier': 'message-3'}]
    }
    assert await application.status_of(first.json()['submission_id']) == 'complete'
    assert await application.status_of(last.json()['submission_id']) == 'received'


async def test_the_worker_app_answers_a_queue_event_posted_to_it_with_the_batch_failures(
    application: Application,
) -> None:
    accepted = await application.post_claim(expense_claim_delivery(unique_claim_id()))
    (start,) = application.queue.receive()
    event = {
        'Records': [
            lambda_record('message-1', start.message.model_dump_json()),
            lambda_record('message-2', '{"type": "unknown"}'),
        ]
    }
    worker_app = create_worker_app(application.worker)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=worker_app), base_url='http://worker'
    ) as worker_client:
        response = await worker_client.post('/events', json=event)

    assert response.json() == {'batchItemFailures': [{'itemIdentifier': 'message-2'}]}
    assert await application.status_of(accepted.json()['submission_id']) == 'complete'


def flagged(
    code: FindingCode, clause_id: str, quotes: list[QuotedText], **fields: Any
) -> CheckVerdict:
    return CheckVerdict.model_validate(
        {
            'code': code,
            'outcome': 'flagged',
            'clause_reference': clause_reference(clause_id),
            'quotes': quotes,
            'cited_precedent_ids': [],
            'precedent_id': None,
        }
        | fields
    )


PURPOSE_QUOTE = QuotedText(field='business_purpose', text='visiting the Leeds office')


@pytest.mark.parametrize(
    ('verdict', 'reason'),
    [
        (flagged('PURPOSE_UNCLEAR', 'approved_suppliers', [PURPOSE_QUOTE]), 'clause_not_retrieved'),
        (
            flagged(
                'ALCOHOL_NOT_PERMITTED',
                'no_alcohol',
                [QuotedText(field='receipt_text', text='1 x Pint of lager')],
            ),
            'quote_not_verbatim',
        ),
        (
            flagged(
                'ALCOHOL_NOT_PERMITTED',
                'no_alcohol',
                [QuotedText(field='receipt_text', text='1 x Soup 6.50')],
                cited_precedent_ids=['PRECEDENT-1'],
            ),
            'precedent_not_supplied',
        ),
    ],
    ids=['an unretrieved clause', 'a quote not in the submission', 'an unsupplied precedent'],
)
async def test_a_judgement_finding_failing_verification_is_dropped_and_audited(
    application: Application, verdict: CheckVerdict, reason: str
) -> None:
    application.chat_model.verdicts = [verdict]
    response = await application.post_claim(expense_claim_delivery(unique_claim_id()))
    submission_id = response.json()['submission_id']

    await application.run_worker()

    state = await application.review_state(submission_id)
    assert verdict.clause_reference is not None
    assert (
        verdict.clause_reference
        in [clause['clause_reference'] for clause in state['retrieved_clauses']]
    ) is (reason != 'clause_not_retrieved')
    assert state['findings'] == []
    assert {
        'node': 'verify',
        'event': 'finding_dropped',
        'detail': {
            'code': verdict.code,
            'clause_reference': verdict.clause_reference,
            'reason': reason,
        },
    } in state['audit']
    assert await application.status_of(submission_id) == 'complete'


async def test_rule_and_judgement_findings_reach_review_as_one_verified_set(
    application: Application,
) -> None:
    application.chat_model.verdicts = [
        flagged(
            'ALCOHOL_NOT_PERMITTED',
            'no_alcohol',
            [QuotedText(field='receipt_text', text='1 x Pint of   lager 12.50')],
        ),
        flagged(
            'ALCOHOL_NOT_PERMITTED',
            'no_alcohol',
            [QuotedText(field='receipt_text', text='Pint of lager')],
        ),
    ]
    response = await application.post_claim(
        expense_claim_delivery(
            unique_claim_id(),
            spent_on='2026-09-14',
            amount='42.50',
            receipt_text=(
                'Harbour Kitchen\n1 x Set menu 30.00\n1 x Pint of lager 12.50\nTOTAL 42.50'
            ),
        )
    )
    submission_id = response.json()['submission_id']
    await application.run_worker()

    listed = await application.pending_review(application.issue_token('EMP-001'))

    [review] = [
        item for item in listed.json()['submissions'] if item['submission_id'] == submission_id
    ]
    assert [
        (finding['code'], finding['source'], finding['severity'], finding['quotes'])
        for finding in review['findings']
    ] == [
        ('DUPLICATE_SUBMISSION', 'rule', 'high', []),
        (
            'ALCOHOL_NOT_PERMITTED',
            'judgement',
            'medium',
            [{'field': 'receipt_text', 'text': '1 x Pint of   lager 12.50'}],
        ),
    ]
    assert review['recommendation'] == 'reject'


async def test_a_duplicate_resume_message_after_completion_sends_no_second_event(
    application: Application,
) -> None:
    submission_id = await claim_waiting_for_review(application)
    await application.decide(
        submission_id, application.issue_token('EMP-001'), missing_receipt_decision('approve', None)
    )
    [resume] = application.queue.receive()
    await process_message(resume.message, application.worker)

    await process_message(resume.message, application.worker)

    assert await application.status_of(submission_id) == 'complete'
    assert [event.submission_id for event in application.sender.sent] == [submission_id]


async def test_a_resume_message_for_an_undecided_submission_is_a_no_op(
    application: Application,
) -> None:
    submission_id = await claim_waiting_for_review(application)
    record = await submissions.find_submission(
        application.connection_pool, uuid.UUID(submission_id)
    )
    assert record is not None

    await process_message(
        QueueMessage(type='resume', submission_id=record.id, thread_id=record.idempotency_key),
        application.worker,
    )

    assert await application.status_of(submission_id) == 'pending_review'
    assert application.sender.sent == []


@dataclass
class FailingFirstSender:
    inner: DecisionEventSender
    attempts: int = 0

    async def send(self, event: DecisionEvent) -> None:
        self.attempts += 1
        if self.attempts == 1:
            raise httpx.ConnectError('the ERP is unreachable')
        await self.inner.send(event)


async def test_a_resume_redelivered_after_a_failed_erp_send_completes_with_one_rejection(
    application: Application,
) -> None:
    submission_id = await claim_waiting_for_review(application)
    await application.decide(
        submission_id,
        application.issue_token('EMP-001'),
        missing_receipt_decision('reject', 'Receipt never supplied'),
    )
    [resume] = application.queue.receive()
    application.sender.inner = FailingFirstSender(application.sender.inner)
    with pytest.raises(httpx.ConnectError):
        await process_message(resume.message, application.worker)

    await process_message(resume.message, application.worker)

    assert await application.status_of(submission_id) == 'complete'
    events = await application.recorded_events(submission_id)
    assert [(event['outcome'], event['reason']) for event in events] == [
        ('rejected', 'Receipt never supplied')
    ]


def every_submission_changed_until_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=1)


async def test_a_decided_claim_whose_resume_was_lost_is_swept_and_completes_once(
    application: Application,
) -> None:
    submission_id = await claim_waiting_for_review(application)
    await application.decide(
        submission_id, application.issue_token('EMP-001'), missing_receipt_decision('approve', None)
    )
    [lost_resume] = application.queue.receive()
    application.queue.acknowledge(lost_resume.receipt_handle)

    await sweep(
        application.connection_pool, application.queue, every_submission_changed_until_now()
    )
    await application.run_worker_for(submission_id)

    assert await application.status_of(submission_id) == 'complete'
    assert [event.submission_id for event in application.sender.sent] == [submission_id]


async def test_a_sweep_duplicating_an_in_flight_resume_still_completes_once(
    application: Application,
) -> None:
    submission_id = await claim_waiting_for_review(application)
    await application.decide(
        submission_id, application.issue_token('EMP-001'), missing_receipt_decision('approve', None)
    )

    await sweep(
        application.connection_pool, application.queue, every_submission_changed_until_now()
    )
    resumes = [
        delivery.message
        for delivery in application.queue.receive()
        if str(delivery.message.submission_id) == submission_id
    ]
    for resume in resumes:
        await process_message(resume, application.worker)

    assert [resume.type for resume in resumes] == ['resume', 'resume']
    assert await application.status_of(submission_id) == 'complete'
    assert [event.submission_id for event in application.sender.sent] == [submission_id]


async def test_a_claim_decided_more_recently_than_the_threshold_is_not_swept(
    application: Application,
) -> None:
    submission_id = await claim_waiting_for_review(application)
    await application.decide(
        submission_id, application.issue_token('EMP-001'), missing_receipt_decision('approve', None)
    )
    [resume] = application.queue.receive()
    application.queue.acknowledge(resume.receipt_handle)

    await sweep(
        application.connection_pool,
        application.queue,
        datetime.datetime.now(datetime.UTC)
        - datetime.timedelta(seconds=SETTINGS.sweeper_stuck_after_seconds),
    )

    assert submission_id not in [
        str(delivery.message.submission_id) for delivery in application.queue.receive()
    ]
    assert await application.status_of(submission_id) == 'decided'


async def test_a_received_claim_whose_start_was_lost_is_swept_and_completes_once(
    application: Application,
) -> None:
    response = await application.post_claim(expense_claim_delivery(unique_claim_id()))
    submission_id = response.json()['submission_id']
    [lost_start] = application.queue.receive()
    application.queue.acknowledge(lost_start.receipt_handle)

    await sweep(
        application.connection_pool, application.queue, every_submission_changed_until_now()
    )
    await application.run_worker_for(submission_id)

    assert await application.status_of(submission_id) == 'complete'
    assert [event.submission_id for event in application.sender.sent] == [submission_id]


async def test_a_claim_waiting_for_review_is_never_swept(application: Application) -> None:
    submission_id = await claim_waiting_for_review(application)

    await sweep(
        application.connection_pool, application.queue, every_submission_changed_until_now()
    )

    assert submission_id not in [
        str(delivery.message.submission_id) for delivery in application.queue.receive()
    ]


async def test_a_stuck_claim_is_swept_once_until_its_status_changes(
    application: Application,
) -> None:
    submission_id = await claim_waiting_for_review(application)
    await application.decide(
        submission_id, application.issue_token('EMP-001'), missing_receipt_decision('approve', None)
    )
    [lost_resume] = application.queue.receive()
    application.queue.acknowledge(lost_resume.receipt_handle)

    await sweep(
        application.connection_pool, application.queue, every_submission_changed_until_now()
    )
    await sweep(
        application.connection_pool, application.queue, every_submission_changed_until_now()
    )

    swept = [
        delivery.message.type
        for delivery in application.queue.receive()
        if str(delivery.message.submission_id) == submission_id
    ]
    assert swept == ['resume']


async def test_the_sweeper_app_sweeps_claims_unchanged_for_the_configured_threshold(
    application: Application,
) -> None:
    submission_id = await claim_waiting_for_review(application)
    await application.decide(
        submission_id, application.issue_token('EMP-001'), missing_receipt_decision('approve', None)
    )
    [lost_resume] = application.queue.receive()
    application.queue.acknowledge(lost_resume.receipt_handle)
    decided_at = datetime.datetime.now(datetime.UTC)
    seconds_since_decision = [SETTINGS.sweeper_stuck_after_seconds - 60]
    sweeper_app = create_sweeper_app(
        application.connection_pool,
        application.queue,
        SETTINGS,
        clock=lambda: decided_at + datetime.timedelta(seconds=seconds_since_decision[0]),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=sweeper_app), base_url='http://sweeper'
    ) as sweeper:
        await sweeper.post('/events', json={})
        swept_too_early = submission_id in [
            str(delivery.message.submission_id) for delivery in application.queue.receive()
        ]
        seconds_since_decision[0] = SETTINGS.sweeper_stuck_after_seconds + 1
        response = await sweeper.post('/events', json={})
    await application.run_worker_for(submission_id)

    assert not swept_too_early
    assert response.status_code == 200
    assert await application.status_of(submission_id) == 'complete'
