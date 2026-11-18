import asyncio
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

import httpx
from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from spendguard import settings
from spendguard.db.database import (
    ConnectionPool,
    Lifespan,
    async_connection_pool,
    branch_connection_info,
    opened_pool,
    pool_lifespan,
)
from spendguard.graph.build import ReviewGraph, build_graph
from spendguard.graph.config import GraphConfig, thread_config
from spendguard.graph.nodes.execute import SignedDecisionEventSender
from spendguard.graph.state import initial_review_state
from spendguard.models.factory import build_chat_model, build_embedder
from spendguard.observability.tracing import (
    Tracing,
    create_tracing,
    flush_traces_before_return,
    traced_advance,
)
from spendguard.policy.loader import POLICY_PATH, load_policy
from spendguard.queue.messages import STATUS_EXPECTED_BY, ConsumableQueue, QueueMessage
from spendguard.queue.sqs import SqsQueue, ensure_branch_queue, sqs_client
from spendguard.seed.loader import SEED_DIRECTORY, read_seed_directory
from spendguard.settings import Settings
from spendguard.store import submissions
from spendguard.structured_logging import configure_logging, log_event

IDLE_POLL_SECONDS = 1


@dataclass(frozen=True)
class WorkerRuntime:
    graph: ReviewGraph
    connection_pool: ConnectionPool
    tracing: Tracing | None


async def process_message(message: QueueMessage, runtime: WorkerRuntime) -> None:
    submission = await submissions.find_submission(runtime.connection_pool, message.submission_id)
    if submission is None or submission.status != STATUS_EXPECTED_BY[message.type]:
        log_event(
            'stale_message_dropped',
            submission_id=str(message.submission_id),
            message_type=message.type,
        )
        return
    thread = thread_config(submission.idempotency_key)
    snapshot = await runtime.graph.aget_state(thread)
    if snapshot.interrupts:
        await traced_advance(
            runtime.tracing,
            runtime.graph,
            Command(resume=submission.decision),
            thread,
            str(submission.id),
        )
        return
    if snapshot.next:
        await traced_advance(runtime.tracing, runtime.graph, None, thread, str(submission.id))
        return
    initial_state = initial_review_state(
        submission.id, submission.idempotency_key, submission.submission_type, submission.payload
    )
    await traced_advance(runtime.tracing, runtime.graph, initial_state, thread, str(submission.id))


class SqsRecord(BaseModel):
    message_id: str = Field(alias='messageId')
    body: str


class SqsEvent(BaseModel):
    records: list[SqsRecord] = Field(alias='Records')


class BatchItemFailure(TypedDict):
    itemIdentifier: str


class BatchResponse(TypedDict):
    batchItemFailures: list[BatchItemFailure]


async def handle_sqs_event(event: Mapping[str, object], runtime: WorkerRuntime) -> BatchResponse:
    failed_message_ids: list[str] = []
    for record in SqsEvent.model_validate(event).records:
        if failed_message_ids:
            failed_message_ids.append(record.message_id)
            continue
        try:
            await process_message(QueueMessage.model_validate_json(record.body), runtime)
        except Exception as error:
            log_event(
                'message_failed', message_id=record.message_id, error_type=type(error).__name__
            )
            failed_message_ids.append(record.message_id)
    return {
        'batchItemFailures': [{'itemIdentifier': message_id} for message_id in failed_message_ids]
    }


async def process_available_messages(queue: ConsumableQueue, runtime: WorkerRuntime) -> int:
    deliveries = queue.receive()
    for delivery in deliveries:
        try:
            await process_message(delivery.message, runtime)
        except Exception as error:
            log_event(
                'message_failed',
                submission_id=str(delivery.message.submission_id),
                error_type=type(error).__name__,
            )
            break
        queue.acknowledge(delivery.receipt_handle)
    return len(deliveries)


def build_worker_runtime(loaded_settings: Settings) -> WorkerRuntime:
    if (
        loaded_settings.decision_event_signing_key is None
        or loaded_settings.decision_event_url is None
    ):
        raise ValueError(
            'DECISION_EVENT_SIGNING_KEY and DECISION_EVENT_URL must be set'
            ' for the worker to send events'
        )
    connection_pool = async_connection_pool(branch_connection_info(loaded_settings))
    graph = build_graph(
        GraphConfig(
            settings=loaded_settings,
            policy=load_policy(POLICY_PATH),
            seed_data=read_seed_directory(SEED_DIRECTORY),
            chat_model=build_chat_model(loaded_settings, loaded_settings.primary_model_key),
            embedder=build_embedder(loaded_settings),
            connection_pool=connection_pool,
            checkpointer=AsyncPostgresSaver(connection_pool),
            decision_event_sender=SignedDecisionEventSender(
                client=httpx.AsyncClient(),
                url=loaded_settings.decision_event_url,
                key=loaded_settings.decision_event_signing_key.get_secret_value().encode(),
            ),
            precedent_memory=True,
            record_precedents=True,
            policy_context='retrieved_clauses',
        )
    )
    return WorkerRuntime(
        graph=graph, connection_pool=connection_pool, tracing=create_tracing(loaded_settings)
    )


def create_worker_app(runtime: WorkerRuntime, lifespan: Lifespan | None = None) -> FastAPI:
    app = FastAPI(title='SpendGuard worker', lifespan=lifespan)

    @app.post('/events')
    @flush_traces_before_return
    async def events(event: dict[str, object]) -> BatchResponse:
        return await handle_sqs_event(event, runtime)

    return app


def create_production_worker_app() -> FastAPI:
    loaded_settings = settings.load_settings()
    configure_logging(loaded_settings.log_level)
    runtime = build_worker_runtime(loaded_settings)

    return create_worker_app(runtime, lifespan=pool_lifespan(runtime.connection_pool))


async def _run_local_worker() -> None:
    loaded_settings = settings.load_settings()
    if loaded_settings.sqs_endpoint_url is None:
        sys.exit('the local worker consumes ElasticMQ: set SQS_ENDPOINT_URL in .env')
    configure_logging(loaded_settings.log_level)
    runtime = build_worker_runtime(loaded_settings)
    queue = SqsQueue(
        client=sqs_client(loaded_settings), queue_url=ensure_branch_queue(loaded_settings)
    )
    async with opened_pool(runtime.connection_pool):
        while True:
            if await process_available_messages(queue, runtime) == 0:
                await asyncio.sleep(IDLE_POLL_SECONDS)


if __name__ == '__main__':
    asyncio.run(_run_local_worker())
