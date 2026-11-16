import asyncio
import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from spendguard import settings
from spendguard.db.database import ConnectionPool, async_connection_pool, branch_connection_info
from spendguard.graph.build import ReviewGraph, build_graph
from spendguard.graph.config import GraphConfig
from spendguard.graph.nodes.execute import SignedDecisionEventSender
from spendguard.graph.state import ReviewState
from spendguard.models.factory import build_chat_model, build_embedder
from spendguard.policy.loader import POLICY_PATH, load_policy
from spendguard.queue.messages import MessageType, QueueMessage
from spendguard.seed.loader import SEED_DIRECTORY, read_seed_directory
from spendguard.settings import Settings
from spendguard.store import submissions
from spendguard.store.submissions import SubmissionStatus
from spendguard.structured_logging import configure_logging, log_event

STATUS_EXPECTED_BY: dict[MessageType, SubmissionStatus] = {
    'start': 'received',
    'resume': 'decided',
}


@dataclass(frozen=True)
class WorkerRuntime:
    graph: ReviewGraph
    connection_pool: ConnectionPool


async def process_message(message: QueueMessage, runtime: WorkerRuntime) -> None:
    submission = await submissions.find_submission(runtime.connection_pool, message.submission_id)
    if submission is None or submission.status != STATUS_EXPECTED_BY[message.type]:
        log_event(
            'stale_message_dropped',
            submission_id=str(message.submission_id),
            message_type=message.type,
        )
        return
    thread: RunnableConfig = {'configurable': {'thread_id': submission.idempotency_key}}
    snapshot = await runtime.graph.aget_state(thread)
    if snapshot.interrupts:
        await runtime.graph.ainvoke(Command(resume=submission.decision), thread)
        return
    if snapshot.next:
        await runtime.graph.ainvoke(None, thread)
        return
    initial_state: ReviewState = {
        'submission_id': str(submission.id),
        'idempotency_key': submission.idempotency_key,
        'submission_type': submission.submission_type,
        'submission': submission.payload,
    }
    await runtime.graph.ainvoke(initial_state, thread)


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


async def open_worker_runtime(loaded_settings: Settings) -> WorkerRuntime:
    if loaded_settings.decision_event_signing_key is None:
        raise ValueError('DECISION_EVENT_SIGNING_KEY must be set for the worker to send events')
    connection_pool = async_connection_pool(branch_connection_info(loaded_settings))
    await connection_pool.open(wait=True)
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
        )
    )
    return WorkerRuntime(graph=graph, connection_pool=connection_pool)


@functools.cache
def _lambda_worker() -> tuple[asyncio.AbstractEventLoop, WorkerRuntime]:
    loaded_settings = settings.load_settings()
    configure_logging(loaded_settings.log_level)
    event_loop = asyncio.new_event_loop()
    runtime = event_loop.run_until_complete(open_worker_runtime(loaded_settings))
    return event_loop, runtime


def handler(event: dict[str, object], context: object) -> BatchResponse:
    event_loop, runtime = _lambda_worker()
    return event_loop.run_until_complete(handle_sqs_event(event, runtime))
