from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from spendguard.db.database import ConnectionPool
from spendguard.graph.build import ReviewGraph
from spendguard.graph.state import ReviewState
from spendguard.queue.messages import MessageType, QueueMessage
from spendguard.store import submissions
from spendguard.store.submissions import SubmissionStatus
from spendguard.structured_logging import log_event

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


def handler(event: dict[str, object], context: object) -> dict[str, object]:
    raise NotImplementedError
