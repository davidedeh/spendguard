import datetime
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI

from spendguard import settings
from spendguard.api.app import Lifespan
from spendguard.db.database import ConnectionPool, async_connection_pool, branch_connection_info
from spendguard.queue.messages import STATUS_EXPECTED_BY, MessageQueue, MessageType, QueueMessage
from spendguard.queue.sqs import SqsQueue, ensure_branch_queue, sqs_client
from spendguard.settings import Settings
from spendguard.store import submissions
from spendguard.store.submissions import StuckSubmission, SubmissionStatus
from spendguard.structured_logging import configure_logging, log_event

MESSAGE_RESUMING: dict[SubmissionStatus, MessageType] = {
    status: message_type for message_type, status in STATUS_EXPECTED_BY.items()
}


async def sweep(
    connection_pool: ConnectionPool, queue: MessageQueue, stuck_before: datetime.datetime
) -> int:
    def enqueue(submission: StuckSubmission) -> None:
        queue.send(
            QueueMessage(
                type=MESSAGE_RESUMING[submission.status],
                submission_id=submission.id,
                thread_id=submission.idempotency_key,
            )
        )

    return await submissions.sweep_stuck_submissions(
        connection_pool, MESSAGE_RESUMING.keys(), stuck_before, enqueue
    )


def current_time() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def create_sweeper_app(
    connection_pool: ConnectionPool,
    queue: MessageQueue,
    loaded_settings: Settings,
    clock: Callable[[], datetime.datetime] = current_time,
    lifespan: Lifespan | None = None,
) -> FastAPI:
    app = FastAPI(title='SpendGuard sweeper', lifespan=lifespan)

    @app.post('/events')
    async def events() -> dict[str, int]:
        stuck_before = clock() - datetime.timedelta(
            seconds=loaded_settings.sweeper_stuck_after_seconds
        )
        swept_count = await sweep(connection_pool, queue, stuck_before)
        log_event('submissions_swept', swept_count=swept_count)
        return {'swept': swept_count}

    return app


def create_production_sweeper_app() -> FastAPI:
    loaded_settings = settings.load_settings()
    configure_logging(loaded_settings.log_level)
    connection_pool = async_connection_pool(branch_connection_info(loaded_settings))
    queue = SqsQueue(
        client=sqs_client(loaded_settings), queue_url=ensure_branch_queue(loaded_settings)
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await connection_pool.open(wait=True)
        yield
        await connection_pool.close()

    return create_sweeper_app(connection_pool, queue, loaded_settings, lifespan=lifespan)
