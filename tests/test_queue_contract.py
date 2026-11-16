import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pytest

from settings_factory import make_test_settings
from spendguard.branch_names import queue_name_for_branch
from spendguard.queue.memory import InMemoryQueue
from spendguard.queue.messages import ConsumableQueue, MessageType, QueueMessage
from spendguard.queue.sqs import SqsQueue, ensure_queue, sqs_client
from spendguard.settings import Settings

CONTRACT_VISIBILITY_TIMEOUT_SECONDS = 1


@dataclass
class FakeClock:
    now: float = 1000.0

    def __call__(self) -> float:
        return self.now


@dataclass
class QueueUnderTest:
    queue: ConsumableQueue
    wait_past_visibility_timeout: Callable[[], None]
    dead_letters: Callable[[], list[QueueMessage]]


def in_memory_queue_under_test() -> QueueUnderTest:
    clock = FakeClock()
    queue = InMemoryQueue(
        visibility_timeout_seconds=CONTRACT_VISIBILITY_TIMEOUT_SECONDS, clock=clock
    )

    def wait_past_visibility_timeout() -> None:
        clock.now += CONTRACT_VISIBILITY_TIMEOUT_SECONDS + 0.1

    return QueueUnderTest(
        queue=queue,
        wait_past_visibility_timeout=wait_past_visibility_timeout,
        dead_letters=lambda: list(queue.dead_letters),
    )


def elasticmq_queue_under_test(settings: Settings) -> QueueUnderTest:
    client = sqs_client(settings)
    queue_urls = ensure_queue(
        client,
        queue_name_for_branch(f'{settings.branch_name}-contract'),
        CONTRACT_VISIBILITY_TIMEOUT_SECONDS,
    )
    client.purge_queue(QueueUrl=queue_urls.queue_url)
    client.purge_queue(QueueUrl=queue_urls.dead_letter_queue_url)

    def wait_past_visibility_timeout() -> None:
        time.sleep(CONTRACT_VISIBILITY_TIMEOUT_SECONDS + 0.2)

    dead_letter_queue = SqsQueue(client=client, queue_url=queue_urls.dead_letter_queue_url)
    return QueueUnderTest(
        queue=SqsQueue(client=client, queue_url=queue_urls.queue_url),
        wait_past_visibility_timeout=wait_past_visibility_timeout,
        dead_letters=lambda: [delivery.message for delivery in dead_letter_queue.receive()],
    )


@pytest.fixture(params=['in_memory', 'elasticmq'])
def queue_under_test(request: pytest.FixtureRequest) -> Iterator[QueueUnderTest]:
    if request.param == 'in_memory':
        yield in_memory_queue_under_test()
    else:
        yield elasticmq_queue_under_test(make_test_settings())


def message(message_type: MessageType, thread_id: str) -> QueueMessage:
    return QueueMessage(type=message_type, submission_id=uuid.uuid4(), thread_id=thread_id)


def test_messages_in_one_group_arrive_in_the_order_they_were_sent(
    queue_under_test: QueueUnderTest,
) -> None:
    queue = queue_under_test.queue
    start = message('start', 'thread-a')
    resume = message('resume', 'thread-a')
    queue.send(start)
    queue.send(resume)

    received = []
    for _ in range(2):
        for delivery in queue.receive():
            received.append(delivery.message)
            queue.acknowledge(delivery.receipt_handle)

    assert received == [start, resume]


def test_a_group_is_held_while_its_earlier_message_is_in_flight(
    queue_under_test: QueueUnderTest,
) -> None:
    queue = queue_under_test.queue
    first_start = message('start', 'thread-a')
    queue.send(first_start)
    assert [delivery.message for delivery in queue.receive()] == [first_start]
    held_resume = message('resume', 'thread-a')
    other_start = message('start', 'thread-b')
    queue.send(held_resume)
    queue.send(other_start)

    assert [delivery.message for delivery in queue.receive()] == [other_start]


def test_an_unacknowledged_message_is_redelivered_after_the_visibility_timeout(
    queue_under_test: QueueUnderTest,
) -> None:
    queue = queue_under_test.queue
    unacknowledged = message('start', 'thread-a')
    acknowledged = message('start', 'thread-b')
    queue.send(unacknowledged)
    queue.send(acknowledged)
    for delivery in queue.receive():
        if delivery.message == acknowledged:
            queue.acknowledge(delivery.receipt_handle)
    assert queue.receive() == []

    queue_under_test.wait_past_visibility_timeout()

    assert [delivery.message for delivery in queue.receive()] == [unacknowledged]


def test_a_message_received_three_times_without_acknowledgement_is_dead_lettered(
    queue_under_test: QueueUnderTest,
) -> None:
    queue = queue_under_test.queue
    failing = message('start', 'thread-a')
    succeeding = message('start', 'thread-b')
    queue.send(failing)
    queue.send(succeeding)
    for delivery in queue.receive():
        if delivery.message == succeeding:
            queue.acknowledge(delivery.receipt_handle)
    for _ in range(2):
        queue_under_test.wait_past_visibility_timeout()
        assert [delivery.message for delivery in queue.receive()] == [failing]

    queue_under_test.wait_past_visibility_timeout()

    assert queue.receive() == []
    assert queue_under_test.dead_letters() == [failing]
