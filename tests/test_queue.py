import io
import json
import time
import uuid

import pytest
from mypy_boto3_sqs import SQSClient

from network import closed_port
from settings_factory import make_test_settings
from spendguard.branch_names import queue_name_for_branch
from spendguard.queue.messages import MAX_RECEIVE_COUNT, QueueMessage
from spendguard.queue.sqs import (
    NotALocalQueueError,
    QueueUnreachableError,
    SqsQueue,
    ensure_branch_queue,
    ensure_queue,
    purge_branch_queues,
    sqs_client,
)

UNREADABLE_VISIBILITY_TIMEOUT_SECONDS = 1


def test_the_branch_queue_delivers_a_sent_message_and_drops_it_once_acknowledged(
    branch_queue: str,
) -> None:
    queue = SqsQueue(client=sqs_client(make_test_settings()), queue_url=branch_queue)
    sent = QueueMessage(type='start', submission_id=uuid.uuid4(), thread_id='submission-7')
    queue.send(sent)

    (delivery,) = queue.receive()
    queue.acknowledge(delivery.receipt_handle)

    assert delivery.message == sent
    assert queue.receive() == []


def test_an_unreachable_queue_service_raises_instead_of_skipping() -> None:
    settings = make_test_settings(sqs_endpoint_url=f'http://127.0.0.1:{closed_port()}')

    with pytest.raises(QueueUnreachableError, match='make up'):
        ensure_branch_queue(settings)


def test_an_unreadable_message_is_logged_left_queued_and_dead_lettered_after_three_receives(
    log_stream: io.StringIO,
) -> None:
    settings = make_test_settings()
    client: SQSClient = sqs_client(settings)
    queue_urls = ensure_queue(
        client,
        queue_name_for_branch(f'{settings.branch_name}-unreadable'),
        UNREADABLE_VISIBILITY_TIMEOUT_SECONDS,
    )
    client.purge_queue(QueueUrl=queue_urls.queue_url)
    client.purge_queue(QueueUrl=queue_urls.dead_letter_queue_url)
    unreadable = client.send_message(
        QueueUrl=queue_urls.queue_url,
        MessageBody='{"type": "start"}',
        MessageGroupId='submission-7',
        MessageDeduplicationId=uuid.uuid4().hex,
    )
    queue = SqsQueue(client=client, queue_url=queue_urls.queue_url)
    readable = QueueMessage(type='start', submission_id=uuid.uuid4(), thread_id='submission-8')
    queue.send(readable)

    received_per_attempt: list[list[QueueMessage]] = []
    for _ in range(MAX_RECEIVE_COUNT + 1):
        deliveries = queue.receive()
        received_per_attempt.append([delivery.message for delivery in deliveries])
        for delivery in deliveries:
            queue.acknowledge(delivery.receipt_handle)
        time.sleep(UNREADABLE_VISIBILITY_TIMEOUT_SECONDS + 0.2)

    assert received_per_attempt == [[readable], [], [], []]
    skipped = [
        line
        for line in map(json.loads, log_stream.getvalue().splitlines())
        if line['event'] == 'unreadable_message_skipped'
    ]
    assert [line['message_id'] for line in skipped] == [unreadable['MessageId']] * MAX_RECEIVE_COUNT
    dead_letters = client.receive_message(
        QueueUrl=queue_urls.dead_letter_queue_url, WaitTimeSeconds=1
    ).get('Messages', [])
    assert [message['MessageId'] for message in dead_letters] == [unreadable['MessageId']]


def test_purging_empties_the_branch_queue_and_its_dead_letter_queue() -> None:
    settings = make_test_settings(branch_name=f'{make_test_settings().branch_name}-purge')
    client: SQSClient = sqs_client(settings)
    queue_urls = ensure_queue(
        client,
        queue_name_for_branch(settings.branch_name),
        settings.queue_visibility_timeout_seconds,
    )
    for queue_url in (queue_urls.queue_url, queue_urls.dead_letter_queue_url):
        SqsQueue(client=client, queue_url=queue_url).send(
            QueueMessage(type='start', submission_id=uuid.uuid4(), thread_id='submission-7')
        )

    purge_branch_queues(settings)

    assert [
        client.receive_message(QueueUrl=queue_url).get('Messages', [])
        for queue_url in (queue_urls.queue_url, queue_urls.dead_letter_queue_url)
    ] == [[], []]


def test_purging_refuses_a_queue_service_that_is_not_local() -> None:
    settings = make_test_settings(sqs_endpoint_url=None)

    with pytest.raises(NotALocalQueueError):
        purge_branch_queues(settings)
