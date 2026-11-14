import pytest
from mypy_boto3_sqs import SQSClient

from network import closed_port
from settings_factory import make_test_settings
from spendguard.queue.sqs import QueueUnreachableError, ensure_branch_queue, sqs_client


def test_the_branch_queue_delivers_a_message_within_its_group(
    branch_queue: str,
) -> None:
    client: SQSClient = sqs_client(make_test_settings())
    client.send_message(
        QueueUrl=branch_queue,
        MessageBody='{"type": "start"}',
        MessageGroupId='submission-7',
        MessageDeduplicationId='submission-7-start',
    )

    received = client.receive_message(QueueUrl=branch_queue, WaitTimeSeconds=1)

    assert [message['Body'] for message in received['Messages']] == ['{"type": "start"}']


def test_an_unreachable_queue_service_raises_instead_of_skipping() -> None:
    settings = make_test_settings(sqs_endpoint_url=f'http://127.0.0.1:{closed_port()}')

    with pytest.raises(QueueUnreachableError, match='make up'):
        ensure_branch_queue(settings)
