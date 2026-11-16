import json
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import boto3
from botocore.config import Config
from botocore.exceptions import EndpointConnectionError

from spendguard.branch_names import dead_letter_queue_name_for, queue_name_for_branch
from spendguard.queue.messages import MAX_RECEIVE_COUNT, QueueMessage, ReceivedMessage
from spendguard.settings import Settings

if TYPE_CHECKING:
    from mypy_boto3_sqs import SQSClient

LOCAL_CREDENTIAL = 'local'
CONNECT_TIMEOUT_SECONDS = 3
MAX_MESSAGES_PER_RECEIVE = 10


class QueueUnreachableError(RuntimeError):
    pass


@dataclass(frozen=True)
class QueueUrls:
    queue_url: str
    dead_letter_queue_url: str


def sqs_client(settings: Settings) -> 'SQSClient':
    client_config = Config(
        connect_timeout=CONNECT_TIMEOUT_SECONDS,
        retries={'max_attempts': 1},
    )
    if settings.sqs_endpoint_url is None:
        return boto3.client('sqs', region_name=settings.aws_region, config=client_config)
    return boto3.client(
        'sqs',
        region_name=settings.aws_region,
        endpoint_url=settings.sqs_endpoint_url,
        aws_access_key_id=LOCAL_CREDENTIAL,
        aws_secret_access_key=LOCAL_CREDENTIAL,
        config=client_config,
    )


def _create_fifo_queue(client: 'SQSClient', queue_name: str) -> str:
    return client.create_queue(QueueName=queue_name, Attributes={'FifoQueue': 'true'})['QueueUrl']


def ensure_queue(
    client: 'SQSClient', queue_name: str, visibility_timeout_seconds: int
) -> QueueUrls:
    dead_letter_queue_url = _create_fifo_queue(client, dead_letter_queue_name_for(queue_name))
    dead_letter_queue_arn = client.get_queue_attributes(
        QueueUrl=dead_letter_queue_url, AttributeNames=['QueueArn']
    )['Attributes']['QueueArn']
    queue_url = _create_fifo_queue(client, queue_name)
    client.set_queue_attributes(
        QueueUrl=queue_url,
        Attributes={
            'VisibilityTimeout': str(visibility_timeout_seconds),
            'RedrivePolicy': json.dumps(
                {'deadLetterTargetArn': dead_letter_queue_arn, 'maxReceiveCount': MAX_RECEIVE_COUNT}
            ),
        },
    )
    return QueueUrls(queue_url=queue_url, dead_letter_queue_url=dead_letter_queue_url)


def ensure_branch_queue(settings: Settings) -> str:
    if settings.queue_url is not None:
        return settings.queue_url
    try:
        queue_urls = ensure_queue(
            sqs_client(settings),
            queue_name_for_branch(settings.branch_name),
            settings.queue_visibility_timeout_seconds,
        )
    except EndpointConnectionError as error:
        raise QueueUnreachableError(
            f'The queue service at {settings.sqs_endpoint_url} is unreachable;'
            ' start it with make up'
        ) from error
    return queue_urls.queue_url


@dataclass(frozen=True)
class SqsQueue:
    client: 'SQSClient'
    queue_url: str

    def send(self, message: QueueMessage) -> None:
        self.client.send_message(
            QueueUrl=self.queue_url,
            MessageBody=message.model_dump_json(),
            MessageGroupId=message.thread_id,
            MessageDeduplicationId=uuid.uuid4().hex,
        )

    def receive(self) -> list[ReceivedMessage]:
        response = self.client.receive_message(
            QueueUrl=self.queue_url, MaxNumberOfMessages=MAX_MESSAGES_PER_RECEIVE
        )
        return [
            ReceivedMessage(
                message=QueueMessage.model_validate_json(received['Body']),
                receipt_handle=received['ReceiptHandle'],
            )
            for received in response.get('Messages', [])
        ]

    def acknowledge(self, receipt_handle: str) -> None:
        self.client.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)
