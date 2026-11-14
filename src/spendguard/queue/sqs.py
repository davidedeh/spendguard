from typing import TYPE_CHECKING

import boto3
from botocore.config import Config
from botocore.exceptions import EndpointConnectionError

from spendguard.branch_names import queue_name_for_branch
from spendguard.settings import Settings

if TYPE_CHECKING:
    from mypy_boto3_sqs import SQSClient

LOCAL_CREDENTIAL = 'local'
CONNECT_TIMEOUT_SECONDS = 3


class QueueUnreachableError(RuntimeError):
    pass


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


def ensure_branch_queue(settings: Settings) -> str:
    if settings.queue_url is not None:
        return settings.queue_url
    client = sqs_client(settings)
    try:
        response = client.create_queue(
            QueueName=queue_name_for_branch(settings.branch_name),
            Attributes={'FifoQueue': 'true'},
        )
    except EndpointConnectionError as error:
        raise QueueUnreachableError(
            f'The queue service at {settings.sqs_endpoint_url} is unreachable;'
            ' start it with make up'
        ) from error
    return response['QueueUrl']
