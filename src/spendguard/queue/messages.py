import uuid
from typing import Literal, Protocol

from spendguard.strict_model import StrictModel

MessageType = Literal['start', 'resume']


class QueueMessage(StrictModel):
    type: MessageType
    submission_id: uuid.UUID
    thread_id: str


class MessageQueue(Protocol):
    def send(self, message: QueueMessage) -> None: ...
