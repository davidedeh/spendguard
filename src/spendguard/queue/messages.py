import uuid
from dataclasses import dataclass
from typing import Literal, Protocol

from spendguard.strict_model import StrictModel

MessageType = Literal['start', 'resume']
MAX_RECEIVE_COUNT = 3


class QueueMessage(StrictModel):
    type: MessageType
    submission_id: uuid.UUID
    thread_id: str


@dataclass(frozen=True)
class ReceivedMessage:
    message: QueueMessage
    receipt_handle: str


class MessageQueue(Protocol):
    def send(self, message: QueueMessage) -> None: ...


class ConsumableQueue(MessageQueue, Protocol):
    def receive(self) -> list[ReceivedMessage]: ...

    def acknowledge(self, receipt_handle: str) -> None: ...
