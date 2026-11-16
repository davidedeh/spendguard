import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from spendguard.queue.messages import MAX_RECEIVE_COUNT, QueueMessage, ReceivedMessage


@dataclass
class _StoredMessage:
    message: QueueMessage
    receipt_handle: str
    invisible_until: float
    receive_count: int


class InMemoryQueue:
    def __init__(
        self, visibility_timeout_seconds: int, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._visibility_timeout_seconds = visibility_timeout_seconds
        self._clock = clock
        self._stored: list[_StoredMessage] = []
        self.dead_letters: list[QueueMessage] = []

    def send(self, message: QueueMessage) -> None:
        self._stored.append(
            _StoredMessage(
                message=message,
                receipt_handle=uuid.uuid4().hex,
                invisible_until=self._clock(),
                receive_count=0,
            )
        )

    def receive(self) -> list[ReceivedMessage]:
        now = self._clock()
        self._move_exhausted_messages_to_dead_letters(now)
        held_thread_ids = {
            stored.message.thread_id for stored in self._stored if stored.invisible_until > now
        }
        received: list[ReceivedMessage] = []
        for stored in self._stored:
            if stored.message.thread_id in held_thread_ids:
                continue
            stored.invisible_until = now + self._visibility_timeout_seconds
            stored.receive_count += 1
            received.append(
                ReceivedMessage(message=stored.message, receipt_handle=stored.receipt_handle)
            )
        return received

    def _move_exhausted_messages_to_dead_letters(self, now: float) -> None:
        remaining: list[_StoredMessage] = []
        for stored in self._stored:
            if stored.receive_count >= MAX_RECEIVE_COUNT and stored.invisible_until <= now:
                self.dead_letters.append(stored.message)
            else:
                remaining.append(stored)
        self._stored = remaining

    def acknowledge(self, receipt_handle: str) -> None:
        self._stored = [
            stored for stored in self._stored if stored.receipt_handle != receipt_handle
        ]
