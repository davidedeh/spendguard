from spendguard.queue.messages import QueueMessage


class InMemoryQueue:
    def __init__(self) -> None:
        self._waiting: list[QueueMessage] = []

    def send(self, message: QueueMessage) -> None:
        self._waiting.append(message)

    def receive(self) -> list[QueueMessage]:
        received, self._waiting = self._waiting, []
        return received
