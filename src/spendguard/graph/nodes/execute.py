import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import httpx

from spendguard.graph.state import ReviewState
from spendguard.security.signatures import SIGNATURE_HEADER, sign
from spendguard.strict_model import StrictModel

if TYPE_CHECKING:
    from spendguard.graph.config import GraphConfig

EventOutcome = Literal['approved', 'rejected']
EVENT_OUTCOMES: dict[str, EventOutcome] = {'approve': 'approved', 'reject': 'rejected'}


class DecisionEvent(StrictModel):
    idempotency_key: str
    submission_id: str
    outcome: EventOutcome
    decided_by: str | None
    reason: str | None


class DecisionEventSender(Protocol):
    async def send(self, event: DecisionEvent) -> None: ...


@dataclass(frozen=True)
class SignedDecisionEventSender:
    client: httpx.AsyncClient
    url: str
    key: bytes

    async def send(self, event: DecisionEvent) -> None:
        raw_body = event.model_dump_json().encode()
        response = await self.client.post(
            self.url,
            content=raw_body,
            headers={
                SIGNATURE_HEADER: sign(self.key, int(time.time()), raw_body),
                'Content-Type': 'application/json',
            },
        )
        response.raise_for_status()


async def execute(state: ReviewState, graph_config: 'GraphConfig') -> ReviewState:
    decision = state['decision']
    event = DecisionEvent(
        idempotency_key=state['submission_id'],
        submission_id=state['submission_id'],
        outcome=EVENT_OUTCOMES[decision['outcome']],
        decided_by=decision['decided_by'],
        reason=decision['reason'],
    )
    await graph_config.decision_event_sender.send(event)
    return {
        'audit': [
            {
                'node': 'execute',
                'event': 'decision_event_sent',
                'detail': {'outcome': event.outcome},
            }
        ]
    }
