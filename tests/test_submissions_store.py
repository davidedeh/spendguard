import uuid
from typing import Any

from spendguard.db.database import ConnectionPool
from spendguard.store import submissions

PAYLOAD: dict[str, Any] = {
    'type': 'expense_claim',
    'claimant': 'EMP-003',
    'amount': '38.00',
    'business_purpose': 'Team dinner for Halden Group',
}


async def test_the_same_payload_again_is_a_repeat_whatever_its_key_order(
    connection_pool: ConnectionPool,
) -> None:
    idempotency_key = f'expense_claim:store-test:{uuid.uuid4()}'
    first = await submissions.record_submission(
        connection_pool,
        idempotency_key=idempotency_key,
        submission_type='expense_claim',
        payload=PAYLOAD,
    )

    again = await submissions.record_submission(
        connection_pool,
        idempotency_key=idempotency_key,
        submission_type='expense_claim',
        payload=dict(reversed(PAYLOAD.items())),
    )

    assert (first.outcome, again.outcome) == ('created', 'repeated')
    assert again.submission_id == first.submission_id


async def test_a_changed_payload_under_a_recorded_key_conflicts(
    connection_pool: ConnectionPool,
) -> None:
    idempotency_key = f'expense_claim:store-test:{uuid.uuid4()}'
    await submissions.record_submission(
        connection_pool,
        idempotency_key=idempotency_key,
        submission_type='expense_claim',
        payload=PAYLOAD,
    )

    changed = await submissions.record_submission(
        connection_pool,
        idempotency_key=idempotency_key,
        submission_type='expense_claim',
        payload=PAYLOAD | {'amount': '38.01'},
    )

    assert changed.outcome == 'conflicting'
