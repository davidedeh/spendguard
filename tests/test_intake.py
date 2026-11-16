from pathlib import Path
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from settings_factory import make_test_settings
from spendguard.db.database import ConnectionPool
from spendguard.graph.config import GraphConfig
from spendguard.graph.nodes.execute import DecisionEvent
from spendguard.graph.nodes.intake import intake
from spendguard.graph.state import ReviewState
from spendguard.models.factory import build_embedder
from spendguard.models.fake import FakeChatModel
from spendguard.policy.loader import POLICY_PATH, load_policy
from spendguard.seed.loader import read_seed_directory

POLICY = load_policy(POLICY_PATH)
SEED_DATA = read_seed_directory(Path(__file__).parent / 'fixtures' / 'seed')


class UnusedSender:
    async def send(self, event: DecisionEvent) -> None:
        raise AssertionError('intake never sends decision events')


def graph_config(connection_pool: ConnectionPool, **setting_overrides: Any) -> GraphConfig:
    settings = make_test_settings(**setting_overrides)
    return GraphConfig(
        settings=settings,
        policy=POLICY,
        seed_data=SEED_DATA,
        chat_model=FakeChatModel(model_id='fake-chat'),
        embedder=build_embedder(settings),
        connection_pool=connection_pool,
        checkpointer=AsyncPostgresSaver(connection_pool),
        decision_event_sender=UnusedSender(),
        precedent_memory=True,
    )


def expense_claim(**overrides: Any) -> dict[str, Any]:
    return {
        'type': 'expense_claim',
        'claimant': 'EMP-003',
        'spent_on': '2026-10-06',
        'category': 'subsistence',
        'quantity': 1,
        'merchant': 'Harbour Kitchen',
        'amount': '14.50',
        'currency': 'GBP',
        'claimed_exchange_rate': None,
        'business_purpose': 'Lunch while visiting the Leeds office for Halden Group',
        'receipt_text': 'Harbour Kitchen\n1 x Soup 6.50\n1 x Sandwich 8.00\nTOTAL 14.50',
    } | overrides


async def test_intake_removes_hidden_characters_and_records_which_field_changed(
    connection_pool: ConnectionPool,
) -> None:
    state: ReviewState = {
        'submission': expense_claim(business_purpose='Lunch\u200b\u200b for Halden Group\u0007')
    }

    result = intake(state, graph_config(connection_pool))

    assert result['submission'] == expense_claim(business_purpose='Lunch for Halden Group')
    assert result['audit'] == [
        {
            'node': 'intake',
            'event': 'field_cleaned',
            'detail': {'field': 'business_purpose', 'original_length': 25, 'cleaned_length': 22},
        }
    ]


async def test_intake_caps_over_length_fields_at_the_configured_length(
    connection_pool: ConnectionPool,
) -> None:
    state: ReviewState = {
        'submission': expense_claim(
            merchant='Harbour Kitchen and Grill',
            business_purpose='Leeds lunch for Halden Group',
            receipt_text='TOTAL 14.50',
        )
    }

    result = intake(state, graph_config(connection_pool, maximum_field_length=20))

    assert result['submission'] == expense_claim(
        merchant='Harbour Kitchen and ',
        business_purpose='Leeds lunch for Hald',
        receipt_text='TOTAL 14.50',
    )


async def test_intake_redacts_personal_data_and_records_which_fields_were_redacted(
    connection_pool: ConnectionPool,
) -> None:
    state: ReviewState = {
        'submission': expense_claim(
            business_purpose='Lunch for Halden Group, call 07700 900123',
            receipt_text='Paid 4111 1111\u200b 1111 1111\nTOTAL 14.50',
        )
    }

    result = intake(state, graph_config(connection_pool))

    assert result['submission'] == expense_claim(
        business_purpose='Lunch for Halden Group, call 07700 900123',
        receipt_text='Paid 4111 1111 1111 1111\nTOTAL 14.50',
    )
    assert result['redacted_submission'] == expense_claim(
        business_purpose='Lunch for Halden Group, call [PHONE]',
        receipt_text='Paid [CARD]\nTOTAL 14.50',
    )
    assert [entry for entry in result['audit'] if entry['event'] == 'field_redacted'] == [
        {'node': 'intake', 'event': 'field_redacted', 'detail': {'field': 'business_purpose'}},
        {'node': 'intake', 'event': 'field_redacted', 'detail': {'field': 'receipt_text'}},
    ]
