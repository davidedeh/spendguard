import uuid

from spendguard.graph.config import GraphConfig
from spendguard.graph.state import ReviewState
from spendguard.store import submissions


async def finalize(state: ReviewState, graph_config: GraphConfig) -> ReviewState:
    decision = state['decision']
    await submissions.move_status(
        graph_config.connection_pool,
        uuid.UUID(state['submission_id']),
        from_status='received' if decision['decided_by'] is None else 'decided',
        to_status='complete',
    )
    return {
        'audit': [
            {
                'node': 'finalize',
                'event': 'outcome_recorded',
                'detail': {'outcome': decision['outcome']},
            }
        ]
    }
