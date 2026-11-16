import uuid

from langgraph.types import interrupt

from spendguard.graph.config import GraphConfig
from spendguard.graph.state import ReviewState
from spendguard.store import submissions


async def human_review(state: ReviewState, graph_config: GraphConfig) -> ReviewState:
    await submissions.move_status(
        graph_config.connection_pool,
        uuid.UUID(state['submission_id']),
        from_status='received',
        to_status='pending_review',
    )
    interrupt({'submission_id': state['submission_id']})
    raise NotImplementedError('resuming after human review')
