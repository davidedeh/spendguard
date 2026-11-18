from spendguard.graph.config import GraphConfig
from spendguard.graph.state import ReviewState
from spendguard.retrieval.policy_index import supplied_clause


async def supply_whole_policy(state: ReviewState, graph_config: GraphConfig) -> ReviewState:
    return {
        'retrieved_clauses': [supplied_clause(clause) for clause in graph_config.policy.clauses],
        'audit': [
            {
                'node': 'supply_whole_policy',
                'event': 'whole_policy_supplied',
                'detail': {'policy_version': graph_config.policy.version},
            }
        ],
    }
