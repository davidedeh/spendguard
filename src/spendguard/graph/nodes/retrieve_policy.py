from spendguard.graph.config import GraphConfig
from spendguard.graph.state import ReviewState
from spendguard.retrieval.policy_index import retrieve_clauses


async def retrieve_policy(state: ReviewState, graph_config: GraphConfig) -> ReviewState:
    async with graph_config.connection_pool.connection() as connection:
        retrieved_clauses = await retrieve_clauses(
            connection,
            graph_config.policy,
            graph_config.embedder,
            [query['query'] for query in state['retrieval_queries']],
            graph_config.settings.retrieval_top_k,
        )
    return {
        'retrieved_clauses': retrieved_clauses,
        'audit': [
            {
                'node': 'retrieve_policy',
                'event': 'policy_retrieved',
                'detail': {
                    'clause_references': [
                        clause['clause_reference'] for clause in retrieved_clauses
                    ]
                },
            }
        ],
    }
