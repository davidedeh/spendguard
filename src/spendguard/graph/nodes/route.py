from decimal import Decimal

from spendguard.graph.config import GraphConfig
from spendguard.graph.state import ReviewState
from spendguard.rules import decisions
from spendguard.rules.checks import Finding


def route(state: ReviewState, graph_config: GraphConfig) -> ReviewState:
    chosen_route = decisions.route(
        findings=[
            Finding(
                code=finding['code'],
                clause_reference=finding['clause_reference'],
                severity=finding['severity'],
            )
            for finding in state.get('findings', [])
        ],
        cleared_by_precedent=any(
            outcome['outcome'] == 'cleared_by_precedent'
            for outcome in state.get('judgement_outcomes', [])
        ),
        base_amount=Decimal(state['base_currency_amount']),
        policy=graph_config.policy,
        force_human_review=graph_config.settings.force_human_review,
    )
    if chosen_route == 'human_review':
        return {'route': chosen_route}
    return {
        'route': chosen_route,
        'decision': {'outcome': 'approve', 'decided_by': None, 'reason': None, 'dispositions': []},
    }
