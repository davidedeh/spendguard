import uuid
from decimal import Decimal

from spendguard.graph.config import GraphConfig
from spendguard.graph.review_payload import review_payload
from spendguard.graph.state import ReviewState
from spendguard.rules import decisions
from spendguard.rules.checks import Finding
from spendguard.rules.decisions import ReviewedFinding
from spendguard.store import submissions


async def route(state: ReviewState, graph_config: GraphConfig) -> ReviewState:
    reviewed_findings = [
        ReviewedFinding(
            finding=Finding(
                code=finding['code'],
                clause_reference=finding['clause_reference'],
                severity=finding['severity'],
            ),
            has_precedent=bool(finding['annotating_precedent_ids']),
        )
        for finding in state.get('findings', [])
    ]
    chosen_route = decisions.route(
        findings=[reviewed.finding for reviewed in reviewed_findings],
        cleared_by_precedent=any(
            outcome['outcome'] == 'cleared_by_precedent'
            for outcome in state.get('judgement_outcomes', [])
        ),
        base_amount=Decimal(state['base_currency_amount']),
        policy=graph_config.policy,
        force_human_review=graph_config.settings.force_human_review
        or bool(state['unanswered_checks']),
    )
    recommendation = decisions.recommend(reviewed_findings)
    await submissions.record_review(
        graph_config.connection_pool,
        uuid.UUID(state['submission_id']),
        review_payload(state | {'recommendation': recommendation}, graph_config.policy),
    )
    if chosen_route == 'human_review':
        return {'route': chosen_route, 'recommendation': recommendation}
    return {
        'route': chosen_route,
        'recommendation': recommendation,
        'decision': {'outcome': 'approve', 'decided_by': None, 'reason': None, 'dispositions': []},
    }
