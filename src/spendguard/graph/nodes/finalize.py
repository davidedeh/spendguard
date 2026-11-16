import uuid

from spendguard.graph.config import GraphConfig
from spendguard.graph.nodes.intake import SUBMISSION_ADAPTER
from spendguard.graph.precedents import precedent_subject
from spendguard.graph.state import ReviewState
from spendguard.store import precedents, submissions
from spendguard.store.precedents import NewPrecedent


async def _record_precedents(state: ReviewState, graph_config: GraphConfig) -> None:
    decision = state['decision']
    approver = decision['decided_by']
    requester = state['requester']
    if approver is None or requester is None:
        return
    submission = SUBMISSION_ADAPTER.validate_python(state['submission'])
    findings = {finding['finding_id']: finding for finding in state['findings']}
    clause_hashes = {clause.reference: clause.hash for clause in graph_config.policy.clauses}
    for disposition in decision['dispositions']:
        finding = findings[disposition['finding_id']]
        subject = precedent_subject(
            graph_config.policy.finding_codes[finding['code']].precedent_scope,
            submission,
            requester,
        )
        if disposition['verdict'] != 'dismissed' or subject is None:
            continue
        if disposition['reason'] is None:
            raise ValueError(f'Dismissal of {finding["finding_id"]} has no reason')
        await precedents.record_precedent(
            graph_config.connection_pool,
            NewPrecedent(
                subject=subject,
                code=finding['code'],
                clause_reference=finding['clause_reference'],
                clause_hash=clause_hashes[finding['clause_reference']],
                reason=disposition['reason'],
                approver=approver,
                submission_id=uuid.UUID(state['submission_id']),
            ),
        )


async def finalize(state: ReviewState, graph_config: GraphConfig) -> ReviewState:
    decision = state['decision']
    await _record_precedents(state, graph_config)
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
