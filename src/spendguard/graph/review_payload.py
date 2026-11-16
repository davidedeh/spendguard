from typing import TypedDict

from spendguard.graph.state import FindingSource, Quote, ReviewState, SuppliedPrecedent
from spendguard.policy.loader import FindingCode, Policy, Severity
from spendguard.rules.decisions import Recommendation
from spendguard.submission import SubmissionType


class ClauseView(TypedDict):
    reference: str
    heading_path: list[str]
    text: str


class FindingView(TypedDict):
    finding_id: str
    code: FindingCode
    severity: Severity
    source: FindingSource
    clause: ClauseView
    quotes: list[Quote]
    precedents: list[SuppliedPrecedent]


class ClearanceView(TypedDict):
    code: FindingCode
    precedent: SuppliedPrecedent


class ReviewPayload(TypedDict):
    submission_id: str
    submission_type: SubmissionType
    requester: str | None
    base_currency_amount: str
    recommendation: Recommendation
    findings: list[FindingView]
    clearances_by_precedent: list[ClearanceView]


def review_payload(state: ReviewState, policy: Policy) -> ReviewPayload:
    clauses = {clause.reference: clause for clause in policy.clauses}
    precedents = {
        precedent['precedent_id']: precedent for precedent in state.get('current_precedents', [])
    }
    return {
        'submission_id': state['submission_id'],
        'submission_type': state['submission_type'],
        'requester': state['requester'],
        'base_currency_amount': state['base_currency_amount'],
        'recommendation': state['recommendation'],
        'findings': [
            {
                'finding_id': finding['finding_id'],
                'code': finding['code'],
                'severity': finding['severity'],
                'source': finding['source'],
                'clause': {
                    'reference': finding['clause_reference'],
                    'heading_path': list(clauses[finding['clause_reference']].heading_path),
                    'text': clauses[finding['clause_reference']].text,
                },
                'quotes': finding['quotes'],
                'precedents': [
                    precedents[precedent_id] for precedent_id in finding['annotating_precedent_ids']
                ],
            }
            for finding in state.get('findings', [])
        ],
        'clearances_by_precedent': [
            {'code': outcome['code'], 'precedent': precedents[outcome['precedent_id']]}
            for outcome in state.get('judgement_outcomes', [])
            if outcome['outcome'] == 'cleared_by_precedent' and outcome['precedent_id'] is not None
        ],
    }
