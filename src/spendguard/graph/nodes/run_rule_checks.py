from spendguard.graph.config import GraphConfig
from spendguard.graph.nodes.intake import SUBMISSION_ADAPTER
from spendguard.graph.state import RecordedFinding, ReviewState
from spendguard.rules.checks import rule_findings


def run_rule_checks(state: ReviewState, graph_config: GraphConfig) -> ReviewState:
    submission = SUBMISSION_ADAPTER.validate_python(state['submission'])
    findings: list[RecordedFinding] = [
        {
            'finding_id': finding.code,
            'code': finding.code,
            'clause_reference': finding.clause_reference,
            'severity': finding.severity,
            'source': 'rule',
            'quotes': [],
            'cited_precedent_ids': [],
            'annotating_precedent_ids': [],
        }
        for finding in rule_findings(submission, graph_config.seed_data, graph_config.policy)
    ]
    return {'findings': findings}
