from collections.abc import Sequence

from spendguard.graph.config import GraphConfig
from spendguard.graph.state import RecordedFinding, SuppliedPrecedent
from spendguard.policy.loader import PrecedentScope
from spendguard.store import precedents
from spendguard.submission import ExpenseClaim, Submission


def precedent_subject(
    scope: PrecedentScope | None, submission: Submission, requester: str
) -> str | None:
    match scope:
        case 'employee':
            return requester
        case 'supplier':
            return None if isinstance(submission, ExpenseClaim) else submission.supplier
        case None:
            return None


async def current_precedents(
    graph_config: GraphConfig, submission: Submission, requester: str
) -> list[SuppliedPrecedent]:
    found: list[SuppliedPrecedent] = []
    for code, definition in graph_config.policy.finding_codes.items():
        subject = precedent_subject(definition.precedent_scope, submission, requester)
        if subject is not None:
            found.extend(
                await precedents.recent_precedents(
                    graph_config.connection_pool, subject, code, graph_config.policy
                )
            )
    return found


def annotated_findings(
    findings: Sequence[RecordedFinding], available: Sequence[SuppliedPrecedent]
) -> list[RecordedFinding]:
    return [
        finding
        | {
            'annotating_precedent_ids': [
                precedent['precedent_id']
                for precedent in available
                if precedent['code'] == finding['code']
            ]
        }
        for finding in findings
    ]
