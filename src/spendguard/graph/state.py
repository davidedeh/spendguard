import operator
from collections.abc import Iterable
from typing import Annotated, Any, Literal, TypedDict

from spendguard.policy.loader import FindingCode, Severity
from spendguard.rules.decisions import Recommendation, Route
from spendguard.submission import SubmissionType

FindingSource = Literal['rule', 'judgement']
JudgementOutcomeKind = Literal['flagged', 'not_flagged', 'cleared_by_precedent']
DecisionOutcome = Literal['approve', 'reject']
DispositionVerdict = Literal['upheld', 'dismissed']


class Quote(TypedDict):
    field: str
    text: str


class RecordedFinding(TypedDict):
    finding_id: str
    code: FindingCode
    clause_reference: str
    severity: Severity
    source: FindingSource
    quotes: list[Quote]
    cited_precedent_ids: list[str]
    annotating_precedent_ids: list[str]


class CandidateFinding(TypedDict):
    code: str
    clause_reference: str
    quotes: list[Quote]
    cited_precedent_ids: list[str]


class FlaggedOutcome(TypedDict):
    code: FindingCode
    outcome: Literal['flagged']
    precedent_id: None
    candidate: CandidateFinding


class NotFlaggedOutcome(TypedDict):
    code: FindingCode
    outcome: Literal['not_flagged']
    precedent_id: None
    candidate: None


class PrecedentClearance(TypedDict):
    code: FindingCode
    outcome: Literal['cleared_by_precedent']
    precedent_id: str
    candidate: None


JudgementOutcome = FlaggedOutcome | NotFlaggedOutcome | PrecedentClearance


class RetrievalQuery(TypedDict):
    query: str


class CheckGuidance(TypedDict):
    code: FindingCode
    look_for: str


class RetrievedClause(TypedDict):
    clause_reference: str
    clause_hash: str
    text: str


class SuppliedPrecedent(TypedDict):
    precedent_id: str
    code: FindingCode
    clause_reference: str
    reason: str


class Disposition(TypedDict):
    finding_id: str
    verdict: DispositionVerdict
    reason: str


class Decision(TypedDict):
    outcome: DecisionOutcome
    decided_by: str | None
    reason: str | None
    dispositions: list[Disposition]


class AuditEntry(TypedDict):
    node: str
    event: str
    detail: dict[str, Any]


class ReviewState(TypedDict, total=False):
    submission_id: str
    idempotency_key: str
    submission_type: SubmissionType
    submission: dict[str, Any]
    redacted_submission: dict[str, Any]
    requester: str | None
    base_currency_amount: str
    retrieval_queries: list[RetrievalQuery]
    check_guidance: list[CheckGuidance]
    retrieved_clauses: list[RetrievedClause]
    judgement_checks: list[FindingCode]
    unanswered_checks: list[FindingCode]
    current_precedents: list[SuppliedPrecedent]
    supplied_precedents: Annotated[list[SuppliedPrecedent], operator.add]
    judgement_outcomes: list[JudgementOutcome]
    rule_findings: list[RecordedFinding]
    findings: Annotated[list[RecordedFinding], operator.add]
    route: Route
    recommendation: Recommendation
    decision: Decision
    audit: Annotated[list[AuditEntry], operator.add]


def precedent_clearances(outcomes: Iterable[JudgementOutcome]) -> list[PrecedentClearance]:
    return [outcome for outcome in outcomes if outcome['outcome'] == 'cleared_by_precedent']
