import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from spendguard.graph.config import GraphConfig
from spendguard.graph.precedents import annotated_findings
from spendguard.graph.state import (
    CandidateFinding,
    JudgementOutcome,
    Quote,
    RecordedFinding,
    ReviewState,
)
from spendguard.policy.loader import FindingCode, Policy

DropReason = Literal[
    'code_not_judgement',
    'duplicate',
    'clause_not_retrieved',
    'quote_not_verbatim',
    'precedent_not_supplied',
]
_WHITESPACE_PATTERN = re.compile(r'\s+')


@dataclass(frozen=True)
class VerificationDrop:
    code: str
    clause_reference: str
    reason: DropReason


@dataclass(frozen=True)
class JudgementEvidence:
    judgement_checks: Collection[FindingCode]
    retrieved_references: Collection[str]
    redacted_submission: Mapping[str, Any]
    supplied_precedent_ids: Collection[str]


@dataclass(frozen=True)
class VerificationResult:
    findings: list[RecordedFinding]
    drops: list[VerificationDrop]


def _whitespace_normalised(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(' ', text).strip()


def _quote_is_verbatim(quote: Quote, redacted_submission: Mapping[str, Any]) -> bool:
    field_value = redacted_submission.get(quote['field'])
    quoted_text = _whitespace_normalised(quote['text'])
    return (
        isinstance(field_value, str)
        and quoted_text != ''
        and quoted_text in _whitespace_normalised(field_value)
    )


def _drop(candidate: CandidateFinding, reason: DropReason) -> VerificationDrop:
    return VerificationDrop(
        code=candidate['code'], clause_reference=candidate['clause_reference'], reason=reason
    )


def _drop_reason(
    candidate: CandidateFinding,
    kept_codes_and_clauses: Collection[tuple[str, str]],
    evidence: JudgementEvidence,
) -> DropReason | None:
    if (candidate['code'], candidate['clause_reference']) in kept_codes_and_clauses:
        return 'duplicate'
    if candidate['clause_reference'] not in evidence.retrieved_references:
        return 'clause_not_retrieved'
    if not all(
        _quote_is_verbatim(quote, evidence.redacted_submission) for quote in candidate['quotes']
    ):
        return 'quote_not_verbatim'
    if not all(
        precedent_id in evidence.supplied_precedent_ids
        for precedent_id in candidate['cited_precedent_ids']
    ):
        return 'precedent_not_supplied'
    return None


def verified_findings(
    candidates: Sequence[CandidateFinding], evidence: JudgementEvidence, policy: Policy
) -> VerificationResult:
    judgement_codes: dict[str, FindingCode] = {code: code for code in evidence.judgement_checks}
    findings: list[RecordedFinding] = []
    drops: list[VerificationDrop] = []
    for candidate in candidates:
        code = judgement_codes.get(candidate['code'])
        if code is None:
            drops.append(_drop(candidate, 'code_not_judgement'))
            continue
        reason = _drop_reason(
            candidate,
            [(finding['code'], finding['clause_reference']) for finding in findings],
            evidence,
        )
        if reason is not None:
            drops.append(_drop(candidate, reason))
            continue
        findings.append(
            {
                'finding_id': f'{code}@{candidate["clause_reference"]}',
                'code': code,
                'clause_reference': candidate['clause_reference'],
                'severity': policy.finding_codes[code].severity,
                'source': 'judgement',
                'quotes': candidate['quotes'],
                'cited_precedent_ids': candidate['cited_precedent_ids'],
                'annotating_precedent_ids': [],
            }
        )
    return VerificationResult(findings=findings, drops=drops)


def verified_clearances(
    outcomes: Sequence[JudgementOutcome], evidence: JudgementEvidence, policy: Policy
) -> tuple[list[JudgementOutcome], list[VerificationDrop]]:
    kept: list[JudgementOutcome] = []
    drops: list[VerificationDrop] = []
    for outcome in outcomes:
        reason: DropReason | None = None
        if outcome['outcome'] == 'cleared_by_precedent':
            if outcome['code'] not in evidence.judgement_checks:
                reason = 'code_not_judgement'
            elif outcome['precedent_id'] not in evidence.supplied_precedent_ids:
                reason = 'precedent_not_supplied'
        if reason is None:
            kept.append(outcome)
        else:
            drops.append(
                VerificationDrop(
                    code=outcome['code'],
                    clause_reference=policy.finding_codes[outcome['code']].clause.reference,
                    reason=reason,
                )
            )
    return kept, drops


def verify(state: ReviewState, graph_config: GraphConfig) -> ReviewState:
    evidence = JudgementEvidence(
        judgement_checks=state['judgement_checks'],
        retrieved_references=[clause['clause_reference'] for clause in state['retrieved_clauses']],
        redacted_submission=state['redacted_submission'],
        supplied_precedent_ids={
            precedent['precedent_id'] for precedent in state['supplied_precedents']
        },
    )
    outcomes = state['judgement_outcomes']
    verification = verified_findings(
        [outcome['candidate'] for outcome in outcomes if outcome['candidate'] is not None],
        evidence,
        graph_config.policy,
    )
    kept_outcomes, clearance_drops = verified_clearances(outcomes, evidence, graph_config.policy)
    return {
        'findings': annotated_findings(
            state['rule_findings'] + verification.findings, state['current_precedents']
        ),
        'judgement_outcomes': kept_outcomes,
        'audit': [
            {
                'node': 'verify',
                'event': 'finding_dropped',
                'detail': {
                    'code': drop.code,
                    'clause_reference': drop.clause_reference,
                    'reason': drop.reason,
                },
            }
            for drop in verification.drops + clearance_drops
        ]
        + [
            {
                'node': 'verify',
                'event': 'cleared_by_precedent',
                'detail': {'code': outcome['code'], 'precedent_id': outcome['precedent_id']},
            }
            for outcome in kept_outcomes
            if outcome['outcome'] == 'cleared_by_precedent'
        ],
    }
