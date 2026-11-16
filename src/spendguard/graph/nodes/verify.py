import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from spendguard.graph.config import GraphConfig
from spendguard.graph.state import CandidateFinding, Quote, RecordedFinding, ReviewState
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
    retrieved_references: Collection[str],
    redacted_submission: Mapping[str, Any],
    supplied_precedent_ids: Collection[str],
) -> DropReason | None:
    if (candidate['code'], candidate['clause_reference']) in kept_codes_and_clauses:
        return 'duplicate'
    if candidate['clause_reference'] not in retrieved_references:
        return 'clause_not_retrieved'
    if not all(_quote_is_verbatim(quote, redacted_submission) for quote in candidate['quotes']):
        return 'quote_not_verbatim'
    if not all(
        precedent_id in supplied_precedent_ids for precedent_id in candidate['cited_precedent_ids']
    ):
        return 'precedent_not_supplied'
    return None


def verified_findings(
    candidates: Sequence[CandidateFinding],
    *,
    retrieved_references: Collection[str],
    redacted_submission: Mapping[str, Any],
    supplied_precedent_ids: Collection[str],
    policy: Policy,
) -> VerificationResult:
    judgement_codes: dict[str, FindingCode] = {
        code: code for code in policy.codes_with_check('judgement')
    }
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
            retrieved_references,
            redacted_submission,
            supplied_precedent_ids,
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


def verify(state: ReviewState, graph_config: GraphConfig) -> ReviewState:
    return {}
