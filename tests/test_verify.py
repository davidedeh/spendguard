from typing import Any

import pytest

from spendguard.graph.nodes.verify import (
    DropReason,
    JudgementEvidence,
    VerificationDrop,
    verified_clearances,
    verified_findings,
)
from spendguard.graph.state import CandidateFinding, JudgementOutcome, Quote
from spendguard.policy.loader import POLICY_PATH, FindingCode, load_policy

POLICY = load_policy(POLICY_PATH)
REDACTED_SUBMISSION: dict[str, Any] = {
    'type': 'expense_claim',
    'claimant': 'EMP-003',
    'quantity': 1,
    'business_purpose': 'Saturday dinner with the Leeds team, call [PHONE]',
    'receipt_text': 'Harbour Kitchen\n1 x Fish pie 24.00\n1 x Pint of   lager 14.00\nTOTAL 38.00',
}


def reference_of(clause_id: str) -> str:
    return next(clause.reference for clause in POLICY.clauses if clause.id == clause_id)


ALCOHOL_CLAUSE = reference_of('no_alcohol')
PURPOSE_CLAUSE = reference_of('business_purpose')
RETRIEVED = [ALCOHOL_CLAUSE, PURPOSE_CLAUSE]
JUDGEMENT_CHECKS: list[FindingCode] = [
    'PURPOSE_UNCLEAR',
    'RECEIPT_MISMATCH',
    'ALCOHOL_NOT_PERMITTED',
]


def evidence(
    retrieved_references: list[str] = RETRIEVED, supplied_precedent_ids: list[str] | None = None
) -> JudgementEvidence:
    return JudgementEvidence(
        judgement_checks=JUDGEMENT_CHECKS,
        retrieved_references=retrieved_references,
        redacted_submission=REDACTED_SUBMISSION,
        supplied_precedent_ids=supplied_precedent_ids or [],
    )


def alcohol_candidate(
    code: str = 'ALCOHOL_NOT_PERMITTED',
    clause_reference: str = ALCOHOL_CLAUSE,
    quotes: list[Quote] | None = None,
    cited_precedent_ids: list[str] | None = None,
) -> CandidateFinding:
    return {
        'code': code,
        'clause_reference': clause_reference,
        'quotes': [{'field': 'receipt_text', 'text': 'Pint of lager'}]
        if quotes is None
        else quotes,
        'cited_precedent_ids': cited_precedent_ids or [],
    }


def test_a_finding_that_passes_every_check_is_kept_with_its_severity_from_the_policy() -> None:
    result = verified_findings(
        [alcohol_candidate()],
        evidence(),
        POLICY,
    )

    assert result.findings == [
        {
            'finding_id': f'ALCOHOL_NOT_PERMITTED@{ALCOHOL_CLAUSE}',
            'code': 'ALCOHOL_NOT_PERMITTED',
            'clause_reference': ALCOHOL_CLAUSE,
            'severity': 'medium',
            'source': 'judgement',
            'quotes': [{'field': 'receipt_text', 'text': 'Pint of lager'}],
            'cited_precedent_ids': [],
            'annotating_precedent_ids': [],
        }
    ]
    assert result.drops == []


@pytest.mark.parametrize(
    ('candidate', 'reason'),
    [
        (alcohol_candidate(code='MISSING_RECEIPT'), 'code_not_judgement'),
        (alcohol_candidate(code='WEEKEND_SPEND_UNJUSTIFIED'), 'code_not_judgement'),
        (alcohol_candidate(code='LATE_CLAIM'), 'code_not_judgement'),
        (alcohol_candidate(clause_reference=reference_of('tips')), 'clause_not_retrieved'),
        (alcohol_candidate(clause_reference='§99.1'), 'clause_not_retrieved'),
        (
            alcohol_candidate(quotes=[{'field': 'receipt_text', 'text': 'a pint of beer'}]),
            'quote_not_verbatim',
        ),
        (
            alcohol_candidate(quotes=[{'field': 'business_purpose', 'text': 'Pint of lager'}]),
            'quote_not_verbatim',
        ),
        (
            alcohol_candidate(quotes=[{'field': 'merchant_notes', 'text': 'Pint of lager'}]),
            'quote_not_verbatim',
        ),
        (alcohol_candidate(quotes=[{'field': 'quantity', 'text': '1'}]), 'quote_not_verbatim'),
        (alcohol_candidate(quotes=[{'field': 'receipt_text', 'text': '  '}]), 'quote_not_verbatim'),
        (
            alcohol_candidate(
                quotes=[
                    {'field': 'receipt_text', 'text': 'Pint of lager'},
                    {'field': 'business_purpose', 'text': 'call 07700 900123'},
                ]
            ),
            'quote_not_verbatim',
        ),
        (alcohol_candidate(cited_precedent_ids=['PRECEDENT-2']), 'precedent_not_supplied'),
    ],
    ids=[
        'a rule code',
        'a judgement check not run for this submission',
        'a code outside the closed set',
        'a clause that was not retrieved',
        'a clause the policy does not have',
        'a paraphrased quote',
        'a quote from another field',
        'a quote from a field the submission does not have',
        'a quote from a field that is not text',
        'a blank quote',
        'a quote of the text before redaction',
        'a precedent that was not supplied',
    ],
)
def test_a_finding_failing_one_check_is_dropped_with_that_reason(
    candidate: CandidateFinding, reason: DropReason
) -> None:
    result = verified_findings(
        [candidate],
        evidence(supplied_precedent_ids=['PRECEDENT-1']),
        POLICY,
    )

    assert result.findings == []
    assert result.drops == [
        VerificationDrop(
            code=candidate['code'], clause_reference=candidate['clause_reference'], reason=reason
        )
    ]


def test_findings_sharing_a_code_and_clause_collapse_into_the_first_and_the_rest_are_audited() -> (
    None
):
    first = alcohol_candidate()
    repeated = alcohol_candidate(quotes=[{'field': 'receipt_text', 'text': 'lager 14.00'}])
    under_another_clause = alcohol_candidate(clause_reference=PURPOSE_CLAUSE)

    result = verified_findings(
        [first, repeated, under_another_clause, repeated],
        evidence(),
        POLICY,
    )

    assert [(finding['clause_reference'], finding['quotes']) for finding in result.findings] == [
        (ALCOHOL_CLAUSE, first['quotes']),
        (PURPOSE_CLAUSE, under_another_clause['quotes']),
    ]
    assert (
        result.drops
        == [
            VerificationDrop(
                code='ALCOHOL_NOT_PERMITTED', clause_reference=ALCOHOL_CLAUSE, reason='duplicate'
            )
        ]
        * 2
    )


def test_severity_comes_from_the_policy_for_each_code() -> None:
    receipt_clause = reference_of('receipt_matches_claim')

    result = verified_findings(
        [
            alcohol_candidate(code='RECEIPT_MISMATCH', clause_reference=receipt_clause),
            alcohol_candidate(),
        ],
        evidence(retrieved_references=[*RETRIEVED, receipt_clause]),
        POLICY,
    )

    assert [(finding['code'], finding['severity']) for finding in result.findings] == [
        ('RECEIPT_MISMATCH', 'high'),
        ('ALCOHOL_NOT_PERMITTED', 'medium'),
    ]


def clearance(code: FindingCode, precedent_id: str) -> JudgementOutcome:
    return {
        'code': code,
        'outcome': 'cleared_by_precedent',
        'precedent_id': precedent_id,
        'candidate': None,
    }


def test_a_clearance_is_kept_only_for_a_check_run_and_a_precedent_supplied() -> None:
    supported = clearance('PURPOSE_UNCLEAR', 'PRECEDENT-1')
    not_flagged: JudgementOutcome = {
        'code': 'RECEIPT_MISMATCH',
        'outcome': 'not_flagged',
        'precedent_id': None,
        'candidate': None,
    }

    kept, drops = verified_clearances(
        [
            supported,
            clearance('PURPOSE_UNCLEAR', 'PRECEDENT-2'),
            not_flagged,
            clearance('WEEKEND_SPEND_UNJUSTIFIED', 'PRECEDENT-1'),
        ],
        evidence(supplied_precedent_ids=['PRECEDENT-1']),
        POLICY,
    )

    assert kept == [supported, not_flagged]
    assert drops == [
        VerificationDrop(
            code='PURPOSE_UNCLEAR', clause_reference=PURPOSE_CLAUSE, reason='precedent_not_supplied'
        ),
        VerificationDrop(
            code='WEEKEND_SPEND_UNJUSTIFIED',
            clause_reference=reference_of('weekend_and_holiday_spend'),
            reason='code_not_judgement',
        ),
    ]
