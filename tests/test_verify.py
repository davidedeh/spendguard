from typing import Any

import pytest

from spendguard.graph.nodes.verify import DropReason, VerificationDrop, verified_findings
from spendguard.graph.state import CandidateFinding, Quote
from spendguard.policy.loader import POLICY_PATH, load_policy

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
        retrieved_references=RETRIEVED,
        redacted_submission=REDACTED_SUBMISSION,
        supplied_precedent_ids=[],
        policy=POLICY,
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
        retrieved_references=RETRIEVED,
        redacted_submission=REDACTED_SUBMISSION,
        supplied_precedent_ids=['PRECEDENT-1'],
        policy=POLICY,
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
        retrieved_references=RETRIEVED,
        redacted_submission=REDACTED_SUBMISSION,
        supplied_precedent_ids=[],
        policy=POLICY,
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
        retrieved_references=[*RETRIEVED, receipt_clause],
        redacted_submission=REDACTED_SUBMISSION,
        supplied_precedent_ids=[],
        policy=POLICY,
    )

    assert [(finding['code'], finding['severity']) for finding in result.findings] == [
        ('RECEIPT_MISMATCH', 'high'),
        ('ALCOHOL_NOT_PERMITTED', 'medium'),
    ]
