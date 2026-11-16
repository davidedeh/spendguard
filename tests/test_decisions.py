from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from spendguard.policy.loader import load_policy
from spendguard.rules.checks import Finding
from spendguard.rules.decisions import Recommendation, ReviewedFinding, recommend, route

POLICY = load_policy(Path(__file__).parent / 'fixtures' / 'policy' / 'valid_policy.yaml')
MEDIUM = Finding(code='MISSING_PO', clause_reference='§3.2', severity='medium')
HIGH = Finding(code='DUPLICATE_SUBMISSION', clause_reference='§3.4', severity='high')
LOW = Finding(code='CURRENCY_ANOMALY', clause_reference='§3.5', severity='low')
AUTO_APPROVABLE: dict[str, Any] = {
    'findings': (),
    'cleared_by_precedent': False,
    'base_amount': Decimal('250.00'),
    'force_human_review': False,
}


@pytest.mark.parametrize(
    ('switched_off', 'expected'),
    [
        ({}, 'auto_approve'),
        ({'findings': (LOW,)}, 'human_review'),
        ({'cleared_by_precedent': True}, 'human_review'),
        ({'base_amount': Decimal('250.01')}, 'human_review'),
        ({'force_human_review': True}, 'human_review'),
    ],
)
def test_route_auto_approves_only_while_every_condition_holds(
    switched_off: dict[str, Any], expected: str
) -> None:
    arguments = {**AUTO_APPROVABLE, **switched_off}

    assert route(policy=POLICY, **arguments) == expected


@pytest.mark.parametrize(
    ('findings', 'expected'),
    [
        ((ReviewedFinding(HIGH, has_precedent=False),), 'reject'),
        (
            (
                ReviewedFinding(HIGH, has_precedent=False),
                ReviewedFinding(MEDIUM, has_precedent=True),
            ),
            'reject',
        ),
        ((ReviewedFinding(HIGH, has_precedent=True),), 'approve'),
        (
            (ReviewedFinding(HIGH, has_precedent=True), ReviewedFinding(LOW, has_precedent=True)),
            'approve',
        ),
        ((), 'approve'),
        ((ReviewedFinding(MEDIUM, has_precedent=False),), 'none'),
        (
            (ReviewedFinding(HIGH, has_precedent=True), ReviewedFinding(LOW, has_precedent=False)),
            'none',
        ),
    ],
)
def test_the_recommendation_takes_the_first_rule_that_matches(
    findings: tuple[ReviewedFinding, ...], expected: Recommendation
) -> None:
    assert recommend(findings) == expected
