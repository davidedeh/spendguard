from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from spendguard.policy.loader import Policy
from spendguard.rules.checks import Finding

Route = Literal['auto_approve', 'human_review']
Recommendation = Literal['approve', 'reject', 'none']


@dataclass(frozen=True)
class ReviewedFinding:
    finding: Finding
    has_precedent: bool


def route(
    *,
    findings: Sequence[Finding],
    cleared_by_precedent: bool,
    base_amount: Decimal,
    policy: Policy,
    force_human_review: bool,
) -> Route:
    if (
        findings
        or cleared_by_precedent
        or base_amount > policy.auto_approve_threshold
        or force_human_review
    ):
        return 'human_review'
    return 'auto_approve'


def recommend(findings: Sequence[ReviewedFinding]) -> Recommendation:
    if any(
        reviewed.finding.severity == 'high' and not reviewed.has_precedent for reviewed in findings
    ):
        return 'reject'
    if all(reviewed.has_precedent for reviewed in findings):
        return 'approve'
    return 'none'
