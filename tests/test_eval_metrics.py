import datetime
from decimal import Decimal
from pathlib import Path
from typing import TypedDict, Unpack

import pytest
import yaml

from spendguard.eval.cases import EvalCase
from spendguard.eval.metrics import (
    CaseFailure,
    CaseOutcome,
    Proportion,
    memory_metrics,
    row_metrics,
    rule_metrics,
    wilson_interval,
)
from spendguard.policy.loader import FindingCode, load_policy
from spendguard.rules.decisions import Recommendation, Route
from spendguard.settings import ModelPrice

FIXTURES = Path(__file__).parent / 'fixtures'
FIXTURE_POLICY = load_policy(FIXTURES / 'policy' / 'valid_policy.yaml')
PRICE = ModelPrice(
    input_per_million_tokens_usd=Decimal('1.00'),
    output_per_million_tokens_usd=Decimal('5.00'),
    priced_on=datetime.date(2026, 11, 2),
)


def fixture_case(name: str) -> EvalCase:
    return EvalCase.model_validate(
        yaml.safe_load((FIXTURES / 'eval' / 'cases' / f'{name}.yaml').read_text())
    )


class OutcomeFields(TypedDict, total=False):
    rule_codes: tuple[FindingCode, ...]
    judgement_codes: tuple[FindingCode, ...]
    route: Route
    recommendation: Recommendation
    proposed: int
    dropped: int
    retrieved: tuple[str, ...]
    latency_seconds: float
    output_tokens: int
    clearances: tuple[tuple[FindingCode, str], ...]
    unanswered: tuple[FindingCode, ...]


def counts(proportion: Proportion) -> tuple[int, int]:
    return proportion.successes, proportion.trials


def outcome(case_id: str, **fields: Unpack[OutcomeFields]) -> CaseOutcome:
    return CaseOutcome(
        case_id=case_id,
        rule_codes=frozenset(fields.get('rule_codes', ())),
        judgement_codes=frozenset(fields.get('judgement_codes', ())),
        route=fields['route'],
        recommendation=fields['recommendation'],
        proposed_judgement_results=fields['proposed'],
        dropped_judgement_results=fields['dropped'],
        retrieved_clause_references=frozenset(fields['retrieved']),
        clearances=frozenset(fields.get('clearances', ())),
        unanswered_judgement_codes=frozenset(fields.get('unanswered', ())),
        latency_seconds=fields['latency_seconds'],
        input_tokens=1000,
        output_tokens=fields['output_tokens'],
    )


CASES = tuple(
    fixture_case(name)
    for name in (
        'test-alcohol',
        'test-weekend',
        'test-hard-negative',
        'test-injection',
        'test-missing-po',
    )
)
OUTCOMES = (
    outcome(
        'test-alcohol',
        judgement_codes=('ALCOHOL_NOT_PERMITTED', 'PURPOSE_UNCLEAR'),
        route='human_review',
        recommendation='none',
        proposed=2,
        dropped=0,
        retrieved=('§2.2', '§1.1'),
        latency_seconds=1.0,
        output_tokens=100,
    ),
    outcome(
        'test-weekend',
        route='auto_approve',
        recommendation='approve',
        proposed=0,
        dropped=0,
        retrieved=('§1.1',),
        latency_seconds=4.0,
        output_tokens=200,
    ),
    outcome(
        'test-hard-negative',
        rule_codes=('MISSING_RECEIPT',),
        judgement_codes=('PURPOSE_UNCLEAR',),
        route='human_review',
        recommendation='none',
        proposed=1,
        dropped=1,
        retrieved=(),
        latency_seconds=2.0,
        output_tokens=0,
    ),
    outcome(
        'test-injection',
        judgement_codes=('RECEIPT_MISMATCH',),
        route='human_review',
        recommendation='reject',
        proposed=3,
        dropped=1,
        retrieved=('§2.2',),
        latency_seconds=8.0,
        output_tokens=0,
    ),
    outcome(
        'test-missing-po',
        rule_codes=('MISSING_PO',),
        route='human_review',
        recommendation='none',
        proposed=0,
        dropped=0,
        retrieved=(),
        latency_seconds=3.0,
        output_tokens=300,
    ),
)


def test_the_wilson_interval_for_13_of_16_matches_the_hand_computed_bounds() -> None:
    interval = wilson_interval(13, 16)

    assert interval is not None
    assert interval.low == pytest.approx(0.5699, abs=5e-5)
    assert interval.high == pytest.approx(0.9341, abs=5e-5)


def test_the_wilson_interval_for_none_of_4_starts_at_zero() -> None:
    interval = wilson_interval(0, 4)

    assert interval is not None
    assert interval.low == pytest.approx(0.0, abs=5e-5)
    assert interval.high == pytest.approx(0.4899, abs=5e-5)


def test_there_is_no_wilson_interval_without_trials() -> None:
    assert wilson_interval(0, 0) is None


def test_judgement_precision_and_recall_pool_judgement_findings_outside_injection_cases() -> None:
    metrics = row_metrics(CASES, OUTCOMES, FIXTURE_POLICY, price=PRICE)

    assert counts(metrics.judgement_precision) == (1, 3)
    assert counts(metrics.judgement_recall) == (1, 2)
    precision_interval = metrics.judgement_precision.interval
    recall_interval = metrics.judgement_recall.interval
    assert precision_interval is not None
    assert recall_interval is not None
    assert (precision_interval.low, precision_interval.high) == pytest.approx(
        (0.0615, 0.7923), abs=5e-5
    )
    assert (recall_interval.low, recall_interval.high) == pytest.approx((0.0945, 0.9055), abs=5e-5)


def test_per_code_counts_give_found_over_expected_outside_injection_cases() -> None:
    metrics = row_metrics(CASES, OUTCOMES, FIXTURE_POLICY, price=PRICE)

    assert {code: counts(count) for code, count in metrics.per_code.items()} == {
        'WEEKEND_SPEND_UNJUSTIFIED': (0, 1),
        'PURPOSE_UNCLEAR': (0, 0),
        'RECEIPT_MISMATCH': (0, 0),
        'ALCOHOL_NOT_PERMITTED': (1, 1),
    }


def test_route_and_recommendation_accuracy_count_cases_matching_their_labels() -> None:
    metrics = row_metrics(CASES, OUTCOMES, FIXTURE_POLICY, price=PRICE)

    assert (metrics.route_accuracy.successes, metrics.route_accuracy.trials) == (3, 5)
    assert (
        metrics.recommendation_accuracy.successes,
        metrics.recommendation_accuracy.trials,
    ) == (2, 5)


def test_the_verify_drop_rate_is_dropped_over_proposed_model_results() -> None:
    metrics = row_metrics(CASES, OUTCOMES, FIXTURE_POLICY, price=PRICE)

    assert (metrics.verify_drop_rate.successes, metrics.verify_drop_rate.trials) == (2, 6)


def test_injection_survival_counts_expected_findings_found_in_injection_cases_only() -> None:
    metrics = row_metrics(CASES, OUTCOMES, FIXTURE_POLICY, price=PRICE)

    assert (metrics.injection_survival.successes, metrics.injection_survival.trials) == (0, 1)


def test_retrieval_recall_counts_retrieved_gold_clauses_of_expected_judgement_findings() -> None:
    metrics = row_metrics(CASES, OUTCOMES, FIXTURE_POLICY, price=PRICE)

    assert (metrics.retrieval_recall.successes, metrics.retrieval_recall.trials) == (2, 3)


def test_latency_is_the_median_and_cost_is_tokens_times_prices_per_submission() -> None:
    metrics = row_metrics(CASES, OUTCOMES, FIXTURE_POLICY, price=PRICE)

    assert metrics.p50_latency_seconds == 3.0
    assert metrics.cost_per_submission_usd == Decimal('0.0016')


def test_a_failed_case_counts_as_missed_and_wrong_but_adds_no_findings_latency_or_cost() -> None:
    failure = CaseFailure(
        case_id='test-alcohol',
        error_type='RuntimeError',
        error_message='provider down',
        reason='RuntimeError',
    )

    metrics = row_metrics(CASES, (failure, *OUTCOMES[1:]), FIXTURE_POLICY, price=PRICE)

    assert counts(metrics.judgement_precision) == (0, 1)
    assert counts(metrics.judgement_recall) == (0, 2)
    assert counts(metrics.per_code['ALCOHOL_NOT_PERMITTED']) == (0, 1)
    assert counts(metrics.route_accuracy) == (2, 5)
    assert counts(metrics.recommendation_accuracy) == (1, 5)
    assert counts(metrics.verify_drop_rate) == (2, 4)
    assert counts(metrics.retrieval_recall) == (1, 3)
    assert metrics.p50_latency_seconds == 3.5
    assert metrics.cost_per_submission_usd == Decimal('0.001625')
    assert len(metrics.failed_cases) == 1


def test_a_row_whose_every_case_failed_has_no_latency_or_cost() -> None:
    failures = tuple(
        CaseFailure(
            case_id=case.id,
            error_type='RuntimeError',
            error_message='provider down',
            reason='RuntimeError',
        )
        for case in CASES
    )

    metrics = row_metrics(CASES, failures, FIXTURE_POLICY, price=PRICE)

    assert metrics.p50_latency_seconds is None
    assert metrics.cost_per_submission_usd is None
    assert len(metrics.failed_cases) == 5
    assert counts(metrics.route_accuracy) == (0, 5)


def test_rule_metrics_score_rule_findings_against_rule_labels() -> None:
    metrics = rule_metrics(CASES, OUTCOMES, FIXTURE_POLICY)

    assert (metrics.precision.successes, metrics.precision.trials) == (1, 2)
    assert (metrics.recall.successes, metrics.recall.trials) == (1, 1)
    assert (metrics.per_code['MISSING_PO'].successes, metrics.per_code['MISSING_PO'].trials) == (
        1,
        1,
    )
    assert 'ALCOHOL_NOT_PERMITTED' not in metrics.per_code


def follow_up_outcome(
    case_id: str,
    recommendation: Recommendation,
    judgement_codes: tuple[FindingCode, ...] = (),
    clearances: tuple[tuple[FindingCode, str], ...] = (),
    unanswered: tuple[FindingCode, ...] = (),
) -> CaseOutcome:
    return outcome(
        case_id,
        judgement_codes=judgement_codes,
        clearances=clearances,
        unanswered=unanswered,
        route='human_review',
        recommendation=recommendation,
        proposed=1,
        dropped=0,
        retrieved=(),
        latency_seconds=1.0,
        output_tokens=0,
    )


def test_a_follow_up_is_cleared_by_its_pair_flagged_forced_to_review_or_silently_cleared() -> None:
    follow_up = fixture_case('test-memory-follow-up')
    follow_up_ids = (
        'cleared',
        'flagged',
        'flagged-on-another-code',
        'not-flagged',
        'not-flagged-again',
        'cleared-by-another',
        'no-verdict',
    )
    cases = (
        fixture_case('test-alcohol'),
        *(follow_up.model_copy(update={'id': case_id}) for case_id in follow_up_ids),
    )
    outcomes = (
        OUTCOMES[0],
        follow_up_outcome('cleared', 'approve', clearances=(('PURPOSE_UNCLEAR', 'pair-1'),)),
        follow_up_outcome('flagged', 'none', judgement_codes=('PURPOSE_UNCLEAR',)),
        follow_up_outcome(
            'flagged-on-another-code', 'none', judgement_codes=('ALCOHOL_NOT_PERMITTED',)
        ),
        follow_up_outcome('not-flagged', 'approve'),
        follow_up_outcome('not-flagged-again', 'approve'),
        follow_up_outcome(
            'cleared-by-another', 'approve', clearances=(('PURPOSE_UNCLEAR', 'elsewhere'),)
        ),
        follow_up_outcome('no-verdict', 'none', unanswered=('PURPOSE_UNCLEAR',)),
    )

    metrics = memory_metrics(cases, outcomes, {'test-purpose-unclear': frozenset({'pair-1'})})

    assert counts(metrics.false_positive_rate) == (2, 7)
    assert counts(metrics.recommendation_agreement) == (4, 7)
    assert counts(metrics.cleared_with_pair_precedent) == (1, 7)
    assert counts(metrics.forced_review) == (1, 7)
    assert counts(metrics.silent_clearances) == (3, 7)


def test_a_failed_follow_up_is_counted_as_failed_and_disagreeing_not_as_a_silent_clearance() -> (
    None
):
    follow_up = fixture_case('test-memory-follow-up')
    cases = tuple(
        follow_up.model_copy(update={'id': case_id})
        for case_id in ('failed', 'failed-again', 'not-flagged')
    )
    results = (
        CaseFailure(
            case_id='failed',
            error_type='RuntimeError',
            error_message='provider down',
            reason='RuntimeError',
        ),
        CaseFailure(
            case_id='failed-again',
            error_type='TimeoutError',
            error_message='timed out',
            reason='TimeoutError',
        ),
        follow_up_outcome('not-flagged', 'approve'),
    )

    metrics = memory_metrics(cases, results, {})

    assert counts(metrics.failed) == (2, 3)
    assert counts(metrics.silent_clearances) == (1, 3)
    assert counts(metrics.recommendation_agreement) == (1, 3)
    assert counts(metrics.false_positive_rate) == (0, 3)
