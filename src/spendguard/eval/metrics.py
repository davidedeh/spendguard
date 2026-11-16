import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from spendguard.eval.cases import EvalCase
from spendguard.policy.loader import CheckKind, FindingCode, Policy
from spendguard.rules.decisions import Recommendation, Route
from spendguard.settings import ModelPrice

NINETY_FIVE_PERCENT_Z = 1.959964
TOKENS_PER_MILLION = 1_000_000


@dataclass(frozen=True)
class Interval:
    low: float
    high: float


def wilson_interval(successes: int, trials: int) -> Interval | None:
    if trials == 0:
        return None
    z_squared = NINETY_FIVE_PERCENT_Z**2
    proportion = successes / trials
    denominator = 1 + z_squared / trials
    centre = (proportion + z_squared / (2 * trials)) / denominator
    half_width = (
        NINETY_FIVE_PERCENT_Z
        * math.sqrt(proportion * (1 - proportion) / trials + z_squared / (4 * trials**2))
        / denominator
    )
    return Interval(low=max(0.0, centre - half_width), high=min(1.0, centre + half_width))


@dataclass(frozen=True)
class Proportion:
    successes: int
    trials: int

    @property
    def interval(self) -> Interval | None:
        return wilson_interval(self.successes, self.trials)


@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    rule_codes: frozenset[FindingCode]
    judgement_codes: frozenset[FindingCode]
    route: Route
    recommendation: Recommendation
    proposed_judgement_results: int
    dropped_judgement_results: int
    retrieved_clause_references: frozenset[str]
    clearances: frozenset[tuple[FindingCode, str]]
    latency_seconds: float
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class RowMetrics:
    judgement_precision: Proportion
    judgement_recall: Proportion
    per_code: dict[FindingCode, Proportion]
    route_accuracy: Proportion
    recommendation_accuracy: Proportion
    verify_drop_rate: Proportion
    injection_survival: Proportion
    retrieval_recall: Proportion
    p50_latency_seconds: float
    cost_per_submission_usd: Decimal | None


@dataclass(frozen=True)
class RuleMetrics:
    precision: Proportion
    recall: Proportion
    per_code: dict[FindingCode, Proportion]


type ExpectedAndFound = tuple[frozenset[str], frozenset[str]]


def _overlap_over_expected(pairs: Sequence[ExpectedAndFound]) -> Proportion:
    return Proportion(
        sum(len(expected & found) for expected, found in pairs),
        sum(len(expected) for expected, _ in pairs),
    )


def _overlap_over_found(pairs: Sequence[ExpectedAndFound]) -> Proportion:
    return Proportion(
        sum(len(expected & found) for expected, found in pairs),
        sum(len(found) for _, found in pairs),
    )


def _expected_codes(case: EvalCase, policy: Policy, check: CheckKind) -> frozenset[FindingCode]:
    return frozenset(
        finding.code
        for finding in case.expected_findings
        if policy.finding_codes[finding.code].check == check
    )


def _found_codes(outcome: CaseOutcome, check: CheckKind) -> frozenset[FindingCode]:
    return outcome.judgement_codes if check == 'judgement' else outcome.rule_codes


def _paired(
    cases: Sequence[EvalCase], outcomes: Sequence[CaseOutcome]
) -> list[tuple[EvalCase, CaseOutcome]]:
    outcomes_by_case = {outcome.case_id: outcome for outcome in outcomes}
    return [(case, outcomes_by_case[case.id]) for case in cases]


def _codes_by_check(
    paired: Sequence[tuple[EvalCase, CaseOutcome]], policy: Policy, check: CheckKind
) -> list[ExpectedAndFound]:
    return [
        (_expected_codes(case, policy, check), _found_codes(outcome, check))
        for case, outcome in paired
    ]


def _per_code_counts(
    paired: Sequence[tuple[EvalCase, CaseOutcome]], policy: Policy, check: CheckKind
) -> dict[FindingCode, Proportion]:
    return {
        code: _overlap_over_expected(
            [
                (expected & {code}, found & {code})
                for expected, found in _codes_by_check(paired, policy, check)
            ]
        )
        for code in policy.codes_with_check(check)
    }


def _injection_survival(paired: Sequence[tuple[EvalCase, CaseOutcome]]) -> Proportion:
    return _overlap_over_expected(
        [
            (
                frozenset(finding.code for finding in case.expected_findings),
                outcome.rule_codes | outcome.judgement_codes,
            )
            for case, outcome in paired
            if case.slice == 'injection'
        ]
    )


def _retrieval_recall(paired: Sequence[tuple[EvalCase, CaseOutcome]], policy: Policy) -> Proportion:
    return _overlap_over_expected(
        [
            (
                frozenset(
                    clause
                    for finding in case.expected_findings
                    if policy.finding_codes[finding.code].check == 'judgement'
                    for clause in finding.gold_clauses
                ),
                outcome.retrieved_clause_references,
            )
            for case, outcome in paired
        ]
    )


def _cost_per_submission(outcomes: Sequence[CaseOutcome], price: ModelPrice) -> Decimal:
    total_cost = (
        sum(outcome.input_tokens for outcome in outcomes) * price.input_per_million_tokens_usd
        + sum(outcome.output_tokens for outcome in outcomes) * price.output_per_million_tokens_usd
    ) / TOKENS_PER_MILLION
    return total_cost / len(outcomes)


def row_metrics(
    cases: Sequence[EvalCase],
    outcomes: Sequence[CaseOutcome],
    policy: Policy,
    price: ModelPrice | None,
) -> RowMetrics:
    paired = _paired(cases, outcomes)
    judgement_codes = _codes_by_check(paired, policy, 'judgement')
    return RowMetrics(
        judgement_precision=_overlap_over_found(judgement_codes),
        judgement_recall=_overlap_over_expected(judgement_codes),
        per_code=_per_code_counts(paired, policy, 'judgement'),
        route_accuracy=Proportion(
            sum(case.expected_route == outcome.route for case, outcome in paired), len(paired)
        ),
        recommendation_accuracy=Proportion(
            sum(case.expected_recommendation == outcome.recommendation for case, outcome in paired),
            len(paired),
        ),
        verify_drop_rate=Proportion(
            sum(outcome.dropped_judgement_results for outcome in outcomes),
            sum(outcome.proposed_judgement_results for outcome in outcomes),
        ),
        injection_survival=_injection_survival(paired),
        retrieval_recall=_retrieval_recall(paired, policy),
        p50_latency_seconds=statistics.median(outcome.latency_seconds for outcome in outcomes),
        cost_per_submission_usd=None if price is None else _cost_per_submission(outcomes, price),
    )


@dataclass(frozen=True)
class MemoryMetrics:
    false_positive_rate: Proportion
    recommendation_agreement: Proportion
    cleared_with_pair_precedent: Proportion
    silent_clearances: Proportion


def memory_metrics(
    cases: Sequence[EvalCase],
    outcomes: Sequence[CaseOutcome],
    precedent_ids_by_pair: Mapping[str, frozenset[str]],
) -> MemoryMetrics:
    flagged = 0
    cleared = 0
    agreeing = 0
    follow_ups = 0
    for case, outcome in _paired(cases, outcomes):
        clearance = case.expected_clearance
        if clearance is None:
            continue
        follow_ups += 1
        agreeing += outcome.recommendation == case.expected_recommendation
        pair_precedent_ids = precedent_ids_by_pair.get(clearance.precedent_from, frozenset())
        if clearance.code in outcome.judgement_codes:
            flagged += 1
        elif any(
            code == clearance.code and precedent_id in pair_precedent_ids
            for code, precedent_id in outcome.clearances
        ):
            cleared += 1
    return MemoryMetrics(
        false_positive_rate=Proportion(flagged, follow_ups),
        recommendation_agreement=Proportion(agreeing, follow_ups),
        cleared_with_pair_precedent=Proportion(cleared, follow_ups),
        silent_clearances=Proportion(follow_ups - flagged - cleared, follow_ups),
    )


def rule_metrics(
    cases: Sequence[EvalCase], outcomes: Sequence[CaseOutcome], policy: Policy
) -> RuleMetrics:
    paired = _paired(cases, outcomes)
    rule_codes = _codes_by_check(paired, policy, 'rule')
    return RuleMetrics(
        precision=_overlap_over_found(rule_codes),
        recall=_overlap_over_expected(rule_codes),
        per_code=_per_code_counts(paired, policy, 'rule'),
    )
