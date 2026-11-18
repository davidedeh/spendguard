import math
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
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
    unanswered_judgement_codes: frozenset[FindingCode]
    latency_seconds: float
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class CaseFailure:
    case_id: str
    error_type: str
    error_message: str
    reason: str


type CaseResult = CaseOutcome | CaseFailure
type PairedResult = tuple[EvalCase, CaseOutcome | None]


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
    p50_latency_seconds: float | None
    cost_per_submission_usd: Decimal | None
    failed_cases: tuple[CaseFailure, ...]


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


def _found_codes(outcome: CaseOutcome | None, check: CheckKind) -> frozenset[FindingCode]:
    if outcome is None:
        return frozenset()
    return outcome.judgement_codes if check == 'judgement' else outcome.rule_codes


def _paired(cases: Sequence[EvalCase], results: Sequence[CaseResult]) -> list[PairedResult]:
    results_by_case = {result.case_id: result for result in results}
    return [
        (case, result if isinstance(result, CaseOutcome) else None)
        for case, result in ((case, results_by_case[case.id]) for case in cases)
    ]


def completed_outcomes(results: Sequence[CaseResult]) -> list[CaseOutcome]:
    return [result for result in results if isinstance(result, CaseOutcome)]


def _codes_by_check(
    paired: Sequence[PairedResult], policy: Policy, check: CheckKind
) -> list[ExpectedAndFound]:
    return [
        (_expected_codes(case, policy, check), _found_codes(outcome, check))
        for case, outcome in paired
    ]


def _per_code_counts(
    paired: Sequence[PairedResult], policy: Policy, check: CheckKind
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


def _injection_survival(paired: Sequence[PairedResult]) -> Proportion:
    return _overlap_over_expected(
        [
            (
                frozenset(finding.code for finding in case.expected_findings),
                _found_codes(outcome, 'rule') | _found_codes(outcome, 'judgement'),
            )
            for case, outcome in paired
            if case.slice == 'injection'
        ]
    )


def _retrieval_recall(paired: Sequence[PairedResult], policy: Policy) -> Proportion:
    return _overlap_over_expected(
        [
            (
                frozenset(
                    clause
                    for finding in case.expected_findings
                    if policy.finding_codes[finding.code].check == 'judgement'
                    for clause in finding.gold_clauses
                ),
                frozenset() if outcome is None else outcome.retrieved_clause_references,
            )
            for case, outcome in paired
        ]
    )


def _cost_per_submission(outcomes: Sequence[CaseOutcome], price: ModelPrice) -> Decimal | None:
    if not outcomes:
        return None
    total_cost = (
        sum(outcome.input_tokens for outcome in outcomes) * price.input_per_million_tokens_usd
        + sum(outcome.output_tokens for outcome in outcomes) * price.output_per_million_tokens_usd
    ) / TOKENS_PER_MILLION
    return total_cost / len(outcomes)


def _matching(
    paired: Sequence[PairedResult], matches: Callable[[EvalCase, CaseOutcome], bool]
) -> Proportion:
    return Proportion(
        sum(outcome is not None and matches(case, outcome) for case, outcome in paired),
        len(paired),
    )


def judgement_pool(cases: Iterable[EvalCase]) -> list[EvalCase]:
    return [case for case in cases if case.slice != 'injection']


def judgement_positive_count(cases: Iterable[EvalCase], policy: Policy) -> int:
    return sum(bool(_expected_codes(case, policy, 'judgement')) for case in judgement_pool(cases))


def row_metrics(
    cases: Sequence[EvalCase],
    results: Sequence[CaseResult],
    policy: Policy,
    price: ModelPrice,
) -> RowMetrics:
    paired = _paired(cases, results)
    pooled_case_ids = {case.id for case in judgement_pool(cases)}
    pooled = [(case, outcome) for case, outcome in paired if case.id in pooled_case_ids]
    judgement_codes = _codes_by_check(pooled, policy, 'judgement')
    outcomes = completed_outcomes(results)
    return RowMetrics(
        judgement_precision=_overlap_over_found(judgement_codes),
        judgement_recall=_overlap_over_expected(judgement_codes),
        per_code=_per_code_counts(pooled, policy, 'judgement'),
        route_accuracy=_matching(
            paired, lambda case, outcome: case.expected_route == outcome.route
        ),
        recommendation_accuracy=_matching(
            paired, lambda case, outcome: case.expected_recommendation == outcome.recommendation
        ),
        verify_drop_rate=Proportion(
            sum(outcome.dropped_judgement_results for outcome in outcomes),
            sum(outcome.proposed_judgement_results for outcome in outcomes),
        ),
        injection_survival=_injection_survival(paired),
        retrieval_recall=_retrieval_recall(paired, policy),
        p50_latency_seconds=(
            statistics.median(outcome.latency_seconds for outcome in outcomes) if outcomes else None
        ),
        cost_per_submission_usd=_cost_per_submission(outcomes, price),
        failed_cases=tuple(result for result in results if isinstance(result, CaseFailure)),
    )


@dataclass(frozen=True)
class MemoryMetrics:
    raised_labelled_code: Proportion
    raised_any_judgement_code: Proportion
    cleared_with_pair_precedent: Proportion
    forced_review: Proportion
    failed: Proportion
    silent_clearances: Proportion
    controls_raised_labelled_code: Proportion
    controls_cleared_by_precedent: Proportion
    controls_failed: Proportion
    recommendation_agreement: Proportion
    other_findings: tuple[tuple[str, tuple[FindingCode, ...]], ...]


def memory_metrics(
    cases: Sequence[EvalCase],
    results: Sequence[CaseResult],
    precedent_id_by_follow_up: Mapping[str, str],
) -> MemoryMetrics:
    flagged = 0
    flagged_on_labelled_code = 0
    cleared = 0
    forced = 0
    failed = 0
    follow_ups = 0
    controls_flagged = 0
    controls_cleared = 0
    controls_failed = 0
    controls = 0
    agreeing = 0
    other_findings: list[tuple[str, tuple[FindingCode, ...]]] = []
    for case, outcome in _paired(cases, results):
        memory = case.memory
        if memory is None:
            continue
        other_codes = (
            () if outcome is None else tuple(sorted(outcome.judgement_codes - {memory.code}))
        )
        if other_codes:
            other_findings.append((case.id, other_codes))
        agreeing += outcome is not None and outcome.recommendation == case.expected_recommendation
        cleared_by_seeded_precedent = outcome is not None and any(
            code == memory.code and precedent_id == precedent_id_by_follow_up.get(case.id)
            for code, precedent_id in outcome.clearances
        )
        if not memory.precedent_applies:
            controls += 1
            if outcome is None:
                controls_failed += 1
            elif memory.code in outcome.judgement_codes:
                controls_flagged += 1
            elif cleared_by_seeded_precedent:
                controls_cleared += 1
            continue
        follow_ups += 1
        if outcome is None:
            failed += 1
            continue
        flagged_on_labelled_code += memory.code in outcome.judgement_codes
        if outcome.judgement_codes:
            flagged += 1
        elif cleared_by_seeded_precedent:
            cleared += 1
        elif memory.code in outcome.unanswered_judgement_codes:
            forced += 1
    return MemoryMetrics(
        raised_labelled_code=Proportion(flagged_on_labelled_code, follow_ups),
        raised_any_judgement_code=Proportion(flagged, follow_ups),
        cleared_with_pair_precedent=Proportion(cleared, follow_ups),
        forced_review=Proportion(forced, follow_ups),
        failed=Proportion(failed, follow_ups),
        silent_clearances=Proportion(follow_ups - flagged - cleared - forced - failed, follow_ups),
        controls_raised_labelled_code=Proportion(controls_flagged, controls),
        controls_cleared_by_precedent=Proportion(controls_cleared, controls),
        controls_failed=Proportion(controls_failed, controls),
        recommendation_agreement=Proportion(agreeing, follow_ups + controls),
        other_findings=tuple(other_findings),
    )


def rule_metrics(
    cases: Sequence[EvalCase], results: Sequence[CaseResult], policy: Policy
) -> RuleMetrics:
    paired = _paired(cases, results)
    rule_codes = _codes_by_check(paired, policy, 'rule')
    return RuleMetrics(
        precision=_overlap_over_found(rule_codes),
        recall=_overlap_over_expected(rule_codes),
        per_code=_per_code_counts(paired, policy, 'rule'),
    )
