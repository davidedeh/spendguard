import datetime
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import get_args

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from spendguard.eval.cases import SINGLE_FINDING_SLICES, EvalCase, Slice, SlicePlan, Split
from spendguard.policy.loader import CheckKind, FindingCode, Policy
from spendguard.seed.schema import SeedData
from spendguard.submission import ExpenseClaim, SupplierInvoice


@dataclass(frozen=True)
class DatasetReport:
    problems: tuple[str, ...]
    dataset_hash: str | None


def validate_dataset(
    cases: Mapping[str, object], policy: Policy, seed_data: SeedData, plan: SlicePlan
) -> DatasetReport:
    problems: list[str] = []
    parsed_cases: dict[str, EvalCase] = {}
    for source, content in cases.items():
        try:
            parsed_cases[source] = EvalCase.model_validate(content)
        except ValidationError as error:
            problems.extend(
                f'{source}: {describe_validation_error(detail)}' for detail in error.errors()
            )
    for source, case in parsed_cases.items():
        problems.extend(
            f'{source}: {problem}' for problem in case_problems(case, policy, seed_data)
        )
    problems.extend(_split_problems(parsed_cases))
    problems.extend(_slice_count_problems(cases, plan))
    if len(parsed_cases) == len(cases):
        problems.extend(_judgement_and_memory_problems(list(parsed_cases.values()), policy, plan))
        problems.extend(_rule_coverage_problems(list(parsed_cases.values()), policy, plan))
    if problems:
        return DatasetReport(problems=tuple(problems), dataset_hash=None)
    return DatasetReport(problems=(), dataset_hash=_dataset_hash(list(parsed_cases.values())))


def describe_validation_error(detail: ErrorDetails) -> str:
    location = '.'.join(str(part) for part in detail['loc'])
    if detail['type'] == 'literal_error' and detail['loc'][-1] == 'code':
        return f'{location}: unknown finding code {detail["input"]!r}'
    return f'{location}: {detail["msg"]}'


def case_problems(case: EvalCase, policy: Policy, seed_data: SeedData) -> Iterator[str]:
    clause_references = {clause.reference for clause in policy.clauses}
    for finding in case.expected_findings:
        for gold_clause in finding.gold_clauses:
            if gold_clause not in clause_references:
                yield f'gold clause {gold_clause} is not in the policy'
    if case.slice in SINGLE_FINDING_SLICES:
        check = SINGLE_FINDING_SLICES[case.slice]
        finding_count = len(_codes_with_check(case, policy, check))
        if finding_count != 1:
            yield (f'a {case.slice} case needs exactly one {check} finding, it has {finding_count}')
    submission = case.submission
    references: list[tuple[str, str | None]]
    if isinstance(submission, ExpenseClaim):
        yield from _category_problems(submission, policy)
        references = [('employee', submission.claimant)]
    else:
        references = [
            ('supplier', submission.supplier),
            ('purchase order', submission.purchase_order),
            ('cost centre', submission.cost_centre),
        ]
    yield from _currency_problems(submission, policy, seed_data)
    if case.scripted_review is not None:
        references.append(('approver', case.scripted_review.approver))
        finding_ids = {finding.id for finding in case.expected_findings}
        for disposition in case.scripted_review.dispositions:
            if disposition.finding_id not in finding_ids:
                yield (
                    f'a scripted disposition names finding {disposition.finding_id},'
                    ' which the case does not expect'
                )
    known_references = {
        'employee': {employee.reference for employee in seed_data.employees},
        'supplier': {supplier.reference for supplier in seed_data.suppliers},
        'purchase order': {order.reference for order in seed_data.purchase_orders},
        'cost centre': {centre.reference for centre in seed_data.cost_centres},
        'approver': {approver.employee for approver in seed_data.approvers},
    }
    for kind, reference in references:
        if reference is not None and reference not in known_references[kind]:
            yield f'unknown {kind} {reference}'


def _category_problems(claim: ExpenseClaim, policy: Policy) -> Iterator[str]:
    if claim.category not in policy.expense_categories:
        category_keys = ', '.join(sorted(policy.expense_categories))
        yield f'category {claim.category!r} is not a policy category ({category_keys})'


def _currency_problems(
    submission: ExpenseClaim | SupplierInvoice, policy: Policy, seed_data: SeedData
) -> Iterator[str]:
    problem = seed_data.company_rate_problem(
        submission.currency, submission.dated_on, policy.base_currency
    )
    if problem is not None:
        yield problem


def _split_problems(cases: Mapping[str, EvalCase]) -> Iterator[str]:
    sources_by_id: dict[str, list[str]] = defaultdict(list)
    for source, case in cases.items():
        sources_by_id[case.id].append(source)
    for case_id, sources in sources_by_id.items():
        if len(sources) > 1:
            listed_sources = ', '.join(sorted(sources))
            if len({cases[source].split for source in sources}) > 1:
                yield f'case {case_id} is in both splits ({listed_sources})'
            else:
                yield f'case id {case_id} is used more than once ({listed_sources})'
    test_cases = [(source, case) for source, case in cases.items() if case.split == 'test']
    dev_cases = [(source, case) for source, case in cases.items() if case.split == 'dev']
    for test_source, test_case in test_cases:
        for dev_source, dev_case in dev_cases:
            if dev_case.id != test_case.id and dev_case.submission == test_case.submission:
                first_source, second_source = sorted((test_source, dev_source))
                yield (
                    f'{first_source} and {second_source} hold the same submission in both splits'
                )


def _slice_count_problems(cases: Mapping[str, object], plan: SlicePlan) -> Iterator[str]:
    slice_counts = Counter(
        (content.get('split'), content.get('slice'))
        for content in cases.values()
        if isinstance(content, dict)
    )
    for split in get_args(Split):
        planned_counts = plan.for_split(split).slices.model_dump()
        for slice_name in get_args(Slice):
            planned_count = planned_counts[slice_name]
            count = slice_counts[split, slice_name]
            if count != planned_count:
                yield (
                    f'the {split} split has {count} {slice_name} cases,'
                    f' the plan says {planned_count}'
                )


def _judgement_and_memory_problems(
    cases: list[EvalCase], policy: Policy, plan: SlicePlan
) -> Iterator[str]:
    for split in get_args(Split):
        split_plan = plan.for_split(split)
        split_cases = [case for case in cases if case.split == split]
        code_counts = Counter(
            code
            for case in split_cases
            if case.slice == 'judgement_positive'
            for code in _codes_with_check(case, policy, 'judgement')
        )
        for code in policy.codes_with_check('judgement'):
            count = code_counts[code]
            if count != split_plan.judgement_positives_per_code:
                yield (
                    f'the {split} split has {count} judgement_positive cases for {code},'
                    f' the plan says {split_plan.judgement_positives_per_code}'
                )
        memory_pair_count = sum(case.scripted_review is not None for case in split_cases)
        if memory_pair_count != split_plan.memory_pairs:
            yield (
                f'the {split} split has {memory_pair_count} memory pairs,'
                f' the plan says {split_plan.memory_pairs}'
            )


def _rule_coverage_problems(
    cases: list[EvalCase], policy: Policy, plan: SlicePlan
) -> Iterator[str]:
    rule_codes = policy.codes_with_check('rule')
    for split in get_args(Split):
        planned_count = plan.for_split(split).slices.rule_coverage
        fewest, remainder = divmod(planned_count, len(rule_codes))
        most = fewest + (remainder > 0)
        code_counts = Counter(
            code
            for case in cases
            if case.split == split and case.slice == 'rule_coverage'
            for code in _codes_with_check(case, policy, 'rule')
        )
        for code in rule_codes:
            count = code_counts[code]
            if fewest <= count <= most:
                continue
            if fewest == most:
                expectation = f'the plan says {fewest}'
            elif count < fewest:
                expectation = f'the plan says at least {fewest}'
            else:
                expectation = f'the plan says at most {most}'
            yield f'the {split} split has {count} rule_coverage cases for {code}, {expectation}'


def _codes_with_check(case: EvalCase, policy: Policy, check: CheckKind) -> list[FindingCode]:
    return [
        finding.code
        for finding in case.expected_findings
        if policy.finding_codes[finding.code].check == check
    ]


def _dataset_hash(cases: list[EvalCase]) -> str:
    canonical_cases = sorted(
        json.dumps(case.model_dump(), sort_keys=True, default=_canonical_scalar) for case in cases
    )
    return hashlib.sha256('\n'.join(canonical_cases).encode()).hexdigest()


def _canonical_scalar(value: object) -> str:
    if isinstance(value, Decimal):
        return format(value.normalize(), 'f')
    if isinstance(value, datetime.date):
        return value.isoformat()
    raise TypeError(f'no canonical form for {type(value).__name__}')
