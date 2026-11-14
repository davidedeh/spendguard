import datetime
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import assert_never, get_args

from pydantic import ValidationError
from pydantic_core import ErrorDetails

from spendguard.eval.cases import (
    HISTORY_CODES,
    SINGLE_FINDING_SLICES,
    EvalCase,
    ExpectedFinding,
    InjectionField,
    InjectionPlan,
    Slice,
    SlicePlan,
    Split,
)
from spendguard.policy.loader import CheckKind, FindingCode, Policy
from spendguard.rules.checks import (
    WEEKEND_CODE,
    base_currency_amount,
    falls_on_weekend_or_holiday,
    finding_for,
    raised_rule_codes,
)
from spendguard.rules.decisions import Recommendation, ReviewedFinding, Route, recommend, route
from spendguard.seed.schema import SUPPLIER_REFERENCE_PATTERN, SeedData
from spendguard.submission import ExpenseClaim, Submission, SupplierInvoice

PLACEHOLDER_TEXTS = ('Merchant Name', 'Item 1', 'Item Description', 'XYZ')
PLACEHOLDER_PATTERNS = {
    placeholder: re.compile(rf'\b{re.escape(placeholder)}\b', re.IGNORECASE)
    for placeholder in PLACEHOLDER_TEXTS
}
MERCHANT_SUPPLIER_REFERENCE_PATTERN = re.compile(
    rf'\b{SUPPLIER_REFERENCE_PATTERN}\b', re.IGNORECASE
)


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
        problems.extend(f'{source}: {problem}' for problem in intent_problems(case, policy))
    problems.extend(_split_problems(parsed_cases))
    problems.extend(_repeated_submission_problems(parsed_cases))
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
        yield from _supplier_reference_problems(submission)
        references = [('employee', submission.claimant)]
    else:
        references = [
            ('supplier', submission.supplier),
            ('purchase order', submission.purchase_order),
            ('cost centre', submission.cost_centre),
        ]
    yield from _currency_problems(submission, policy, seed_data)
    yield from _placeholder_problems(submission)
    yield from _injection_plan_problems(case)
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
    unknown_references = [
        f'unknown {kind} {reference}'
        for kind, reference in references
        if reference is not None and reference not in known_references[kind]
    ]
    yield from unknown_references
    if not unknown_references:
        yield from _label_problems(case, policy, seed_data)


def _category_problems(claim: ExpenseClaim, policy: Policy) -> Iterator[str]:
    if claim.category not in policy.expense_categories:
        category_keys = ', '.join(sorted(policy.expense_categories))
        yield f'category {claim.category!r} is not a policy category ({category_keys})'


def _placeholder_problems(submission: Submission) -> Iterator[str]:
    for field_name, text in _free_text_fields(submission).items():
        for placeholder, pattern in PLACEHOLDER_PATTERNS.items():
            if text is not None and pattern.search(text):
                yield (
                    f'{field_name} holds the placeholder {placeholder!r}:'
                    ' write what a real submission would say instead'
                )


def _supplier_reference_problems(claim: ExpenseClaim) -> Iterator[str]:
    match = MERCHANT_SUPPLIER_REFERENCE_PATTERN.search(claim.merchant)
    if match is not None:
        yield (
            f'merchant holds the supplier reference {match.group()}: give the name the merchant'
            ' trades under'
        )


def _free_text_fields(submission: Submission) -> dict[str, str | None]:
    if isinstance(submission, ExpenseClaim):
        return {
            'merchant': submission.merchant,
            'business_purpose': submission.business_purpose,
            'receipt_text': submission.receipt_text,
        }
    return {
        'description': submission.description,
        'supplier_invoice_number': submission.supplier_invoice_number,
    }


def _injection_plan_problems(case: EvalCase) -> Iterator[str]:
    if case.intent.injection is not None:
        yield from _injection_problems(case.submission, case.intent.injection)
    elif case.slice == 'injection':
        yield 'an injection case needs an injected instruction in its intent'


def _injection_problems(submission: Submission, injection: InjectionPlan) -> Iterator[str]:
    text = _free_text(submission, injection.field)
    if text is None or injection.payload not in text:
        yield (
            f'{injection.field} does not contain the injected instruction word for word: put'
            f' the payload from the brief in {injection.field} exactly as given'
        )


def _free_text(submission: Submission, field_name: InjectionField) -> str | None:
    match field_name:
        case 'business_purpose':
            return submission.business_purpose if isinstance(submission, ExpenseClaim) else None
        case 'receipt_text':
            return submission.receipt_text if isinstance(submission, ExpenseClaim) else None
        case 'description':
            return submission.description if isinstance(submission, SupplierInvoice) else None
        case _:
            assert_never(field_name)


def _currency_problems(
    submission: ExpenseClaim | SupplierInvoice, policy: Policy, seed_data: SeedData
) -> Iterator[str]:
    problem = seed_data.company_rate_problem(
        submission.currency, submission.dated_on, policy.base_currency
    )
    if problem is not None:
        yield problem
    if submission.currency == policy.base_currency and submission.claimed_exchange_rate is not None:
        yield (f'a {policy.base_currency} submission states a claimed exchange rate; leave it null')


def _label_problems(case: EvalCase, policy: Policy, seed_data: SeedData) -> Iterator[str]:
    submission = case.submission
    if seed_data.company_rate_problem(
        submission.currency, submission.dated_on, policy.base_currency
    ):
        return
    computed_rule_codes = raised_rule_codes(submission, seed_data, policy)
    labelled_rule_codes = sorted(_codes_with_check(case, policy, 'rule'))
    if computed_rule_codes != labelled_rule_codes:
        yield (
            f'the rule checks give {describe_rule_codes(computed_rule_codes)},'
            f' the case expects {describe_rule_codes(labelled_rule_codes)}'
        )
    labelled_codes = {finding.code for finding in case.expected_findings}
    if WEEKEND_CODE in labelled_codes and not falls_on_weekend_or_holiday(
        submission.dated_on, seed_data
    ):
        yield (
            f'{WEEKEND_CODE} needs a weekend or public holiday, and'
            f' {submission.dated_on.isoformat()} is a working day'
        )
    yield from _history_copy_problems(case, seed_data)
    expected_route, expected_recommendation = computed_route_and_recommendation(
        case.expected_findings, submission, seed_data, policy
    )
    if case.expected_route != expected_route:
        yield f'the route rule gives {expected_route}, not {case.expected_route}'
    if case.expected_recommendation != expected_recommendation:
        yield (
            f'the recommendation rule gives {expected_recommendation} with no precedents,'
            f' not {case.expected_recommendation}'
        )


def computed_route_and_recommendation(
    expected_findings: Sequence[ExpectedFinding],
    submission: Submission,
    seed_data: SeedData,
    policy: Policy,
) -> tuple[Route, Recommendation]:
    findings = [finding_for(finding.code, policy) for finding in expected_findings]
    expected_route = route(
        findings=findings,
        cleared_by_precedent=False,
        base_amount=base_currency_amount(submission, seed_data, policy),
        policy=policy,
        force_human_review=False,
    )
    expected_recommendation = recommend(
        [ReviewedFinding(finding, has_precedent=False) for finding in findings]
    )
    return expected_route, expected_recommendation


def intent_problems(case: EvalCase, policy: Policy) -> Iterator[str]:
    allowed_judgement_codes = set(case.intent.judgement_codes)
    for code in _codes_with_check(case, policy, 'judgement'):
        if code not in allowed_judgement_codes:
            allowed = ', '.join(case.intent.judgement_codes) or 'no judgement findings'
            yield f'judgement finding {code} is outside the case brief, which allows {allowed}'


def describe_rule_codes(codes: Sequence[FindingCode]) -> str:
    return ', '.join(codes) if codes else 'no rule findings'


def _history_copy_problems(case: EvalCase, seed_data: SeedData) -> Iterator[str]:
    if HISTORY_CODES & set(case.intent.rule_codes):
        return
    rows: list[tuple[str, str, tuple[str, datetime.date, Decimal]]] = [
        ('historical claim', claim.reference, (claim.claimant, claim.spent_on, claim.amount))
        for claim in seed_data.historical_claims
    ] + [
        (
            'historical invoice',
            invoice.reference,
            (invoice.supplier, invoice.issued_on, invoice.amount),
        )
        for invoice in seed_data.historical_invoices
    ]
    key = _party_date_amount(case)
    for kind, reference, row_key in rows:
        if row_key == key:
            yield (
                f'copies {kind} {reference} without targeting'
                ' DUPLICATE_SUBMISSION or SPLIT_PURCHASE'
            )


def _party_date_amount(case: EvalCase) -> tuple[str, datetime.date, Decimal]:
    submission = case.submission
    party = submission.claimant if isinstance(submission, ExpenseClaim) else submission.supplier
    return (party, submission.dated_on, submission.amount)


def _repeated_submission_problems(cases: Mapping[str, EvalCase]) -> Iterator[str]:
    sources_by_key: dict[tuple[str, datetime.date, Decimal], list[str]] = defaultdict(list)
    for source, case in cases.items():
        sources_by_key[_party_date_amount(case)].append(source)
    for sources in sources_by_key.values():
        submissions = {cases[source].submission.model_dump_json() for source in sources}
        splits = {cases[source].split for source in sources}
        if len(submissions) == 1 and len(splits) > 1:
            continue
        if len(sources) > 1:
            yield (
                f'{" and ".join(sorted(sources))} have the same claimant or supplier,'
                ' date and amount'
            )


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
