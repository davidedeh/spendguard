import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from spendguard.eval.cases import load_slice_plan, read_case_directory
from spendguard.eval.validator import DatasetReport, validate_dataset
from spendguard.policy.loader import load_policy
from spendguard.seed.loader import read_seed_directory

FIXTURES = Path(__file__).parent / 'fixtures'
FIXTURE_CASE_DIRECTORY = FIXTURES / 'eval' / 'cases'
FIXTURE_SLICE_PLAN = load_slice_plan(FIXTURES / 'eval' / 'slice_plan.yaml')


FIXTURE_POLICY = load_policy(FIXTURES / 'policy' / 'valid_policy.yaml')
FIXTURE_SEED_DATA = read_seed_directory(FIXTURES / 'seed')


def test_the_fixture_dataset_has_no_problems_and_a_hash() -> None:
    report = validate(read_case_directory(FIXTURE_CASE_DIRECTORY))

    assert report.problems == ()
    assert report.dataset_hash is not None
    assert len(report.dataset_hash) == 64


def fixture_cases() -> dict[str, Any]:
    return copy.deepcopy(read_case_directory(FIXTURE_CASE_DIRECTORY))


def validate(cases: Mapping[str, object]) -> DatasetReport:
    return validate_dataset(cases, FIXTURE_POLICY, FIXTURE_SEED_DATA, FIXTURE_SLICE_PLAN)


def test_an_unknown_finding_code_is_reported() -> None:
    cases = fixture_cases()
    cases['test-alcohol.yaml']['expected_findings'][0]['code'] = 'ALCOHOL_BANNED'

    report = validate(cases)

    assert report.problems == (
        "test-alcohol.yaml: expected_findings.0.code: unknown finding code 'ALCOHOL_BANNED'",
    )
    assert report.dataset_hash is None


def test_a_case_without_drafted_by_is_reported() -> None:
    cases = fixture_cases()
    del cases['test-weekend.yaml']['drafted_by']

    report = validate(cases)

    assert report.problems == ('test-weekend.yaml: drafted_by: Field required',)


@pytest.mark.parametrize(
    ('quantity', 'problem'),
    [
        (None, 'test-weekend.yaml: submission.expense_claim.quantity: Field required'),
        (
            0,
            'test-weekend.yaml: submission.expense_claim.quantity:'
            ' Input should be greater than or equal to 1',
        ),
        (
            1.5,
            'test-weekend.yaml: submission.expense_claim.quantity:'
            ' Input should be a valid integer, got a number with a fractional part',
        ),
    ],
)
def test_an_expense_claim_without_a_whole_quantity_of_at_least_one_is_reported(
    quantity: float | None, problem: str
) -> None:
    cases = fixture_cases()
    if quantity is None:
        del cases['test-weekend.yaml']['submission']['quantity']
    else:
        cases['test-weekend.yaml']['submission']['quantity'] = quantity

    report = validate(cases)

    assert report.problems == (problem,)


def test_an_expense_claim_whose_category_is_not_a_policy_category_key_is_reported() -> None:
    cases = fixture_cases()
    cases['test-weekend.yaml']['submission']['category'] = 'travel'

    report = validate(cases)

    assert report.problems == (
        "test-weekend.yaml: category 'travel' is not a policy category"
        ' (accommodation, subsistence, taxi)',
    )
    assert report.dataset_hash is None


def test_a_gold_clause_missing_from_the_policy_is_reported() -> None:
    cases = fixture_cases()
    cases['test-alcohol.yaml']['expected_findings'][0]['gold_clauses'] = ['§2.2', '§2.9']

    report = validate(cases)

    assert report.problems == ('test-alcohol.yaml: gold clause §2.9 is not in the policy',)
    assert report.dataset_hash is None


@pytest.mark.parametrize(
    ('source', 'path', 'value', 'problem'),
    [
        ('test-alcohol.yaml', ('submission', 'claimant'), 'EMP-999', 'unknown employee EMP-999'),
        ('test-clean.yaml', ('submission', 'supplier'), 'SUP-999', 'unknown supplier SUP-999'),
        (
            'test-clean.yaml',
            ('submission', 'purchase_order'),
            'PO-9999',
            'unknown purchase order PO-9999',
        ),
        (
            'test-missing-po.yaml',
            ('submission', 'cost_centre'),
            'CC-999',
            'unknown cost centre CC-999',
        ),
        (
            'test-purpose-unclear.yaml',
            ('scripted_review', 'approver'),
            'EMP-003',
            'unknown approver EMP-003',
        ),
    ],
)
def test_a_reference_missing_from_the_seed_data_is_reported(
    source: str,
    path: tuple[str, str],
    value: str,
    problem: str,
) -> None:
    cases = fixture_cases()
    cases[source][path[0]][path[1]] = value

    report = validate(cases)

    assert report.problems == (f'{source}: {problem}',)


def test_a_foreign_currency_claim_dated_after_the_last_rate_is_reported() -> None:
    cases = fixture_cases()
    cases['dev-clean.yaml']['submission']['spent_on'] = '2026-10-02'

    report = validate(cases)

    assert report.problems == ('dev-clean.yaml: no company rate for EUR on 2026-10-02',)
    assert report.dataset_hash is None


def test_a_base_currency_claim_needs_no_company_rate() -> None:
    cases = fixture_cases()
    cases['test-alcohol.yaml']['submission']['spent_on'] = '2026-10-02'

    report = validate(cases)

    assert report.problems == ()


def test_a_foreign_currency_claim_dated_before_the_first_rate_is_reported() -> None:
    cases = fixture_cases()
    cases['dev-clean.yaml']['submission']['spent_on'] = '2026-09-30'

    report = validate(cases)

    assert report.problems == ('dev-clean.yaml: no company rate for EUR on 2026-09-30',)


@pytest.mark.parametrize(
    ('issued_on', 'problems'),
    [
        ('2026-10-01', ()),
        ('2026-10-02', ('test-clean.yaml: no company rate for USD on 2026-10-02',)),
    ],
)
def test_a_foreign_currency_invoice_needs_a_company_rate_on_its_issue_date(
    issued_on: str, problems: tuple[str, ...]
) -> None:
    cases = fixture_cases()
    cases['test-clean.yaml']['submission']['currency'] = 'USD'
    cases['test-clean.yaml']['submission']['issued_on'] = issued_on

    report = validate(cases)

    assert report.problems == problems


def test_a_case_in_both_splits_is_reported() -> None:
    cases = fixture_cases()
    del cases['dev-clean.yaml']
    cases['dev-copy.yaml'] = {**cases['test-clean.yaml'], 'split': 'dev'}

    report = validate(cases)

    assert report.problems == (
        'case test-clean is in both splits (dev-copy.yaml, test-clean.yaml)',
    )


def test_the_same_submission_under_another_id_in_the_other_split_is_reported() -> None:
    cases = fixture_cases()
    del cases['dev-clean.yaml']
    cases['dev-copy.yaml'] = {**cases['test-clean.yaml'], 'id': 'dev-copy', 'split': 'dev'}

    report = validate(cases)

    assert report.problems == (
        'dev-copy.yaml and test-clean.yaml hold the same submission in both splits',
    )


def test_a_case_id_used_twice_in_one_split_is_reported() -> None:
    cases = fixture_cases()
    cases['test-weekend.yaml']['id'] = 'test-alcohol'

    report = validate(cases)

    assert report.problems == (
        'case id test-alcohol is used more than once (test-alcohol.yaml, test-weekend.yaml)',
    )


def test_a_slice_count_off_the_plan_is_reported() -> None:
    cases = fixture_cases()
    del cases['test-hard-negative.yaml']
    cases['test-missing-po.yaml']['slice'] = 'clean'

    report = validate(cases)

    assert report.problems == (
        'the test split has 0 hard_negative cases, the plan says 1',
        'the test split has 0 rule_coverage cases, the plan says 1',
        'the test split has 2 clean cases, the plan says 1',
    )


def test_judgement_positives_off_the_plan_for_a_code_are_reported() -> None:
    cases = fixture_cases()
    cases['test-weekend.yaml']['expected_findings'] = [
        {'id': 'purpose', 'code': 'PURPOSE_UNCLEAR', 'gold_clauses': ['§1.1']}
    ]

    report = validate(cases)

    assert report.problems == (
        'the test split has 0 judgement_positive cases for WEEKEND_SPEND_UNJUSTIFIED,'
        ' the plan says 1',
        'the test split has 2 judgement_positive cases for PURPOSE_UNCLEAR, the plan says 1',
    )


def test_a_judgement_positive_without_one_judgement_finding_is_reported() -> None:
    cases = fixture_cases()
    cases['test-weekend.yaml']['expected_findings'] = [
        {'id': 'receipt', 'code': 'MISSING_RECEIPT', 'gold_clauses': ['§1.2']}
    ]

    report = validate(cases)

    assert report.problems == (
        'test-weekend.yaml: a judgement_positive case needs exactly one judgement finding,'
        ' it has 0',
        'the test split has 0 judgement_positive cases for WEEKEND_SPEND_UNJUSTIFIED,'
        ' the plan says 1',
    )


def test_memory_pairs_off_the_plan_are_reported() -> None:
    cases = fixture_cases()
    del cases['test-purpose-unclear.yaml']['scripted_review']

    report = validate(cases)

    assert report.problems == ('the test split has 0 memory pairs, the plan says 1',)


def test_a_disposition_for_a_finding_the_case_does_not_expect_is_reported() -> None:
    cases = fixture_cases()
    cases['test-purpose-unclear.yaml']['scripted_review']['dispositions'][0]['finding_id'] = 'gone'

    report = validate(cases)

    assert report.problems == (
        'test-purpose-unclear.yaml: a scripted disposition names finding gone,'
        ' which the case does not expect',
    )


def test_several_faults_are_all_reported_in_one_run() -> None:
    cases = fixture_cases()
    del cases['test-weekend.yaml']['drafted_by']
    cases['test-alcohol.yaml']['submission']['claimant'] = 'EMP-999'
    cases['test-injection.yaml']['expected_findings'][0]['gold_clauses'] = ['§7.1']
    cases['test-clean.yaml']['split'] = 'dev'
    cases['dev-clean.yaml']['submission']['spent_on'] = '2026-10-02'

    report = validate(cases)

    assert report.problems == (
        'test-weekend.yaml: drafted_by: Field required',
        'dev-clean.yaml: no company rate for EUR on 2026-10-02',
        'test-alcohol.yaml: unknown employee EMP-999',
        'test-injection.yaml: gold clause §7.1 is not in the policy',
        'the test split has 0 clean cases, the plan says 1',
        'the dev split has 2 clean cases, the plan says 1',
    )


def test_the_hash_ignores_case_order_and_formatting(
    tmp_path: Path,
) -> None:
    reformatted_directory = tmp_path / 'cases'
    reformatted_directory.mkdir()
    for position, (source, content) in enumerate(reversed(fixture_cases().items())):
        content['submission']['amount'] = f'{content["submission"]["amount"]:.3f}'
        reformatted_text = yaml.safe_dump(content, default_flow_style=True, sort_keys=True)
        (reformatted_directory / f'{position:02}-{source}').write_text(reformatted_text)

    original = validate(fixture_cases())
    reformatted = validate(read_case_directory(reformatted_directory))

    assert reformatted.problems == ()
    assert reformatted.dataset_hash == original.dataset_hash


def _set(content: dict[str, Any], path: tuple[str | int, ...], value: object) -> None:
    target: Any = content
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize(
    ('source', 'path', 'value'),
    [
        ('test-alcohol.yaml', ('expected_route',), 'auto_approve'),
        ('test-alcohol.yaml', ('expected_recommendation',), 'none'),
        ('test-injection.yaml', ('expected_findings', 0, 'code'), 'RECEIPT_MISMATCH'),
        ('test-alcohol.yaml', ('expected_findings', 0, 'gold_clauses'), ['§2.2', '§2.1']),
        (
            'test-purpose-unclear.yaml',
            ('scripted_review', 'dispositions', 0, 'disposition'),
            'upheld',
        ),
        ('test-weekend.yaml', ('drafted_by', 'model_id'), 'fake-drafter-b'),
        ('test-clean.yaml', ('submission', 'amount'), 240.01),
    ],
)
def test_the_hash_changes_when_any_label_changes(
    source: str,
    path: tuple[str | int, ...],
    value: object,
) -> None:
    cases = fixture_cases()
    _set(cases[source], path, value)

    original = validate(fixture_cases())
    relabelled = validate(cases)

    assert relabelled.problems == ()
    assert relabelled.dataset_hash != original.dataset_hash


def test_the_committed_slice_plan_is_the_one_in_the_spec() -> None:
    plan = load_slice_plan(Path(__file__).parents[1] / 'evals' / 'slice_plan.yaml')

    assert plan.test.slices.model_dump() == {
        'judgement_positive': 16,
        'hard_negative': 8,
        'rule_coverage': 8,
        'clean': 6,
        'injection': 4,
    }
    assert (plan.test.judgement_positives_per_code, plan.test.memory_pairs) == (4, 3)
    dev_counts = plan.dev.slices.model_dump().values()
    assert sum(dev_counts) == 10
    assert min(dev_counts) >= 1
