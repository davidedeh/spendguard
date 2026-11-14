from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from spendguard.policy.loader import ExpenseCategory, PolicyError, load_policy

FIXTURE_POLICY_PATH = Path(__file__).parent / 'fixtures' / 'policy' / 'valid_policy.yaml'

type PolicyEdit = Callable[[dict[str, Any]], None]


def write_edited_policy(directory: Path, edit: PolicyEdit) -> Path:
    policy_source = yaml.safe_load(FIXTURE_POLICY_PATH.read_text())
    edit(policy_source)
    edited_path = directory / 'policy.yaml'
    edited_path.write_text(yaml.safe_dump(policy_source, sort_keys=False))
    return edited_path


def test_a_valid_policy_loads_with_clauses_numbered_by_section_and_position() -> None:
    policy = load_policy(FIXTURE_POLICY_PATH)

    assert policy.version == 'fixture-1'
    assert [clause.reference for clause in policy.clauses] == [
        '§1.1',
        '§1.2',
        '§1.3',
        '§2.1',
        '§2.2',
        '§3.1',
        '§3.2',
        '§3.3',
        '§3.4',
        '§3.5',
    ]
    receipts = policy.clauses[1]
    assert receipts.heading_path == ('Claiming expenses', 'Receipts')
    assert receipts.text == (
        'A receipt is required for any item over 25 GBP, and it must match the amount claimed.'
    )
    assert policy.clauses[2].heading_path == ('Claiming expenses',)


def test_each_finding_code_carries_its_severity_check_kind_precedent_scope_and_clause() -> None:
    policy = load_policy(FIXTURE_POLICY_PATH)

    receipt_mismatch = policy.finding_codes['RECEIPT_MISMATCH']
    assert receipt_mismatch.severity == 'high'
    assert receipt_mismatch.check == 'judgement'
    assert receipt_mismatch.precedent_scope is None
    assert receipt_mismatch.clause.reference == '§1.2'

    non_approved_supplier = policy.finding_codes['NON_APPROVED_SUPPLIER']
    assert non_approved_supplier.severity == 'medium'
    assert non_approved_supplier.check == 'rule'
    assert non_approved_supplier.precedent_scope == 'supplier'
    assert non_approved_supplier.clause.reference == '§3.1'

    assert policy.finding_codes['PURPOSE_UNCLEAR'].precedent_scope == 'employee'
    assert len(policy.finding_codes) == 12


def test_the_applicability_table_lists_the_checks_for_each_submission_type() -> None:
    policy = load_policy(FIXTURE_POLICY_PATH)

    assert policy.applicability['expense_claim'] == (
        'OVER_CATEGORY_LIMIT',
        'MISSING_RECEIPT',
        'DUPLICATE_SUBMISSION',
        'CURRENCY_ANOMALY',
        'WEEKEND_SPEND_UNJUSTIFIED',
    )
    assert 'MISSING_PO' in policy.applicability['supplier_invoice']
    assert 'MISSING_PO' not in policy.applicability['expense_claim']


def test_each_expense_category_carries_the_unit_its_limit_applies_per() -> None:
    policy = load_policy(FIXTURE_POLICY_PATH)

    assert policy.expense_categories == {
        'subsistence': ExpenseCategory(
            key='subsistence',
            label='Meals',
            unit='person per day',
            limits={'junior': Decimal('40'), 'senior': Decimal('55')},
        ),
        'accommodation': ExpenseCategory(
            key='accommodation',
            label='Hotels',
            unit='night',
            limits={'junior': Decimal('120'), 'senior': Decimal('150')},
        ),
        'taxi': ExpenseCategory(
            key='taxi',
            label='Taxis',
            unit='journey',
            limits={'junior': Decimal('30'), 'senior': Decimal('30')},
        ),
    }


def test_a_category_with_no_unit_fails_to_load(tmp_path: Path) -> None:
    def remove_unit(policy_source: dict[str, Any]) -> None:
        del policy_source['parameters']['category_limits']['accommodation']['unit']

    with pytest.raises(PolicyError, match=r'accommodation\.unit'):
        load_policy(write_edited_policy(tmp_path, remove_unit))


def test_a_policy_with_no_category_limits_fails_to_load(tmp_path: Path) -> None:
    def remove_category_limits(policy_source: dict[str, Any]) -> None:
        del policy_source['parameters']['category_limits']

    with pytest.raises(PolicyError, match='category_limits'):
        load_policy(write_edited_policy(tmp_path, remove_category_limits))


def test_a_finding_code_with_no_severity_fails_to_load(tmp_path: Path) -> None:
    def remove_severity(policy_source: dict[str, Any]) -> None:
        del policy_source['finding_codes']['MISSING_PO']['severity']

    with pytest.raises(PolicyError, match=r'MISSING_PO[\s\S]*severity'):
        load_policy(write_edited_policy(tmp_path, remove_severity))


def test_a_finding_code_pointing_to_a_missing_clause_fails_to_load(tmp_path: Path) -> None:
    def point_at_missing_clause(policy_source: dict[str, Any]) -> None:
        policy_source['finding_codes']['BUDGET_EXCEEDED']['clause'] = 'budget_holders'

    with pytest.raises(PolicyError, match=r'BUDGET_EXCEEDED.*budget_holders'):
        load_policy(write_edited_policy(tmp_path, point_at_missing_clause))


def test_an_unknown_finding_code_fails_to_load(tmp_path: Path) -> None:
    def add_unknown_code(policy_source: dict[str, Any]) -> None:
        policy_source['finding_codes']['LATE_SUBMISSION'] = dict(
            policy_source['finding_codes']['MISSING_RECEIPT']
        )

    with pytest.raises(PolicyError, match='LATE_SUBMISSION'):
        load_policy(write_edited_policy(tmp_path, add_unknown_code))


def test_a_policy_missing_one_of_the_finding_codes_fails_to_load(tmp_path: Path) -> None:
    def remove_code(policy_source: dict[str, Any]) -> None:
        del policy_source['finding_codes']['CURRENCY_ANOMALY']
        for codes in policy_source['applicability'].values():
            codes.remove('CURRENCY_ANOMALY')

    with pytest.raises(PolicyError, match='CURRENCY_ANOMALY'):
        load_policy(write_edited_policy(tmp_path, remove_code))


def test_a_precedent_eligible_code_with_no_scope_fails_to_load(tmp_path: Path) -> None:
    def remove_scope(policy_source: dict[str, Any]) -> None:
        del policy_source['finding_codes']['PURPOSE_UNCLEAR']['precedent_scope']

    with pytest.raises(PolicyError, match=r'PURPOSE_UNCLEAR.*scope'):
        load_policy(write_edited_policy(tmp_path, remove_scope))


def test_an_instance_only_code_with_a_scope_fails_to_load(tmp_path: Path) -> None:
    def add_scope(policy_source: dict[str, Any]) -> None:
        policy_source['finding_codes']['ALCOHOL_NOT_PERMITTED']['precedent_scope'] = 'employee'

    with pytest.raises(PolicyError, match=r'ALCOHOL_NOT_PERMITTED.*scope'):
        load_policy(write_edited_policy(tmp_path, add_scope))


def test_a_clause_using_an_undefined_parameter_fails_to_load(tmp_path: Path) -> None:
    def remove_meal_limit(policy_source: dict[str, Any]) -> None:
        del policy_source['parameters']['meal_limit']

    with pytest.raises(PolicyError, match='meal_limit'):
        load_policy(write_edited_policy(tmp_path, remove_meal_limit))


def test_two_clauses_with_the_same_id_fail_to_load(tmp_path: Path) -> None:
    def duplicate_clause_id(policy_source: dict[str, Any]) -> None:
        policy_source['sections'][2]['clauses'][3]['id'] = 'budgets'

    with pytest.raises(PolicyError, match='budgets'):
        load_policy(write_edited_policy(tmp_path, duplicate_clause_id))


def test_an_applicability_table_missing_a_submission_type_fails_to_load(tmp_path: Path) -> None:
    def remove_supplier_invoice(policy_source: dict[str, Any]) -> None:
        del policy_source['applicability']['supplier_invoice']

    with pytest.raises(PolicyError, match='supplier_invoice'):
        load_policy(write_edited_policy(tmp_path, remove_supplier_invoice))


def clause_hashes(policy_path: Path) -> dict[str, str]:
    return {clause.id: clause.hash for clause in load_policy(policy_path).clauses}


def test_a_clause_hash_is_the_sha256_of_its_text_and_stable_across_loads() -> None:
    first_hashes = clause_hashes(FIXTURE_POLICY_PATH)

    assert first_hashes == clause_hashes(FIXTURE_POLICY_PATH)
    assert first_hashes['no_alcohol'] == (
        'f9ee08722d1a70ffdade27323b0c7f82e6a7a99a7f67c91e048fc20bd718a395'
    )


def test_reflowing_a_clause_without_changing_its_words_keeps_its_hash(tmp_path: Path) -> None:
    def reflow_alcohol_clause(policy_source: dict[str, Any]) -> None:
        policy_source['sections'][1]['clauses'][1]['text'] = '  Alcohol is\n  not   reimbursed.\n'

    edited_hashes = clause_hashes(write_edited_policy(tmp_path, reflow_alcohol_clause))

    assert edited_hashes == clause_hashes(FIXTURE_POLICY_PATH)


def changed_clause_ids(before: dict[str, str], after: dict[str, str]) -> set[str]:
    return {
        clause_id for clause_id, clause_hash in after.items() if before[clause_id] != clause_hash
    }


def test_editing_one_clause_changes_only_that_clause_hash(tmp_path: Path) -> None:
    def edit_budget_clause(policy_source: dict[str, Any]) -> None:
        policy_source['sections'][2]['clauses'][2]['text'] = 'Spend must stay within budget.'

    edited_hashes = clause_hashes(write_edited_policy(tmp_path, edit_budget_clause))

    assert changed_clause_ids(clause_hashes(FIXTURE_POLICY_PATH), edited_hashes) == {'budgets'}


def test_changing_a_parameter_changes_only_the_clause_that_uses_it(tmp_path: Path) -> None:
    def raise_meal_limit(policy_source: dict[str, Any]) -> None:
        policy_source['parameters']['meal_limit'] = 45

    edited_policy = load_policy(write_edited_policy(tmp_path, raise_meal_limit))
    edited_hashes = {clause.id: clause.hash for clause in edited_policy.clauses}

    assert changed_clause_ids(clause_hashes(FIXTURE_POLICY_PATH), edited_hashes) == {'meal_limit'}
    assert edited_policy.clauses[3].text == 'A meal may cost up to 45 GBP per person.'


def test_the_rendered_policy_numbers_each_section_and_clause_under_its_heading() -> None:
    policy = load_policy(FIXTURE_POLICY_PATH)

    assert policy.rendered_text.startswith(
        '# Fixture Expenses Policy\n'
        '\n'
        'Version fixture-1\n'
        '\n'
        '## 1. Claiming expenses\n'
        '\n'
        '**§1.1 Business purpose.** Every claim states the business purpose of the spend.\n'
        '\n'
        '**§1.2 Receipts.** A receipt is required for any item over 25 GBP, '
        'and it must match the amount claimed.\n'
        '\n'
        '**§1.3** Spend on a weekend or public holiday needs a written justification.\n'
        '\n'
        '## 2. Meals\n'
    )


def test_the_rendered_policy_contains_exactly_the_clause_text_that_was_hashed(
    tmp_path: Path,
) -> None:
    def raise_purchase_order_threshold(policy_source: dict[str, Any]) -> None:
        policy_source['parameters']['purchase_order_threshold'] = 750

    policy = load_policy(write_edited_policy(tmp_path, raise_purchase_order_threshold))

    assert '**§3.2** Any purchase over 750 GBP needs a purchase order' in policy.rendered_text
    for clause in policy.clauses:
        assert f'**{clause.reference}' in policy.rendered_text
        assert f'{clause.text}\n' in policy.rendered_text
