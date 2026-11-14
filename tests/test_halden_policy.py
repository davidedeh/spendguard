from pathlib import Path

import yaml

from spendguard.policy.loader import FINDING_CODES, POLICY_PATH, load_policy

INT_TAG = 'tag:yaml.org,2002:int'
FLOAT_TAG = 'tag:yaml.org,2002:float'


def test_the_halden_policy_defines_the_closed_code_set_and_its_precedent_eligibility() -> None:
    policy = load_policy(POLICY_PATH)

    assert set(policy.finding_codes) == set(FINDING_CODES)
    precedent_scopes = {
        code: definition.precedent_scope
        for code, definition in policy.finding_codes.items()
        if definition.precedent_scope is not None
    }
    assert precedent_scopes == {
        'NON_APPROVED_SUPPLIER': 'supplier',
        'PURPOSE_UNCLEAR': 'employee',
        'WEEKEND_SPEND_UNJUSTIFIED': 'employee',
    }
    judgement_codes = {
        code for code, definition in policy.finding_codes.items() if definition.check == 'judgement'
    }
    assert judgement_codes == {
        'PURPOSE_UNCLEAR',
        'RECEIPT_MISMATCH',
        'ALCOHOL_NOT_PERMITTED',
        'WEEKEND_SPEND_UNJUSTIFIED',
    }
    clause_ids = {clause.id for clause in policy.clauses}
    assert all(definition.clause.id in clause_ids for definition in policy.finding_codes.values())


EXCEPTIONS_TO_BASE_CLAUSES = {
    'entertainment_alcohol': 'no_alcohol',
    'trips_over_weekends': 'weekend_and_holiday_spend',
    'tips': 'receipt_matches_claim',
    'foreign_currency_receipts': 'receipt_matches_claim',
    'standing_approvals': 'business_purpose',
}


def test_the_halden_policy_places_each_exception_in_a_different_section_from_its_base_rule() -> (
    None
):
    policy = load_policy(POLICY_PATH)

    assert 36 <= len(policy.clauses) <= 44
    sections_by_clause_id = {clause.id: clause.heading_path[0] for clause in policy.clauses}
    for exception_id, base_id in EXCEPTIONS_TO_BASE_CLAUSES.items():
        assert sections_by_clause_id[exception_id] != sections_by_clause_id[base_id], exception_id


DISTRACTOR_SECTIONS = {
    'Booking travel',
    'Gifts and hospitality received',
    'IT equipment',
    'Conference and course fees',
}


def test_no_finding_code_or_exception_rests_on_a_distractor_section() -> None:
    policy = load_policy(POLICY_PATH)

    section_headings = {clause.heading_path[0] for clause in policy.clauses}
    assert section_headings >= DISTRACTOR_SECTIONS
    sections_by_clause_id = {clause.id: clause.heading_path[0] for clause in policy.clauses}
    load_bearing_clause_ids = {
        definition.clause.id for definition in policy.finding_codes.values()
    } | set(EXCEPTIONS_TO_BASE_CLAUSES)
    assert not {
        clause_id
        for clause_id in load_bearing_clause_ids
        if sections_by_clause_id[clause_id] in DISTRACTOR_SECTIONS
    }


def test_every_category_sets_a_limit_for_each_grade_and_only_approving_grades_have_limits() -> None:
    policy = load_policy(POLICY_PATH)

    grades = policy.parameters['grades']
    category_limits = policy.parameters['category_limits']
    approval_limits = policy.parameters['approval_limits']
    assert isinstance(grades, list)
    assert isinstance(category_limits, dict)
    assert isinstance(approval_limits, dict)
    for category_name, category in category_limits.items():
        assert isinstance(category, dict)
        limits = category['limits']
        assert isinstance(limits, dict)
        assert list(limits) == grades, category_name
    assert set(approval_limits) == {'manager', 'executive'}


RENDERED_POLICY_PATH = Path(__file__).parents[1] / 'docs' / 'halden-policy.md'


def test_the_committed_prose_policy_matches_the_rendered_policy() -> None:
    policy = load_policy(POLICY_PATH)

    assert RENDERED_POLICY_PATH.read_text() == policy.rendered_text, (
        'Regenerate with: uv run python -m spendguard.policy > docs/halden-policy.md'
    )


GUT_PICK_MARKER = '# gut pick'


def test_every_numeric_parameter_and_every_severity_is_marked_as_a_gut_pick() -> None:
    policy_lines = POLICY_PATH.read_text().splitlines()
    document = yaml.compose(POLICY_PATH.read_text())
    assert isinstance(document, yaml.MappingNode)
    top_level = {key.value: value for key, value in document.value}

    unmarked_lines = [
        node.start_mark.line + 1
        for node in [
            *_numeric_scalars(top_level['parameters']),
            *_severity_scalars(top_level['finding_codes']),
        ]
        if GUT_PICK_MARKER not in policy_lines[node.start_mark.line]
    ]

    assert not unmarked_lines


def _numeric_scalars(node: yaml.Node) -> list[yaml.ScalarNode]:
    if isinstance(node, yaml.ScalarNode):
        return [node] if node.tag in {INT_TAG, FLOAT_TAG} else []
    if isinstance(node, yaml.SequenceNode):
        return [scalar for item in node.value for scalar in _numeric_scalars(item)]
    if isinstance(node, yaml.MappingNode):
        return [scalar for _, value in node.value for scalar in _numeric_scalars(value)]
    return []


def _severity_scalars(finding_codes: yaml.Node) -> list[yaml.Node]:
    assert isinstance(finding_codes, yaml.MappingNode)
    return [
        value
        for _, definition in finding_codes.value
        for key, value in definition.value
        if key.value == 'severity'
    ]
