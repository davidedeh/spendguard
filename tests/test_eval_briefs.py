import math
from collections import Counter
from pathlib import Path
from typing import get_args

import pytest

from spendguard.eval.briefs import (
    INJECTION_PAYLOADS,
    CaseBrief,
    example_submissions,
    generate_briefs,
)
from spendguard.eval.cases import Slice, SlicePlan, Split, load_slice_plan
from spendguard.policy.loader import POLICY_PATH, load_policy
from spendguard.rules.checks import falls_on_weekend_or_holiday, rule_findings
from spendguard.seed.agreement import CLIENT_ENTERTAINMENT, CLIENT_ENTERTAINMENT_MINIMUM_QUANTITY
from spendguard.seed.loader import SEED_DIRECTORY, read_seed_directory

PLAN = load_slice_plan(Path(__file__).parents[1] / 'evals' / 'slice_plan.yaml')
POLICY = load_policy(POLICY_PATH)
SEED = read_seed_directory(SEED_DIRECTORY)
BRIEFS = generate_briefs(PLAN, POLICY, SEED)


def briefs_in(split: Split) -> list[CaseBrief]:
    return [brief for brief in BRIEFS if brief.split == split]


@pytest.mark.parametrize('split', get_args(Split))
def test_brief_slice_counts_and_quotas_match_the_plan(split: Split) -> None:
    split_plan = PLAN.for_split(split)
    briefs = briefs_in(split)

    assert Counter(brief.slice for brief in briefs) == {
        slice_name: count for slice_name, count in split_plan.slices.model_dump().items() if count
    }
    assert sum(brief.submission_type == 'supplier_invoice' for brief in briefs) == (
        split_plan.supplier_invoices
    )
    assert sum(brief.currency != POLICY.base_currency for brief in briefs) == (
        split_plan.foreign_currency
    )
    assert sum(brief.memory_pair for brief in briefs) == split_plan.memory_pairs
    memory_pair_codes = {brief.intent.judgement_codes for brief in briefs if brief.memory_pair}
    assert len(memory_pair_codes) == min(split_plan.memory_pairs, 2)


@pytest.mark.parametrize('split', get_args(Split))
def test_briefs_target_the_planned_codes_for_each_slice(split: Split) -> None:
    split_plan = PLAN.for_split(split)
    briefs = briefs_in(split)
    by_slice: dict[Slice, list[CaseBrief]] = {
        slice_name: [brief for brief in briefs if brief.slice == slice_name]
        for slice_name in get_args(Slice)
    }

    judgement_targets = Counter(
        code for brief in by_slice['judgement_positive'] for code in brief.intent.judgement_codes
    )
    assert all(
        count <= split_plan.judgement_positives_per_code[code]
        for code, count in judgement_targets.items()
    )
    assert judgement_targets.total() == split_plan.slices.judgement_positive
    assert all(
        len(brief.intent.judgement_codes) == 1 and not brief.intent.rule_codes
        for brief in by_slice['judgement_positive'] + by_slice['injection']
    )
    assert all(
        len(brief.intent.rule_codes) == 1 and not brief.intent.judgement_codes
        for brief in by_slice['rule_coverage']
    )
    assert all(
        brief.intent.lookalike_code is not None
        and not brief.intent.rule_codes
        and not brief.intent.judgement_codes
        for brief in by_slice['hard_negative']
    )
    assert all(
        not brief.intent.rule_codes and not brief.intent.judgement_codes
        for brief in by_slice['clean']
    )


def test_briefs_off_the_weekend_code_and_its_follow_ups_fall_on_a_working_day() -> None:
    for brief in BRIEFS:
        targets_weekend = 'WEEKEND_SPEND_UNJUSTIFIED' in (
            *brief.intent.judgement_codes,
            brief.intent.lookalike_code,
            None if brief.memory is None else brief.memory.code,
        )
        assert falls_on_weekend_or_holiday(brief.dated_on, SEED) is targets_weekend, brief.case_id


def test_every_example_submission_raises_exactly_the_briefs_rule_findings() -> None:
    for brief in BRIEFS:
        for submission in example_submissions(brief, POLICY):
            raised = tuple(finding.code for finding in rule_findings(submission, SEED, POLICY))
            assert sorted(raised) == sorted(brief.intent.rule_codes), brief.case_id


def test_a_cost_centre_has_budget_room_unless_the_brief_intends_budget_exceeded() -> None:
    rooms = {
        centre.reference: centre.budget_gbp - centre.spend_to_date_gbp
        for centre in SEED.cost_centres
    }
    for brief in BRIEFS:
        has_room = rooms[brief.cost_centre] > 0
        assert has_room is ('BUDGET_EXCEEDED' not in brief.intent.rule_codes), brief.case_id


def test_no_claimant_or_supplier_carries_more_than_its_share_of_cases() -> None:
    claimants = Counter(brief.party for brief in BRIEFS if brief.submission_type == 'expense_claim')
    suppliers = Counter(
        brief.party for brief in BRIEFS if brief.submission_type == 'supplier_invoice'
    )

    assert max(claimants.values()) <= math.ceil(sum(claimants.values()) / len(SEED.employees)) + 1
    assert max(suppliers.values()) <= math.ceil(sum(suppliers.values()) / len(SEED.suppliers)) + 1


def test_only_duplicate_and_split_purchase_briefs_show_history_rows() -> None:
    for brief in BRIEFS:
        shows_history = bool(brief.history_claims or brief.history_invoices)
        targets_history = bool(
            {'DUPLICATE_SUBMISSION', 'SPLIT_PURCHASE'} & set(brief.intent.rule_codes)
        )
        assert shows_history is targets_history, brief.case_id


def test_every_injection_brief_carries_a_payload_in_a_field_that_suits_its_submission() -> None:
    for brief in BRIEFS:
        injection = brief.intent.injection
        if brief.slice != 'injection':
            assert injection is None, brief.case_id
            continue
        assert injection is not None, brief.case_id
        assert injection.payload in INJECTION_PAYLOADS[injection.kind], brief.case_id
        if brief.submission_type == 'supplier_invoice':
            assert injection.field == 'description', brief.case_id
        elif brief.receipt is not None and brief.receipt.present:
            assert injection.field in ('business_purpose', 'receipt_text'), brief.case_id
        else:
            assert injection.field == 'business_purpose', brief.case_id


def test_injection_briefs_rotate_through_every_kind_of_injected_instruction() -> None:
    injections = [brief.intent.injection for brief in BRIEFS if brief.slice == 'injection']
    kinds = Counter(injection.kind for injection in injections if injection is not None)

    assert len(injections) >= len(INJECTION_PAYLOADS)
    assert kinds.keys() == INJECTION_PAYLOADS.keys()
    assert max(kinds.values()) - min(kinds.values()) <= 1
    non_receipt_kind_fields = Counter(
        injection.field
        for injection in injections
        if injection is not None and injection.kind != 'text_in_receipt'
    )
    assert non_receipt_kind_fields.keys() == {'business_purpose', 'receipt_text'}


def test_brief_generation_is_deterministic() -> None:
    assert generate_briefs(PLAN, POLICY, SEED) == BRIEFS


def test_no_client_entertainment_brief_has_fewer_people_than_the_claimant_and_a_guest() -> None:
    entertainment_quantities = [
        brief.quantity for brief in BRIEFS if brief.category == CLIENT_ENTERTAINMENT
    ]

    assert entertainment_quantities
    assert all(
        quantity is not None and quantity >= CLIENT_ENTERTAINMENT_MINIMUM_QUANTITY
        for quantity in entertainment_quantities
    ), entertainment_quantities


def follow_ups_by_pair() -> dict[str, list[CaseBrief]]:
    follow_ups: dict[str, list[CaseBrief]] = {
        brief.case_id: [] for brief in BRIEFS if brief.memory_pair
    }
    for brief in BRIEFS:
        if brief.memory is not None:
            follow_ups[brief.memory.precedent_from].append(brief)
    return follow_ups


def test_each_memory_pair_has_one_follow_up_for_the_same_subject_code_and_split() -> None:
    briefs_by_id = {brief.case_id: brief for brief in BRIEFS}

    for pair_id, follow_ups in follow_ups_by_pair().items():
        pair = briefs_by_id[pair_id]
        assert len(follow_ups) == 1, pair_id
        follow_up = follow_ups[0]
        assert follow_up.slice == 'memory_follow_up'
        assert (follow_up.party, follow_up.split) == (pair.party, pair.split)
        assert follow_up.memory is not None
        assert (follow_up.memory.code,) == pair.intent.judgement_codes
        assert follow_up.intent.judgement_codes == ()


def test_a_follow_up_comes_later_than_its_pair_in_the_base_currency_within_auto_approval() -> None:
    briefs_by_id = {brief.case_id: brief for brief in BRIEFS}

    for pair_id, (follow_up,) in follow_ups_by_pair().items():
        assert follow_up.dated_on > briefs_by_id[pair_id].dated_on, pair_id
        assert follow_up.currency == POLICY.base_currency
        assert follow_up.intent.rule_codes == ()
        assert follow_up.amount.at_most is not None
        assert follow_up.amount.at_most <= POLICY.auto_approve_threshold


FIXTURES = Path(__file__).parent / 'fixtures'
FEW_CLAIMANT_BRIEFS = generate_briefs(
    SlicePlan.model_validate(
        {
            split: {
                'slices': {
                    'judgement_positive': 8,
                    'hard_negative': 8,
                    'rule_coverage': 0,
                    'clean': 0,
                    'injection': 4,
                    'memory_follow_up': 2,
                },
                'judgement_positives_per_code': {
                    'WEEKEND_SPEND_UNJUSTIFIED': 2,
                    'PURPOSE_UNCLEAR': 2,
                    'RECEIPT_MISMATCH': 2,
                    'ALCOHOL_NOT_PERMITTED': 2,
                },
                'memory_pairs': 2,
                'supplier_invoices': 0,
                'foreign_currency': 0,
            }
            for split in get_args(Split)
        }
    ),
    load_policy(FIXTURES / 'policy' / 'valid_policy.yaml'),
    read_seed_directory(FIXTURES / 'drafting_seed'),
)


@pytest.mark.parametrize('briefs', [BRIEFS, FEW_CLAIMANT_BRIEFS])
def test_no_brief_but_a_memory_pairs_own_shares_its_subject_and_code_in_its_split(
    briefs: list[CaseBrief],
) -> None:
    for pair in [brief for brief in briefs if brief.memory_pair]:
        sharing = [
            brief.case_id
            for brief in briefs
            if brief.split == pair.split
            and brief.party == pair.party
            and pair.intent.judgement_codes[0] in brief.intent.codes
            and brief.case_id != pair.case_id
        ]
        assert sharing == [], pair.case_id
