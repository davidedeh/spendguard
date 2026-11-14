import math
from collections import Counter
from pathlib import Path
from typing import get_args

import pytest

from spendguard.eval.briefs import CaseBrief, example_submissions, generate_briefs
from spendguard.eval.cases import Slice, Split, load_slice_plan
from spendguard.policy.loader import POLICY_PATH, load_policy
from spendguard.rules.checks import falls_on_weekend_or_holiday, rule_findings
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
    assert set(judgement_targets.values()) <= {split_plan.judgement_positives_per_code}
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


def test_every_brief_not_targeting_the_weekend_code_falls_on_a_working_day() -> None:
    for brief in BRIEFS:
        targets_weekend = 'WEEKEND_SPEND_UNJUSTIFIED' in (
            *brief.intent.judgement_codes,
            brief.intent.lookalike_code,
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


def test_brief_generation_is_deterministic() -> None:
    assert generate_briefs(PLAN, POLICY, SEED) == BRIEFS
