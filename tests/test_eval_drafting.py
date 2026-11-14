import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from settings_factory import make_test_settings
from spendguard.eval.briefs import CaseBrief, generate_briefs
from spendguard.eval.cases import DraftedBy, EvalCase, SlicePlan, read_case_directory
from spendguard.eval.drafting import DRAFTED_CASE_SCHEMA, assign_drafters, run
from spendguard.models.fake import FakeChatModel
from spendguard.policy.loader import FindingCode, load_policy
from spendguard.rules.checks import base_currency_amount, finding_for, rule_findings
from spendguard.rules.decisions import ReviewedFinding, recommend, route
from spendguard.seed.loader import read_seed_directory
from spendguard.settings import ChatModelSettings, Settings
from spendguard.submission import ExpenseClaim

FIXTURES = Path(__file__).parent / 'fixtures'
FIXTURE_POLICY_PATH = FIXTURES / 'policy' / 'valid_policy.yaml'
FIXTURE_POLICY = load_policy(FIXTURE_POLICY_PATH)
DRAFTING_SEED_DIRECTORY = FIXTURES / 'drafting_seed'
DRAFTING_SEED = read_seed_directory(DRAFTING_SEED_DIRECTORY)
DRAFTER_MODEL_IDS = {
    'fake_claude': 'fake-claude-drafter',
    'fake_gpt': 'fake-gpt-drafter',
    'fake_nova': 'fake-nova-drafter',
}
DRAFTING_PLAN: dict[str, dict[str, Any]] = {
    'test': {
        'slices': {
            'judgement_positive': 8,
            'hard_negative': 5,
            'rule_coverage': 4,
            'clean': 2,
            'injection': 1,
        },
        'judgement_positives_per_code': 2,
        'memory_pairs': 2,
        'supplier_invoices': 4,
        'foreign_currency': 3,
    },
    'dev': {
        'slices': {
            'judgement_positive': 4,
            'hard_negative': 2,
            'rule_coverage': 1,
            'clean': 1,
            'injection': 0,
        },
        'judgement_positives_per_code': 1,
        'memory_pairs': 1,
        'supplier_invoices': 1,
        'foreign_currency': 1,
    },
}
BRIEFS = generate_briefs(SlicePlan.model_validate(DRAFTING_PLAN), FIXTURE_POLICY, DRAFTING_SEED)
BRIEFS_BY_ID = {brief.case_id: brief for brief in BRIEFS}
ASSIGNED = dict(
    zip(
        (brief.case_id for brief in BRIEFS),
        assign_drafters(BRIEFS, tuple(DRAFTER_MODEL_IDS)),
        strict=True,
    )
)
BRIEF_PATTERN = re.compile(r'## Brief\n\n```yaml\n(.*?)```', re.DOTALL)
REJECTION_PATTERN = re.compile(r'was rejected: (.*)\. Draft the case again', re.DOTALL)

FailureKind = Literal[
    'malformed',
    'off_brief',
    'outside_judgement',
    'no_review',
    'anomalous_rate',
    'small_without_receipt',
    'receipt_total',
    'unitemised_receipt',
    'equal_total',
    'missing_injection',
    'placeholder',
    'supplier_merchant',
    'decorated_json',
    'omitted_nulls',
    'empty_nulls',
]
REPLY_FAILURES = ('decorated_json', 'omitted_nulls', 'empty_nulls')


def brief_in(messages: list[BaseMessage]) -> dict[str, Any]:
    match = BRIEF_PATTERN.search(str(messages[1].content))
    assert match is not None
    brief: dict[str, Any] = yaml.safe_load(match.group(1))
    return brief


def satisfying_draft(brief: dict[str, Any], marker: str) -> dict[str, Any]:
    bounds = brief['amount']
    if 'exactly' in bounds:
        amount = Decimal(str(bounds['exactly']))
    elif 'at_most' in bounds:
        amount = Decimal(str(bounds['at_most']))
    else:
        amount = Decimal(str(bounds['more_than'])) + 1
    claimed_rate = None
    if brief['claimed_rate_must_deviate']:
        rate = Decimal(str(brief['company_rate'])) * Decimal('1.05')
        claimed_rate = str(rate.quantize(Decimal('0.0001')))
    if brief['submission_type'] == 'expense_claim':
        submission: dict[str, Any] = {
            'type': 'expense_claim',
            'claimant': brief['party'],
            'spent_on': brief['dated_on'],
            'category': brief['category'],
            'quantity': brief['quantity'],
            'merchant': 'Harbour Kitchen',
            'amount': str(amount),
            'currency': brief['currency'],
            'claimed_exchange_rate': claimed_rate,
            'business_purpose': marker,
            'receipt_text': None,
        }
        receipt = brief['receipt']
        if receipt['present']:
            total = amount if 'total_differs_because' not in receipt else amount - 1
            submission['receipt_text'] = f'Harbour Kitchen\n1 x Lunch {total}\nTOTAL {total}'
    else:
        submission = {
            'type': 'supplier_invoice',
            'supplier': brief['party'],
            'supplier_invoice_number': brief.get('supplier_invoice_number', marker),
            'purchase_order': brief.get('purchase_order'),
            'cost_centre': None if brief.get('purchase_order') else brief['cost_centre'],
            'issued_on': brief['dated_on'],
            'amount': str(amount),
            'currency': brief['currency'],
            'claimed_exchange_rate': claimed_rate,
            'description': marker,
        }
    injection = brief['intent'].get('injection')
    if injection is not None:
        separator = '\n' if injection['field'] == 'receipt_text' else ' '
        submission[injection['field']] += separator + injection['payload']
    findings = [
        {
            'id': f'finding-{index}',
            'code': code,
            'gold_clauses': [FIXTURE_POLICY.finding_codes[code].clause.reference],
        }
        for index, code in enumerate(brief['intent']['judgement_codes'], start=1)
    ]
    review = None
    if brief['memory_pair']:
        review = {
            'approver': 'EMP-002' if brief['party'] == 'EMP-001' else 'EMP-001',
            'dispositions': [
                {
                    'finding_id': 'finding-1',
                    'disposition': 'dismissed',
                    'reason': 'A standing arrangement agreed with the manager.',
                }
            ],
        }
    return {'submission': submission, 'judgement_findings': findings, 'scripted_review': review}


def with_failure(draft: dict[str, Any], brief: dict[str, Any], failure: FailureKind) -> None:
    submission = draft['submission']
    match failure:
        case 'malformed':
            del draft['submission']
        case 'off_brief':
            if submission['type'] == 'expense_claim':
                submission['claimant'] = 'EMP-004'
            else:
                submission['supplier'] = 'SUP-002'
        case 'outside_judgement':
            code: FindingCode = (
                'RECEIPT_MISMATCH'
                if 'PURPOSE_UNCLEAR' in brief['intent']['judgement_codes']
                else 'PURPOSE_UNCLEAR'
            )
            draft['judgement_findings'].append(
                {
                    'id': 'extra',
                    'code': code,
                    'gold_clauses': [FIXTURE_POLICY.finding_codes[code].clause.reference],
                }
            )
        case 'no_review':
            draft['scripted_review'] = None
        case 'anomalous_rate':
            if 'company_rate' in brief:
                submission['claimed_exchange_rate'] = str(Decimal(str(brief['company_rate'])) * 2)
        case 'missing_injection' | 'placeholder' | 'supplier_merchant':
            _break_free_text(submission, brief, failure)
        case 'small_without_receipt' | 'receipt_total' | 'unitemised_receipt' | 'equal_total':
            _break_receipt(submission, failure)
        case 'decorated_json' | 'omitted_nulls' | 'empty_nulls':
            pass


def _break_free_text(
    submission: dict[str, Any], brief: dict[str, Any], failure: FailureKind
) -> None:
    is_claim = submission['type'] == 'expense_claim'
    match failure:
        case 'missing_injection':
            injection = brief['intent']['injection']
            submission[injection['field']] = submission[injection['field']].replace(
                injection['payload'], ''
            )
        case 'placeholder':
            if is_claim:
                submission['merchant'] = 'Merchant Name'
            else:
                submission['description'] = 'Item Description'
        case 'supplier_merchant':
            if is_claim:
                submission['merchant'] = 'SUP-003'
        case _:
            raise ValueError(f'{failure} does not break free text')


def _break_receipt(submission: dict[str, Any], failure: FailureKind) -> None:
    match failure:
        case 'small_without_receipt':
            submission['amount'] = '20.00'
            submission['receipt_text'] = None
        case 'receipt_total':
            amount = Decimal(submission['amount'])
            submission['receipt_text'] = (
                f'Harbour Kitchen\n1 x Lunch {amount + 1}\nTOTAL {amount + 1}'
            )
        case 'unitemised_receipt':
            submission['receipt_text'] = 'Paid in full, thank you'
        case 'equal_total':
            amount = Decimal(submission['amount'])
            submission['receipt_text'] = f'Harbour Kitchen\n1 x Lunch {amount}\nTOTAL {amount}'
        case _:
            raise ValueError(f'{failure} does not break a receipt')


class DraftingFake(FakeChatModel):
    failing_calls: int = 0
    failing_slice: str | None = None
    failure: FailureKind = 'malformed'
    reviews_every_case: bool = False
    tool_schemas: list[Any] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        self.tool_schemas.append(kwargs.get('tools'))
        brief = brief_in(messages)
        draft = satisfying_draft(brief, f'{self.model_id} call {len(self.received)}')
        if self.reviews_every_case and draft['judgement_findings'] and not brief['memory_pair']:
            draft['scripted_review'] = satisfying_draft({**brief, 'memory_pair': True}, '')[
                'scripted_review'
            ]
        failing = len(self.received) <= self.failing_calls or brief['slice'] == self.failing_slice
        if failing and self.failure not in REPLY_FAILURES:
            with_failure(draft, brief, self.failure)
        if failing and self.failure == 'omitted_nulls':
            draft = _without_nulls(draft)
        if failing and self.failure == 'empty_nulls':
            draft = _nulls_as_empty_text(draft)
        if self.structured_output_calls[-1][0] == 'json_schema':
            decoration = '**' if failing and self.failure == 'decorated_json' else ''
            message = AIMessage(content=decoration + json.dumps(draft))
        else:
            message = AIMessage(
                content='',
                tool_calls=[{'name': 'DraftedCase', 'args': draft, 'id': 'call_DraftedCase'}],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def _without_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_nulls(item) for item in value]
    return value


def _nulls_as_empty_text(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _nulls_as_empty_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_nulls_as_empty_text(item) for item in value]
    return '' if value is None else value


class DrafterPool:
    def __init__(
        self,
        failures: dict[str, tuple[FailureKind, int]] | None = None,
        failing_slice: tuple[FailureKind, str] | None = None,
        reviews_every_case: bool = False,
    ) -> None:
        self.failures = failures or {}
        self.failing_slice = failing_slice
        self.reviews_every_case = reviews_every_case
        self.models: dict[str, DraftingFake] = {}

    def build(self, settings: Settings, model_key: str) -> BaseChatModel:
        failure, failing_calls = self.failures.get(model_key, ('malformed', 0))
        failing_slice = None
        if self.failing_slice is not None:
            failure, failing_slice = self.failing_slice
        model = DraftingFake(
            model_id=settings.chat_models[model_key].model_id,
            failing_calls=failing_calls,
            failing_slice=failing_slice,
            failure=failure,
            reviews_every_case=self.reviews_every_case,
        )
        self.models[model_key] = model
        return model


def drafting_settings(json_schema_keys: tuple[str, ...] = ()) -> Settings:
    return make_test_settings(
        chat_models={
            key: ChatModelSettings(
                provider='fake',
                model_id=model_id,
                structured_output_method=(
                    'json_schema' if key in json_schema_keys else 'function_calling'
                ),
            )
            for key, model_id in DRAFTER_MODEL_IDS.items()
        },
        primary_model_key='fake_claude',
        drafting_model_keys=tuple(DRAFTER_MODEL_IDS),
    )


def drafting_arguments(tmp_path: Path) -> list[str]:
    plan_path = tmp_path / 'slice_plan.yaml'
    plan_path.write_text(yaml.safe_dump(DRAFTING_PLAN))
    return [
        '--cases',
        str(tmp_path / 'cases'),
        '--plan',
        str(plan_path),
        '--policy',
        str(FIXTURE_POLICY_PATH),
        '--seed',
        str(DRAFTING_SEED_DIRECTORY),
        '--lock',
        str(tmp_path / 'dataset.lock.json'),
        '--drafts',
        str(tmp_path / 'drafts'),
    ]


def read_drafted_cases(tmp_path: Path) -> list[EvalCase]:
    return [
        EvalCase.model_validate(content)
        for content in read_case_directory(tmp_path / 'cases').values()
    ]


def marker_of(case: EvalCase) -> str:
    submission = case.submission
    if isinstance(submission, ExpenseClaim):
        return submission.business_purpose
    return submission.description


def rejections_for(drafters: DrafterPool, brief: CaseBrief) -> list[str]:
    brief_text = yaml.safe_dump(
        brief.model_dump(mode='json', exclude={'case_id', 'split'}, exclude_none=True),
        sort_keys=False,
        allow_unicode=True,
    )
    rejections = []
    for model in drafters.models.values():
        for prompt in model.received:
            text = str(prompt[1].content)
            match = REJECTION_PATTERN.search(text)
            if brief_text in text and match:
                rejections.append(match.group(1))
    return rejections


def _first_brief(predicate: Any) -> CaseBrief:
    return next(brief for brief in BRIEFS if predicate(brief))


FOREIGN_HARD_NEGATIVE = _first_brief(
    lambda brief: brief.slice == 'hard_negative' and brief.company_rate is not None
)
CLEAN_BRIEF = _first_brief(lambda brief: brief.slice == 'clean')
RECEIPTED_HARD_NEGATIVE = _first_brief(
    lambda brief: (
        brief.slice == 'hard_negative'
        and brief.receipt is not None
        and brief.receipt.total_differs_because is None
    )
)
MEMORY_PAIR_BRIEF = _first_brief(lambda brief: brief.memory_pair)
INJECTION_BRIEF = _first_brief(lambda brief: brief.slice == 'injection')
INJECTION_FIELD = INJECTION_BRIEF.intent.injection.field if INJECTION_BRIEF.intent.injection else ''
DIFFERING_RECEIPT_BRIEF = _first_brief(
    lambda brief: (
        brief.slice == 'hard_negative'
        and brief.receipt is not None
        and brief.receipt.total_differs_because is not None
    )
)


def test_the_drafters_output_schema_holds_no_rule_findings_route_or_recommendation() -> None:
    assert set(DRAFTED_CASE_SCHEMA['properties']) == {
        'submission',
        'judgement_findings',
        'scripted_review',
    }


def test_every_slice_is_split_across_the_drafters_with_none_more_than_one_off_a_third(
    tmp_path: Path,
) -> None:
    exit_code = run(drafting_arguments(tmp_path), drafting_settings(), DrafterPool().build)

    assert exit_code == 0
    cases = read_drafted_cases(tmp_path)
    assert len(cases) == len(BRIEFS)
    for split, split_plan in DRAFTING_PLAN.items():
        for slice_name, planned_count in split_plan['slices'].items():
            drafter_counts = Counter(
                case.drafted_by.model_id
                for case in cases
                if case.split == split and case.slice == slice_name
            )
            for model_id in DRAFTER_MODEL_IDS.values():
                assert abs(drafter_counts[model_id] - planned_count / 3) < 1, (
                    split,
                    slice_name,
                    drafter_counts,
                )


def test_every_case_carries_the_drafted_by_of_the_model_that_produced_it(
    tmp_path: Path,
) -> None:
    run(drafting_arguments(tmp_path), drafting_settings(), DrafterPool().build)

    for case in read_drafted_cases(tmp_path):
        producing_model_id = marker_of(case).split(' call ')[0]
        assert case.drafted_by == DraftedBy(provider='fake', model_id=producing_model_id)


def test_rule_findings_route_and_recommendation_are_computed_by_code(tmp_path: Path) -> None:
    run(drafting_arguments(tmp_path), drafting_settings(), DrafterPool().build)

    for case in read_drafted_cases(tmp_path):
        computed = rule_findings(case.submission, DRAFTING_SEED, FIXTURE_POLICY)
        labelled_rules = [
            finding.code
            for finding in case.expected_findings
            if FIXTURE_POLICY.finding_codes[finding.code].check == 'rule'
        ]
        assert labelled_rules == [finding.code for finding in computed]
        assert sorted(labelled_rules) == sorted(BRIEFS_BY_ID[case.id].intent.rule_codes)
        findings = [finding_for(finding.code, FIXTURE_POLICY) for finding in case.expected_findings]
        assert case.expected_route == route(
            findings=findings,
            cleared_by_precedent=False,
            base_amount=base_currency_amount(case.submission, DRAFTING_SEED, FIXTURE_POLICY),
            policy=FIXTURE_POLICY,
            force_human_review=False,
        )
        assert case.expected_recommendation == recommend(
            [ReviewedFinding(finding, has_precedent=False) for finding in findings]
        )
        assert case.intent.judgement_codes == BRIEFS_BY_ID[case.id].intent.judgement_codes


def test_a_case_its_drafter_cannot_draft_falls_back_and_is_counted_in_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    drafters = DrafterPool(failures={'fake_nova': ('off_brief', 1000)})

    exit_code = run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    assert exit_code == 0
    nova_case_ids = {case_id for case_id, key in ASSIGNED.items() if key == 'fake_nova'}
    fallback_models = Counter(
        case.drafted_by.model_id
        for case in read_drafted_cases(tmp_path)
        if case.id in nova_case_ids
    )
    assert fallback_models.keys() == {'fake-claude-drafter', 'fake-gpt-drafter'}
    assert abs(fallback_models['fake-claude-drafter'] - fallback_models['fake-gpt-drafter']) <= 1
    summary = capsys.readouterr().out
    assert f'Fallbacks: {len(nova_case_ids)} (' in summary
    assert 'fake_nova 0' in summary


def test_drafts_that_fail_validation_are_not_written_and_the_problems_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    drafters = DrafterPool(failing_slice=('off_brief', 'injection'))

    exit_code = run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    assert exit_code == 1
    assert not (tmp_path / 'cases').exists()
    errors = capsys.readouterr().err
    party = BRIEFS_BY_ID['test-injection-01'].party
    assert (
        'test-injection-01.yaml: no valid draft after 3 attempts: claimant is EMP-004,'
        f' the brief says {party}'
    ) in errors.splitlines()
    assert 'test-judgement-positive-01.yaml' not in errors


@pytest.mark.parametrize(
    ('failure', 'brief', 'problem'),
    [
        (
            'anomalous_rate',
            FOREIGN_HARD_NEGATIVE,
            'the rule checks raise CURRENCY_ANOMALY on this submission, the brief intends no'
            " rule findings: keep to the brief's figures",
        ),
        (
            'outside_judgement',
            CLEAN_BRIEF,
            'the draft labels PURPOSE_UNCLEAR, which this brief does not intend: change the'
            ' submission so that this concern does not arise, rather than only dropping the label',
        ),
        (
            'small_without_receipt',
            RECEIPTED_HARD_NEGATIVE,
            'the brief says the claim has a receipt, and receipt_text is missing',
        ),
        (
            'receipt_total',
            RECEIPTED_HARD_NEGATIVE,
            f'the receipt total {(RECEIPTED_HARD_NEGATIVE.amount.at_most or 0) + 1}'
            f' differs from the claimed amount {RECEIPTED_HARD_NEGATIVE.amount.at_most}',
        ),
        (
            'unitemised_receipt',
            RECEIPTED_HARD_NEGATIVE,
            'receipt_text is not an itemised receipt: give one line per item with its price and'
            ' a final line starting TOTAL with the amount',
        ),
        (
            'equal_total',
            DIFFERING_RECEIPT_BRIEF,
            'the receipt total equals the claimed amount, the brief says they differ because of'
            f' {(DIFFERING_RECEIPT_BRIEF.receipt.total_differs_because or "").replace("_", " ")}'
            if DIFFERING_RECEIPT_BRIEF.receipt
            else '',
        ),
        (
            'no_review',
            MEMORY_PAIR_BRIEF,
            'the brief asks for a scripted review and the draft has none',
        ),
        (
            'missing_injection',
            INJECTION_BRIEF,
            f'{INJECTION_FIELD} does not contain the injected instruction word for word: put the'
            f' payload from the brief in {INJECTION_FIELD} exactly as given',
        ),
        (
            'placeholder',
            RECEIPTED_HARD_NEGATIVE,
            "merchant holds the placeholder 'Merchant Name': write what a real submission would"
            ' say instead',
        ),
        (
            'supplier_merchant',
            RECEIPTED_HARD_NEGATIVE,
            'merchant holds the supplier reference SUP-003: give the name the merchant trades'
            ' under',
        ),
    ],
)
def test_each_rejection_names_exactly_the_problem_with_the_draft(
    tmp_path: Path, failure: FailureKind, brief: CaseBrief, problem: str
) -> None:
    drafters = DrafterPool(failing_slice=(failure, brief.slice))

    run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    assert rejections_for(drafters, brief)[0] == problem


def test_drafting_refuses_once_the_lock_file_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = drafting_arguments(tmp_path)
    (tmp_path / 'dataset.lock.json').write_text('{"dataset_hash": "frozen"}\n')
    drafters = DrafterPool()

    exit_code = run(arguments, drafting_settings(), drafters.build)

    assert exit_code == 1
    assert drafters.models == {}
    assert not (tmp_path / 'cases').exists()
    assert 'is frozen' in capsys.readouterr().err


def test_drafting_refuses_to_overwrite_cases_already_in_the_case_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = drafting_arguments(tmp_path)
    (tmp_path / 'cases').mkdir()
    reviewed_case = tmp_path / 'cases' / 'test-clean-01.yaml'
    reviewed_case.write_text('id: test-clean-01\n')
    drafters = DrafterPool()

    exit_code = run(arguments, drafting_settings(), drafters.build)

    assert exit_code == 1
    assert drafters.models == {}
    assert reviewed_case.read_text() == 'id: test-clean-01\n'
    assert 'already holds cases' in capsys.readouterr().err


def test_prompts_hold_the_policy_and_the_brief_and_nothing_from_other_cases(
    tmp_path: Path,
) -> None:
    drafters = DrafterPool()

    run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    cases = read_drafted_cases(tmp_path)
    prompts = [prompt for model in drafters.models.values() for prompt in model.received]
    assert len(prompts) == len(BRIEFS)
    contexts = set()
    for prompt in prompts:
        assert [message.type for message in prompt] == ['system', 'human']
        text = '\n'.join(str(message.content) for message in prompt)
        assert FIXTURE_POLICY.rendered_text in text
        for case in cases:
            assert case.id not in text
            assert marker_of(case) not in text
        for code_marker in ('def ', 'import ', 'spendguard', 'class '):
            assert code_marker not in text
        assert 'fx_rates' not in text
        brief = brief_in(prompt)
        shows_history = any(row in text for row in ('INV-0001', 'INV-0002', 'CLM-0001'))
        targets_history = bool(
            {'DUPLICATE_SUBMISSION', 'SPLIT_PURCHASE'} & set(brief['intent']['rule_codes'])
        )
        assert shows_history is targets_history
        contexts.add(str(prompt[1].content).split('## Brief')[0])
    assert len(contexts) == 1


def test_prompts_name_each_finding_code_and_each_expense_category_with_its_unit(
    tmp_path: Path,
) -> None:
    drafters = DrafterPool()

    run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    prompt_text = str(drafters.models['fake_nova'].received[0][1].content)
    assert '- ALCOHOL_NOT_PERMITTED, a judgement check, rests on §2.2' in prompt_text
    assert '- MISSING_PO, a rule check, rests on §3.2' in prompt_text
    assert '- accommodation (Hotels): limit applies per night\n' in prompt_text


@pytest.mark.parametrize('failure', ['malformed', 'off_brief'])
def test_a_draft_that_fails_once_is_retried_with_the_reason_and_written(
    tmp_path: Path, failure: FailureKind
) -> None:
    drafters = DrafterPool(failures={'fake_gpt': (failure, 1)})

    exit_code = run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    assert exit_code == 0
    assert len(read_drafted_cases(tmp_path)) == len(BRIEFS)
    gpt_briefs = sum(key == 'fake_gpt' for key in ASSIGNED.values())
    first_prompt, retry_prompt = drafters.models['fake_gpt'].received[:2]
    assert len(drafters.models['fake_gpt'].received) == gpt_briefs + 1
    assert str(retry_prompt[1].content).startswith(str(first_prompt[1].content))
    assert 'was rejected: ' in str(retry_prompt[1].content)


def test_a_failed_run_keeps_its_valid_drafts_and_a_rerun_drafts_only_what_failed(
    tmp_path: Path,
) -> None:
    failing_run = run(
        drafting_arguments(tmp_path),
        drafting_settings(),
        DrafterPool(failing_slice=('off_brief', 'injection')).build,
    )
    rerun_drafters = DrafterPool()

    rerun = run(drafting_arguments(tmp_path), drafting_settings(), rerun_drafters.build)

    assert (failing_run, rerun) == (1, 0)
    rerun_prompts = [
        prompt for model in rerun_drafters.models.values() for prompt in model.received
    ]
    assert [brief_in(prompt)['slice'] for prompt in rerun_prompts] == ['injection']
    assert len(read_drafted_cases(tmp_path)) == len(BRIEFS)
    assert not any((tmp_path / 'drafts').glob('*.yaml'))


def test_a_scripted_review_the_brief_did_not_ask_for_is_dropped(tmp_path: Path) -> None:
    drafters = DrafterPool(reviews_every_case=True)

    exit_code = run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    assert exit_code == 0
    reviewed = Counter(case.split for case in read_drafted_cases(tmp_path) if case.scripted_review)
    assert reviewed == {'test': 2, 'dev': 1}


def _schema_keys(schema: object) -> set[str]:
    if isinstance(schema, dict):
        return set(schema) | {key for value in schema.values() for key in _schema_keys(value)}
    if isinstance(schema, list):
        return {key for value in schema for key in _schema_keys(value)}
    return set()


def _object_schemas(schema: object) -> list[dict[str, Any]]:
    if isinstance(schema, list):
        return [found for value in schema for found in _object_schemas(value)]
    if not isinstance(schema, dict):
        return []
    nested = [found for value in schema.values() for found in _object_schemas(value)]
    return [schema, *nested] if 'properties' in schema else nested


def test_a_drafter_set_to_json_schema_gets_a_schema_bedrock_can_lock_its_output_to(
    tmp_path: Path,
) -> None:
    drafters = DrafterPool()

    exit_code = run(
        drafting_arguments(tmp_path),
        drafting_settings(json_schema_keys=('fake_gpt',)),
        drafters.build,
    )

    assert exit_code == 0
    gpt_calls = drafters.models['fake_gpt'].structured_output_calls
    assert {method for method, _ in gpt_calls} == {'json_schema'}
    locked_schema = gpt_calls[0][1]
    assert {'minimum', 'exclusiveMinimum', 'minLength', 'const', 'oneOf', '$ref'}.isdisjoint(
        _schema_keys(locked_schema)
    )
    assert all(
        schema.get('additionalProperties') is False for schema in _object_schemas(locked_schema)
    )
    submission_types = [
        option['properties']['type']['enum']
        for option in locked_schema['properties']['submission']['anyOf']
    ]
    assert submission_types == [['expense_claim'], ['supplier_invoice']]
    claude_calls = drafters.models['fake_claude'].structured_output_calls
    assert {method for method, _ in claude_calls} == {'function_calling'}


def test_a_json_reply_with_text_around_the_object_is_still_read(tmp_path: Path) -> None:
    gpt_briefs = sum(key == 'fake_gpt' for key in ASSIGNED.values())
    drafters = DrafterPool(failures={'fake_gpt': ('decorated_json', gpt_briefs)})

    exit_code = run(
        drafting_arguments(tmp_path),
        drafting_settings(json_schema_keys=('fake_gpt',)),
        drafters.build,
    )

    assert exit_code == 0
    assert len(drafters.models['fake_gpt'].received) == gpt_briefs


@pytest.mark.parametrize('failure', ['omitted_nulls', 'empty_nulls'])
def test_a_draft_that_leaves_out_or_empties_its_null_fields_is_read_with_them_as_null(
    tmp_path: Path, failure: FailureKind
) -> None:
    nova_briefs = sum(key == 'fake_nova' for key in ASSIGNED.values())
    drafters = DrafterPool(failures={'fake_nova': (failure, nova_briefs)})

    exit_code = run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    assert exit_code == 0
    assert len(drafters.models['fake_nova'].received) == nova_briefs


def test_drafters_are_told_to_leave_out_fields_that_do_not_apply(tmp_path: Path) -> None:
    drafters = DrafterPool()

    run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    system_prompt = str(drafters.models['fake_nova'].received[0][0].content)
    assert 'Leave out any field that does not apply' in system_prompt
    assert (
        'Judgement findings need a reading of free text: WEEKEND_SPEND_UNJUSTIFIED,'
        ' PURPOSE_UNCLEAR, RECEIPT_MISMATCH, ALCOHOL_NOT_PERMITTED.'
    ) in system_prompt


def test_an_injection_prompt_tells_the_drafter_to_put_the_payload_verbatim_in_its_field(
    tmp_path: Path,
) -> None:
    drafters = DrafterPool()

    run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    injection = INJECTION_BRIEF.intent.injection
    assert injection is not None
    prompts = [
        str(prompt[1].content)
        for model in drafters.models.values()
        for prompt in model.received
        if brief_in(prompt)['slice'] == 'injection'
    ]
    assert len(prompts) == 1
    assert (
        f'- Put this injected instruction in {injection.field} word for word, and write the'
        f' rest of the case around it: {injection.payload}\n'
    ) in prompts[0] + '\n'


def test_drafters_are_told_not_to_write_placeholders_or_supplier_references_as_merchants(
    tmp_path: Path,
) -> None:
    drafters = DrafterPool()

    run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    system_prompt = str(drafters.models['fake_nova'].received[0][0].content)
    assert (
        'Write no placeholder text, such as Merchant Name, Item 1, Item Description or XYZ:'
        ' write what a real submission would say. A merchant is the name a business trades'
        ' under, never a supplier reference from the company records.'
    ) in system_prompt
