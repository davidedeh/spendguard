from collections import Counter
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from settings_factory import make_test_settings
from spendguard.eval.cases import DraftedBy, EvalCase, load_slice_plan, read_case_directory
from spendguard.eval.drafting import DraftedCase, plan_requests, run
from spendguard.models.fake import FakeChatModel, structured_response
from spendguard.policy.loader import POLICY_PATH, load_policy
from spendguard.settings import ModelSettings, Settings

FIXTURES = Path(__file__).parent / 'fixtures'
FIXTURE_POLICY_PATH = FIXTURES / 'policy' / 'valid_policy.yaml'
FIXTURE_POLICY = load_policy(FIXTURE_POLICY_PATH)
DRAFTER_MODEL_IDS = {
    'fake_claude': 'fake-claude-drafter',
    'fake_gpt': 'fake-gpt-drafter',
    'fake_gemini': 'fake-gemini-drafter',
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
    },
}


FailureKind = Literal['malformed', 'unknown_employee']


class DraftingFake(FakeChatModel):
    claimant: str = 'EMP-003'
    failing_calls: int = 0
    failure: FailureKind = 'malformed'

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        request = str(messages[-1].content).split('## Request')[-1]
        drafted_case = self.draft(request)
        message = structured_response(drafted_case)
        if len(self.received) <= self.failing_calls:
            if self.failure == 'malformed':
                del message.tool_calls[0]['args']['submission']
            else:
                failing_draft = drafted_case.model_copy(
                    update={
                        'submission': drafted_case.submission.model_copy(
                            update={'claimant': 'EMP-999'}
                        )
                    }
                )
                message = structured_response(failing_draft)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def draft(self, request: str) -> DraftedCase:
        requested_codes = [code for code in FIXTURE_POLICY.finding_codes if code in request]
        findings: list[dict[str, Any]] = [
            {
                'id': 'finding-1',
                'code': code,
                'gold_clauses': [FIXTURE_POLICY.finding_codes[code].clause.reference],
            }
            for code in requested_codes
        ]
        scripted_review = None
        if 'scripted review' in request:
            scripted_review = {
                'approver': 'EMP-001',
                'dispositions': [
                    {
                        'finding_id': 'finding-1',
                        'disposition': 'dismissed',
                        'reason': 'A standing arrangement agreed with the manager.',
                    }
                ],
            }
        return DraftedCase.model_validate(
            {
                'submission': {
                    'type': 'expense_claim',
                    'claimant': self.claimant,
                    'spent_on': '2026-10-06',
                    'category': 'subsistence',
                    'quantity': 1,
                    'merchant': f'{self.model_id} call {len(self.received)}',
                    'amount': '18.40',
                    'currency': 'GBP',
                    'claimed_gbp_per_unit': None,
                    'business_purpose': 'Working lunch with the audit team.',
                    'receipt_text': None,
                },
                'expected_findings': findings,
                'expected_route': 'human_review' if findings else 'auto_approve',
                'expected_recommendation': 'none' if findings else 'approve',
                'scripted_review': scripted_review,
            }
        )


class DrafterPool:
    def __init__(
        self,
        claimants: dict[str, str] | None = None,
        failures: dict[str, tuple[FailureKind, int]] | None = None,
    ) -> None:
        self.claimants = claimants or {}
        self.failures = failures or {}
        self.models: dict[str, DraftingFake] = {}

    def build(self, settings: Settings, model_key: str) -> BaseChatModel:
        failure, failing_calls = self.failures.get(model_key, ('malformed', 0))
        model = DraftingFake(
            model_id=settings.chat_models[model_key].model_id,
            claimant=self.claimants.get(model_key, 'EMP-003'),
            failing_calls=failing_calls,
            failure=failure,
        )
        self.models[model_key] = model
        return model


def drafting_settings() -> Settings:
    return make_test_settings(
        chat_models={
            key: ModelSettings(provider='fake', model_id=model_id)
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
        str(FIXTURES / 'seed'),
        '--lock',
        str(tmp_path / 'dataset.lock.json'),
    ]


def read_drafted_cases(tmp_path: Path) -> list[EvalCase]:
    return [
        EvalCase.model_validate(content)
        for content in read_case_directory(tmp_path / 'cases').values()
    ]


def test_every_slice_is_split_across_the_drafters_with_none_more_than_one_off_a_third(
    tmp_path: Path,
) -> None:
    exit_code = run(drafting_arguments(tmp_path), drafting_settings(), DrafterPool().build)

    assert exit_code == 0
    cases = read_drafted_cases(tmp_path)
    assert len(cases) == 28
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
    overall_counts = Counter(case.drafted_by.model_id for case in cases)
    assert sorted(overall_counts.values()) == [9, 9, 10]


def test_every_case_carries_the_drafted_by_of_the_model_that_produced_it(
    tmp_path: Path,
) -> None:
    run(drafting_arguments(tmp_path), drafting_settings(), DrafterPool().build)

    for case in read_drafted_cases(tmp_path):
        assert case.submission.type == 'expense_claim'
        producing_model_id = case.submission.merchant.split(' call ')[0]
        assert case.drafted_by == DraftedBy(provider='fake', model_id=producing_model_id)


def test_drafts_that_fail_validation_are_not_written_and_the_problems_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    drafters = DrafterPool(claimants={'fake_gpt': 'EMP-999'})

    exit_code = run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    assert exit_code == 1
    assert not (tmp_path / 'cases').exists()
    errors = capsys.readouterr().err
    assert 'test-judgement-positive-02.yaml: unknown employee EMP-999' in errors
    assert 'test-judgement-positive-01.yaml' not in errors


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


def test_prompts_hold_the_rendered_policy_and_seed_data_and_nothing_from_other_cases(
    tmp_path: Path,
) -> None:
    drafters = DrafterPool()

    run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    cases = read_drafted_cases(tmp_path)
    seed_references = ['EMP-001', 'EMP-002', 'EMP-003', 'SUP-001', 'SUP-002', 'PO-0001', 'CC-300']
    case_markers = [case.id for case in cases] + [
        case.submission.merchant for case in cases if case.submission.type == 'expense_claim'
    ]
    prompts = [prompt for model in drafters.models.values() for prompt in model.received]
    assert len(prompts) == 28
    contexts = set()
    for prompt in prompts:
        assert [message.type for message in prompt] == ['system', 'human']
        text = '\n'.join(str(message.content) for message in prompt)
        assert FIXTURE_POLICY.rendered_text in text
        for reference in seed_references:
            assert reference in text
        for marker in case_markers:
            assert marker not in text
        for code_marker in ('def ', 'import ', 'spendguard', 'class '):
            assert code_marker not in text
        contexts.add(str(prompt[1].content).split('## Request')[0])
    assert len(contexts) == 1


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


def test_prompts_name_each_finding_code_with_the_clause_it_rests_on(tmp_path: Path) -> None:
    drafters = DrafterPool()

    run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    prompt_text = str(drafters.models['fake_gemini'].received[0][1].content)
    assert '- ALCOHOL_NOT_PERMITTED, a judgement check, rests on §2.2' in prompt_text
    assert '- MISSING_PO, a rule check, rests on §3.2' in prompt_text


def test_prompts_name_each_expense_category_key_with_the_unit_its_quantity_counts(
    tmp_path: Path,
) -> None:
    drafters = DrafterPool()

    run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    prompt_text = str(drafters.models['fake_claude'].received[0][1].content)
    assert (
        '## Expense categories\n\n'
        'An expense claim names one of these category keys and a quantity: a whole number,'
        " at least 1, counting the unit the category's limit applies per.\n\n"
        '- subsistence (Meals): limit applies per person per day\n'
        '- accommodation (Hotels): limit applies per night\n'
        '- taxi (Taxis): limit applies per journey\n'
    ) in prompt_text


def test_rule_coverage_requests_spread_over_the_rule_codes_of_the_policy() -> None:
    plan = load_slice_plan(Path(__file__).parents[1] / 'evals' / 'slice_plan.yaml')
    policy = load_policy(POLICY_PATH)

    requests = plan_requests(plan, policy, tuple(DRAFTER_MODEL_IDS))

    rule_codes = Counter(
        request.finding_code
        for request in requests
        if request.split == 'test' and request.slice == 'rule_coverage'
    )
    assert rule_codes == Counter(
        [
            'OVER_CATEGORY_LIMIT',
            'MISSING_RECEIPT',
            'DUPLICATE_SUBMISSION',
            'SPLIT_PURCHASE',
            'NON_APPROVED_SUPPLIER',
            'MISSING_PO',
            'BUDGET_EXCEEDED',
            'CURRENCY_ANOMALY',
        ]
    )
    dev_rule_codes = [
        request.finding_code
        for request in requests
        if request.split == 'dev' and request.slice == 'rule_coverage'
    ]
    assert dev_rule_codes == ['OVER_CATEGORY_LIMIT', 'MISSING_RECEIPT']


def test_each_rule_coverage_prompt_names_its_one_rule_finding(tmp_path: Path) -> None:
    drafters = DrafterPool()

    run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    cases = read_drafted_cases(tmp_path)
    rule_coverage_codes = sorted(
        finding.code
        for case in cases
        if case.split == 'test' and case.slice == 'rule_coverage'
        for finding in case.expected_findings
    )
    assert rule_coverage_codes == [
        'DUPLICATE_SUBMISSION',
        'MISSING_RECEIPT',
        'OVER_CATEGORY_LIMIT',
        'SPLIT_PURCHASE',
    ]


@pytest.mark.parametrize('failure', ['malformed', 'unknown_employee'])
def test_a_draft_that_fails_once_is_retried_and_written(
    tmp_path: Path, failure: FailureKind
) -> None:
    drafters = DrafterPool(failures={'fake_gpt': (failure, 1)})

    exit_code = run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    assert exit_code == 0
    assert len(read_drafted_cases(tmp_path)) == 28
    assert len(drafters.models['fake_gpt'].received) == 10


@pytest.mark.parametrize(
    ('failure', 'problem'),
    [
        (
            'malformed',
            'test-judgement-positive-02.yaml: no valid draft after 3 attempts:'
            ' submission: Field required',
        ),
        ('unknown_employee', 'test-judgement-positive-02.yaml: unknown employee EMP-999'),
    ],
)
def test_a_draft_that_fails_three_times_blocks_the_write_and_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], failure: FailureKind, problem: str
) -> None:
    drafters = DrafterPool(failures={'fake_gpt': (failure, 3)})

    exit_code = run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    assert exit_code == 1
    assert not (tmp_path / 'cases').exists()
    assert len(drafters.models['fake_gpt'].received) == 11
    error_lines = capsys.readouterr().err.splitlines()
    assert problem in error_lines
    assert not any(line.startswith('test-hard-negative-02.yaml') for line in error_lines)
