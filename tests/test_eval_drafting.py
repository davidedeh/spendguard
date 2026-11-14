import json
from collections import Counter
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
from spendguard.eval.cases import DraftedBy, EvalCase, load_slice_plan, read_case_directory
from spendguard.eval.drafting import DraftedCase, plan_requests, run
from spendguard.models.fake import FakeChatModel, structured_response
from spendguard.policy.loader import POLICY_PATH, FindingCode, load_policy
from spendguard.settings import ChatModelSettings, Settings
from spendguard.submission import ExpenseClaim, SupplierInvoice

FIXTURES = Path(__file__).parent / 'fixtures'
FIXTURE_POLICY_PATH = FIXTURES / 'policy' / 'valid_policy.yaml'
FIXTURE_POLICY = load_policy(FIXTURE_POLICY_PATH)
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


FailureKind = Literal[
    'malformed',
    'unknown_employee',
    'wrong_code',
    'no_review',
    'decorated_json',
    'omitted_nulls',
    'empty_nulls',
]


class DraftingFake(FakeChatModel):
    claimant: str = 'EMP-003'
    failing_calls: int = 0
    failure: FailureKind = 'malformed'
    tool_schemas: list[Any] = Field(default_factory=list)
    reviews_every_case: bool = False
    drafts_invoices: bool = False
    failing_slice: str | None = None

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        self.tool_schemas.append(kwargs.get('tools'))
        request = str(messages[1].content).split('## Request')[-1].split('## Rejected draft')[0]
        drafted_case = self.draft(request)
        message = structured_response(drafted_case)
        failing = len(self.received) <= self.failing_calls or (
            self.failing_slice is not None and f'the {self.failing_slice} slice' in request
        )
        if failing and self.failure not in (
            'decorated_json',
            'omitted_nulls',
            'empty_nulls',
        ):
            if self.failure == 'malformed':
                del message.tool_calls[0]['args']['submission']
            elif self.failure == 'wrong_code':
                message = structured_response(self.draft(request, wrong_code=True))
            elif self.failure == 'no_review':
                message = structured_response(
                    drafted_case.model_copy(update={'scripted_review': None})
                )
            else:
                failing_draft = drafted_case.model_copy(
                    update={
                        'submission': drafted_case.submission.model_copy(
                            update={'claimant': 'EMP-999'}
                        )
                    }
                )
                message = structured_response(failing_draft)
        if self.failure == 'omitted_nulls' and len(self.received) <= self.failing_calls:
            message.tool_calls[0]['args'] = _without_nulls(message.tool_calls[0]['args'])
        if self.failure == 'empty_nulls' and len(self.received) <= self.failing_calls:
            message.tool_calls[0]['args'] = _nulls_as_empty_text(message.tool_calls[0]['args'])
        if self.structured_output_calls[-1][0] == 'json_schema':
            decoration = (
                '**'
                if self.failure == 'decorated_json' and len(self.received) <= self.failing_calls
                else ''
            )
            message = AIMessage(content=decoration + json.dumps(message.tool_calls[0]['args']))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def submission(self) -> dict[str, Any]:
        if self.drafts_invoices:
            return {
                'type': 'supplier_invoice',
                'supplier': 'SUP-001',
                'supplier_invoice_number': f'{self.model_id} call {len(self.received)}',
                'purchase_order': None,
                'cost_centre': 'CC-300',
                'issued_on': '2026-10-06',
                'amount': '240.00',
                'currency': 'GBP',
                'claimed_exchange_rate': None,
                'description': 'Printer toner for the finance office.',
            }
        return {
            'type': 'expense_claim',
            'claimant': self.claimant,
            'spent_on': '2026-10-06',
            'category': 'subsistence',
            'quantity': 1,
            'merchant': f'{self.model_id} call {len(self.received)}',
            'amount': '18.40',
            'currency': 'GBP',
            'claimed_exchange_rate': None,
            'business_purpose': 'Working lunch with the audit team.',
            'receipt_text': None,
        }

    def draft(self, request: str, wrong_code: bool = False) -> DraftedCase:
        requested_codes = [code for code in FIXTURE_POLICY.finding_codes if code in request]
        if wrong_code:
            requested_codes = [
                next(
                    other
                    for other, definition in FIXTURE_POLICY.finding_codes.items()
                    if other != code
                    and definition.check == FIXTURE_POLICY.finding_codes[code].check
                )
                for code in requested_codes
            ]
        findings: list[dict[str, Any]] = [
            {
                'id': 'finding-1',
                'code': code,
                'gold_clauses': [FIXTURE_POLICY.finding_codes[code].clause.reference],
            }
            for code in requested_codes
        ]
        scripted_review = None
        if findings and (self.reviews_every_case or 'scripted review' in request):
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
                'submission': self.submission(),
                'expected_findings': findings,
                'expected_route': 'human_review' if findings else 'auto_approve',
                'expected_recommendation': _recommendation_for(requested_codes),
                'scripted_review': scripted_review,
            }
        )


def _without_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_nulls(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_nulls(item) for item in value]
    return value


def _recommendation_for(codes: list[FindingCode]) -> str:
    if any(FIXTURE_POLICY.finding_codes[code].severity == 'high' for code in codes):
        return 'reject'
    return 'none' if codes else 'approve'


def _nulls_as_empty_text(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _nulls_as_empty_text(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_nulls_as_empty_text(item) for item in value]
    return '' if value is None else value


class DrafterPool:
    def __init__(
        self,
        claimants: dict[str, str] | None = None,
        failures: dict[str, tuple[FailureKind, int]] | None = None,
        reviews_every_case: bool = False,
        drafts_invoices: bool = False,
        failing_slice: tuple[FailureKind, str] | None = None,
    ) -> None:
        self.failing_slice = failing_slice
        self.reviews_every_case = reviews_every_case
        self.drafts_invoices = drafts_invoices
        self.claimants = claimants or {}
        self.failures = failures or {}
        self.models: dict[str, DraftingFake] = {}

    def build(self, settings: Settings, model_key: str) -> BaseChatModel:
        failure, failing_calls = self.failures.get(model_key, ('malformed', 0))
        if self.failing_slice is not None:
            failure = self.failing_slice[0]
        model = DraftingFake(
            model_id=settings.chat_models[model_key].model_id,
            claimant=self.claimants.get(model_key, 'EMP-003'),
            failing_calls=failing_calls,
            failure=failure,
            reviews_every_case=self.reviews_every_case,
            drafts_invoices=self.drafts_invoices and model_key == 'fake_nova',
            failing_slice=None if self.failing_slice is None else self.failing_slice[1],
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
        str(FIXTURES / 'seed'),
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
    drafters = DrafterPool(failing_slice=('unknown_employee', 'injection'))

    exit_code = run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    assert exit_code == 1
    assert not (tmp_path / 'cases').exists()
    errors = capsys.readouterr().err
    assert 'test-injection-01.yaml: unknown employee EMP-999' in errors
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

    prompt_text = str(drafters.models['fake_nova'].received[0][1].content)
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
            'test-injection-01.yaml: no valid draft after 3 attempts: submission: Field required',
        ),
        ('unknown_employee', 'test-injection-01.yaml: unknown employee EMP-999'),
    ],
)
def test_a_draft_every_drafter_fails_blocks_the_write_and_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], failure: FailureKind, problem: str
) -> None:
    drafters = DrafterPool(failing_slice=(failure, 'injection'))

    exit_code = run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    assert exit_code == 1
    assert not (tmp_path / 'cases').exists()
    attempts = sum(len(model.received) for model in drafters.models.values())
    assert attempts == 27 + 3 * len(DRAFTER_MODEL_IDS)
    error_lines = capsys.readouterr().err.splitlines()
    assert problem in error_lines
    assert not any(line.startswith('test-hard-negative-02.yaml') for line in error_lines)


def _schema_keys(schema: object) -> set[str]:
    if isinstance(schema, dict):
        return set(schema) | {key for value in schema.values() for key in _schema_keys(value)}
    if isinstance(schema, list):
        return {key for value in schema for key in _schema_keys(value)}
    return set()


def test_drafters_receive_a_schema_with_enum_tags_and_no_references(
    tmp_path: Path,
) -> None:
    drafters = DrafterPool()

    run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    [tool] = drafters.models['fake_claude'].tool_schemas[0]
    parameters = tool['function']['parameters']
    assert {'const', 'oneOf', 'discriminator', '$ref', '$defs'}.isdisjoint(_schema_keys(parameters))
    submission_types = [
        option['properties']['type']['enum']
        for option in parameters['properties']['submission']['anyOf']
    ]
    assert submission_types == [['expense_claim'], ['supplier_invoice']]
    for option in parameters['properties']['submission']['anyOf']:
        assert (
            'null when the submission is in the base currency'
            in (option['properties']['claimed_exchange_rate']['description'])
        )


@pytest.mark.parametrize(
    ('failure', 'problem'),
    [('malformed', 'submission: Field required'), ('unknown_employee', 'unknown employee EMP-999')],
)
def test_a_retry_tells_the_drafter_why_its_previous_draft_was_rejected(
    tmp_path: Path, failure: FailureKind, problem: str
) -> None:
    drafters = DrafterPool(failures={'fake_gpt': (failure, 1)})

    run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    first_prompt, retry_prompt, next_case_prompt = drafters.models['fake_gpt'].received[:3]
    first_text = str(first_prompt[1].content)
    retry_text = str(retry_prompt[1].content)
    assert retry_prompt[0] == first_prompt[0]
    assert retry_text.startswith(first_text)
    assert problem in retry_text.removeprefix(first_text)
    assert problem not in str(next_case_prompt[-1].content)


def test_a_scripted_review_the_request_did_not_ask_for_is_dropped(tmp_path: Path) -> None:
    drafters = DrafterPool(reviews_every_case=True)

    exit_code = run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    assert exit_code == 0
    reviewed = Counter(case.split for case in read_drafted_cases(tmp_path) if case.scripted_review)
    assert reviewed == {'test': 2, 'dev': 1}


@pytest.mark.parametrize(
    ('failure', 'drafter_key', 'problem'),
    [
        ('wrong_code', 'fake_gpt', 'the request asks for a {code} finding and the draft has none'),
        (
            'no_review',
            'fake_claude',
            'the request asks for a scripted review and the draft has none',
        ),
    ],
)
def test_a_draft_that_ignores_its_request_is_retried_with_the_reason(
    tmp_path: Path, failure: FailureKind, drafter_key: str, problem: str
) -> None:
    drafters = DrafterPool(failures={drafter_key: (failure, 1)})

    exit_code = run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    assert exit_code == 0
    first_prompt, retry_prompt = drafters.models[drafter_key].received[:2]
    first_text = str(first_prompt[1].content)
    requested_code = next(
        code
        for code in FIXTURE_POLICY.finding_codes
        if code in first_text.rsplit('## Request', maxsplit=1)[-1]
    )
    assert problem.format(code=requested_code) in str(retry_prompt[1].content)


def test_a_failed_run_keeps_its_valid_drafts_and_a_rerun_drafts_only_what_failed(
    tmp_path: Path,
) -> None:
    failing_run = run(
        drafting_arguments(tmp_path),
        drafting_settings(),
        DrafterPool(failing_slice=('unknown_employee', 'injection')).build,
    )
    rerun_drafters = DrafterPool()

    rerun = run(drafting_arguments(tmp_path), drafting_settings(), rerun_drafters.build)

    assert (failing_run, rerun) == (1, 0)
    rerun_prompts = [
        prompt for model in rerun_drafters.models.values() for prompt in model.received
    ]
    assert len(rerun_prompts) == 1
    assert 'the injection slice' in str(rerun_prompts[0][1].content)
    assert len(read_drafted_cases(tmp_path)) == 28
    assert not any((tmp_path / 'drafts').glob('*.yaml'))


def test_a_staged_draft_from_a_drafter_no_longer_configured_is_drafted_again(
    tmp_path: Path,
) -> None:
    run(
        drafting_arguments(tmp_path),
        drafting_settings(),
        DrafterPool(failing_slice=('unknown_employee', 'injection')).build,
    )
    replaced_settings = drafting_settings().model_copy(
        update={
            'chat_models': {
                **drafting_settings().chat_models,
                'fake_gpt': ChatModelSettings(
                    provider='fake',
                    model_id='fake-replacement-drafter',
                    structured_output_method='function_calling',
                ),
            }
        }
    )
    rerun_drafters = DrafterPool()

    exit_code = run(drafting_arguments(tmp_path), replaced_settings, rerun_drafters.build)

    assert exit_code == 0
    redrafted = {key: len(model.received) for key, model in rerun_drafters.models.items()}
    assert redrafted == {'fake_claude': 0, 'fake_gpt': 9, 'fake_nova': 0}


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
    assert {'minimum', 'exclusiveMinimum', 'minLength'}.isdisjoint(_schema_keys(locked_schema))
    assert all(
        schema.get('additionalProperties') is False for schema in _object_schemas(locked_schema)
    )
    claude_calls = drafters.models['fake_claude'].structured_output_calls
    assert {method for method, _ in claude_calls} == {'function_calling'}
    assert 'minimum' in _schema_keys(claude_calls[0][1])


def _object_schemas(schema: object) -> list[dict[str, Any]]:
    if isinstance(schema, list):
        return [found for value in schema for found in _object_schemas(value)]
    if not isinstance(schema, dict):
        return []
    nested = [found for value in schema.values() for found in _object_schemas(value)]
    return [schema, *nested] if 'properties' in schema else nested


def test_a_json_reply_with_text_around_the_object_is_still_read(tmp_path: Path) -> None:
    drafters = DrafterPool(failures={'fake_gpt': ('decorated_json', 9)})

    exit_code = run(
        drafting_arguments(tmp_path),
        drafting_settings(json_schema_keys=('fake_gpt',)),
        drafters.build,
    )

    assert exit_code == 0
    assert len(drafters.models['fake_gpt'].received) == 9


@pytest.mark.parametrize('failure', ['omitted_nulls', 'empty_nulls'])
@pytest.mark.parametrize('drafts_invoices', [False, True])
def test_a_draft_that_leaves_out_or_empties_its_null_fields_is_read_with_them_as_null(
    tmp_path: Path, drafts_invoices: bool, failure: FailureKind
) -> None:
    drafters = DrafterPool(failures={'fake_nova': (failure, 9)}, drafts_invoices=drafts_invoices)

    exit_code = run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    assert exit_code == 0
    assert len(drafters.models['fake_nova'].received) == 9
    nova_submissions = [
        case.submission
        for case in read_drafted_cases(tmp_path)
        if case.drafted_by.model_id == 'fake-nova-drafter'
    ]
    assert len(nova_submissions) == 9
    assert all(submission.claimed_exchange_rate is None for submission in nova_submissions)
    if drafts_invoices:
        assert all(
            isinstance(submission, SupplierInvoice) and submission.purchase_order is None
            for submission in nova_submissions
        )


def test_a_draft_still_failing_after_its_retries_is_not_saved(tmp_path: Path) -> None:
    drafters = DrafterPool(failing_slice=('unknown_employee', 'injection'))

    run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    staged = {path.name for path in (tmp_path / 'drafts').glob('*.yaml')}
    assert len(staged) == 27
    assert 'test-injection-01.yaml' not in staged


def test_drafters_are_told_to_leave_out_fields_that_do_not_apply(tmp_path: Path) -> None:
    drafters = DrafterPool()

    run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    system_prompt = str(drafters.models['fake_nova'].received[0][0].content)
    assert 'Leave out any field that does not apply' in system_prompt


def test_cases_a_drafter_cannot_draft_go_to_the_other_drafters_in_turn(tmp_path: Path) -> None:
    drafters = DrafterPool(failures={'fake_nova': ('unknown_employee', 1000)})

    exit_code = run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    assert exit_code == 0
    cases = read_drafted_cases(tmp_path)
    assert len(cases) == 28
    drafted_by = Counter(case.drafted_by.model_id for case in cases)
    assert drafted_by == {'fake-claude-drafter': 15, 'fake-gpt-drafter': 13}
    for case in cases:
        assert isinstance(case.submission, ExpenseClaim)
        assert case.submission.merchant.split(' call ')[0] == case.drafted_by.model_id


def test_prompts_state_the_route_rule_with_the_policy_threshold(tmp_path: Path) -> None:
    drafters = DrafterPool()

    run(drafting_arguments(tmp_path), drafting_settings(), drafters.build)

    prompt_text = str(drafters.models['fake_gpt'].received[0][1].content)
    assert (
        '## Route\n\n'
        'The expected route is auto_approve only when the case has no findings and its amount'
        ' in GBP is at or under GBP 250; otherwise it is human_review.'
    ) in prompt_text
