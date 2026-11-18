import json
from pathlib import Path
from typing import Any, NoReturn

import pytest
import yaml

import spendguard.eval.runner
import spendguard.settings
from settings_factory import make_test_settings
from spendguard.eval.case_runner import pre_flight_cases
from spendguard.eval.cases import EvalCase, load_slice_plan
from spendguard.eval.freeze import CASE_DIRECTORY as FROZEN_CASE_DIRECTORY
from spendguard.eval.freeze import main as freeze
from spendguard.eval.runner import DatasetRefusedError, frozen_cases
from spendguard.eval.runner import main as run_eval_command
from spendguard.policy.loader import load_policy
from spendguard.seed.loader import read_seed_directory
from spendguard.settings import ChatModelSettings

FIXTURES = Path(__file__).parent / 'fixtures'
CASE_DIRECTORY = FIXTURES / 'eval' / 'cases'
FIXTURE_POLICY = load_policy(FIXTURES / 'policy' / 'valid_policy.yaml')
FIXTURE_SEED = read_seed_directory(FIXTURES / 'seed')
FIXTURE_PLAN = load_slice_plan(FIXTURES / 'eval' / 'slice_plan.yaml')


def locked_fixture_dataset(lock_path: Path) -> Path:
    exit_code = freeze(
        [
            '--cases',
            str(CASE_DIRECTORY),
            '--plan',
            str(FIXTURES / 'eval' / 'slice_plan.yaml'),
            '--policy',
            str(FIXTURES / 'policy' / 'valid_policy.yaml'),
            '--seed',
            str(FIXTURES / 'seed'),
            '--lock',
            str(lock_path),
        ]
    )
    assert exit_code == 0
    return lock_path


def test_the_runner_refuses_a_dataset_whose_hash_differs_from_the_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / 'dataset.lock.json'
    lock_path.write_text(json.dumps({'dataset_hash': '0' * 64}))

    with pytest.raises(DatasetRefusedError, match='does not match the lock'):
        frozen_cases(CASE_DIRECTORY, FIXTURE_PLAN, FIXTURE_POLICY, FIXTURE_SEED, lock_path)


def test_the_runner_refuses_when_there_is_no_lock_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetRefusedError, match='no lock file at'):
        frozen_cases(
            CASE_DIRECTORY, FIXTURE_PLAN, FIXTURE_POLICY, FIXTURE_SEED, tmp_path / 'missing.json'
        )


def test_the_runner_returns_every_case_of_a_dataset_matching_its_lock(tmp_path: Path) -> None:
    lock_path = locked_fixture_dataset(tmp_path / 'dataset.lock.json')

    cases = frozen_cases(CASE_DIRECTORY, FIXTURE_PLAN, FIXTURE_POLICY, FIXTURE_SEED, lock_path)

    assert sorted(case.id for case in cases) == sorted(
        path.stem for path in CASE_DIRECTORY.glob('*.yaml')
    )


def test_the_eval_command_on_a_changed_dataset_exits_1_and_writes_no_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lock_path = tmp_path / 'dataset.lock.json'
    lock_path.write_text(json.dumps({'dataset_hash': '0' * 64}))
    readme_path = tmp_path / 'README.md'
    readme_path.write_text('unchanged\n')

    exit_code = run_eval_command(
        [
            '--cases',
            str(CASE_DIRECTORY),
            '--plan',
            str(FIXTURES / 'eval' / 'slice_plan.yaml'),
            '--policy',
            str(FIXTURES / 'policy' / 'valid_policy.yaml'),
            '--seed',
            str(FIXTURES / 'seed'),
            '--lock',
            str(lock_path),
            '--results',
            str(tmp_path / 'results'),
            '--readme',
            str(readme_path),
        ]
    )

    assert exit_code == 1
    assert 'does not match the lock' in capsys.readouterr().err
    assert not (tmp_path / 'results').exists()
    assert readme_path.read_text() == 'unchanged\n'


def eval_command_arguments(lock_path: Path, tmp_path: Path, readme_path: Path) -> list[str]:
    return [
        '--cases',
        str(CASE_DIRECTORY),
        '--plan',
        str(FIXTURES / 'eval' / 'slice_plan.yaml'),
        '--policy',
        str(FIXTURES / 'policy' / 'valid_policy.yaml'),
        '--seed',
        str(FIXTURES / 'seed'),
        '--lock',
        str(lock_path),
        '--results',
        str(tmp_path / 'results'),
        '--readme',
        str(readme_path),
    ]


def refuse_model_calls(*arguments: object) -> NoReturn:
    raise AssertionError('the eval built a model before refusing')


PRICE = {
    'input_per_million_tokens_usd': '1.00',
    'output_per_million_tokens_usd': '5.00',
    'priced_on': '2026-11-02',
}
SECOND_MODEL = ChatModelSettings(
    provider='fake',
    model_id='fake-second',
    structured_output_method='function_calling',
    temperature=0.0,
)
TWO_ROWS = [
    {'model_key': 'fake_primary', 'label': 'Primary', 'policy_context': 'retrieved_clauses'},
    {'model_key': 'fake_second', 'label': 'Second', 'policy_context': 'retrieved_clauses'},
]


@pytest.mark.parametrize(
    ('overrides', 'refusal'),
    [
        (
            {'model_prices': {'fake_primary': PRICE}},
            "Eval row 'Second' uses chat model 'fake_second', which has no price in MODEL_PRICES",
        ),
        (
            {
                'model_prices': {'fake_primary': PRICE, 'fake_second': PRICE},
                'database_url': 'postgresql://someone@pooler.invalid:5432/postgres',
            },
            'DATABASE_URL is set',
        ),
    ],
    ids=['a row without a price', 'a database given by URL'],
)
def test_the_eval_command_refuses_unsafe_settings_before_any_model_call(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    refusal: str,
) -> None:
    lock_path = locked_fixture_dataset(tmp_path / 'dataset.lock.json')
    readme_path = tmp_path / 'README.md'
    readme_path.write_text('unchanged\n')
    settings = make_test_settings(
        chat_models=make_test_settings().chat_models | {'fake_second': SECOND_MODEL},
        eval_rows=TWO_ROWS,
        **overrides,
    )
    monkeypatch.setattr(spendguard.settings, 'load_settings', lambda: settings)
    monkeypatch.setattr(spendguard.eval.runner, 'build_embedder', refuse_model_calls)
    monkeypatch.setattr(spendguard.eval.runner, 'build_chat_model', refuse_model_calls)

    exit_code = run_eval_command(eval_command_arguments(lock_path, tmp_path, readme_path))

    assert exit_code == 1
    assert refusal in capsys.readouterr().err
    assert not (tmp_path / 'results').exists()
    assert readme_path.read_text() == 'unchanged\n'


def test_a_pre_flight_writes_its_row_files_and_results_but_leaves_the_readme_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lock_path = locked_fixture_dataset(tmp_path / 'dataset.lock.json')
    readme_path = tmp_path / 'README.md'
    readme_path.write_text('unchanged\n')
    settings = make_test_settings(eval_rows=TWO_ROWS[:1], model_prices={'fake_primary': PRICE})
    monkeypatch.setattr(spendguard.settings, 'load_settings', lambda: settings)

    exit_code = run_eval_command(
        [*eval_command_arguments(lock_path, tmp_path, readme_path), '--cases-per-row', '2']
    )

    assert exit_code == 0
    [results_file] = (tmp_path / 'results').glob('*.md')
    assert results_file.name.endswith('-preflight.md')
    [row_file] = (tmp_path / 'results' / results_file.stem).glob('*.json')
    assert sorted(case['case_id'] for case in json.loads(row_file.read_text())['cases']) == [
        'test-alcohol',
        'test-clean',
        'test-hard-negative',
        'test-injection',
        'test-memory-follow-up',
        'test-missing-po',
        'test-purpose-unclear',
    ]
    assert readme_path.read_text() == 'unchanged\n'
    assert 'the README is unchanged' in capsys.readouterr().out


def test_two_pre_flights_on_the_same_day_keep_each_runs_results_and_row_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = locked_fixture_dataset(tmp_path / 'dataset.lock.json')
    readme_path = tmp_path / 'README.md'
    readme_path.write_text('unchanged\n')
    settings = make_test_settings(eval_rows=TWO_ROWS[:1], model_prices={'fake_primary': PRICE})
    monkeypatch.setattr(spendguard.settings, 'load_settings', lambda: settings)
    arguments = [*eval_command_arguments(lock_path, tmp_path, readme_path), '--cases-per-row', '1']

    exit_codes = [run_eval_command(arguments), run_eval_command(arguments)]

    assert exit_codes == [0, 0]
    results_files = sorted((tmp_path / 'results').glob('*.md'))
    assert len(results_files) == 2
    run_identifiers = set()
    for results_file in results_files:
        [row_file] = (tmp_path / 'results' / results_file.stem).glob('*.json')
        run_identifier = json.loads(row_file.read_text())['run_identifier']
        assert results_file.stem.endswith(f'-{run_identifier[:8]}-preflight')
        run_identifiers.add(run_identifier)
    assert len(run_identifiers) == 2


def frozen_test_split() -> list[EvalCase]:
    cases = [
        EvalCase.model_validate(yaml.safe_load(path.read_text()))
        for path in sorted(FROZEN_CASE_DIRECTORY.glob('*.yaml'))
    ]
    return [case for case in cases if case.split == 'test']


def test_a_pre_flight_takes_whole_rounds_of_one_case_per_slice_with_each_follow_up_s_pair() -> None:
    cases = frozen_test_split()

    one_round = [case.id for case in pre_flight_cases(cases, 2)]
    two_rounds = [case.id for case in pre_flight_cases(cases, 7)]
    three_rounds = [case.id for case in pre_flight_cases(cases, 13)]

    assert one_round == [
        'test-clean-01',
        'test-hard-negative-01',
        'test-injection-01',
        'test-judgement-positive-01',
        'test-memory-follow-up-01',
        'test-rule-coverage-01',
    ]
    assert len(two_rounds) == 12
    assert {'test-memory-follow-up-02', 'test-judgement-positive-02'} <= set(two_rounds)
    assert {'test-memory-follow-up-03', 'test-judgement-positive-05'} <= set(three_rounds)
    assert 'test-judgement-positive-04' not in three_rounds
    assert {case.slice for case in pre_flight_cases(cases, 2)} == {case.slice for case in cases}
