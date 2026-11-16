import json
from pathlib import Path

import pytest

from spendguard.eval.cases import load_slice_plan
from spendguard.eval.freeze import main as freeze
from spendguard.eval.runner import DatasetRefusedError, frozen_cases
from spendguard.eval.runner import main as run_eval_command
from spendguard.policy.loader import load_policy
from spendguard.seed.loader import read_seed_directory

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
