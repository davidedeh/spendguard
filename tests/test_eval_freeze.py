import shutil
from pathlib import Path

import pytest
import yaml

from spendguard.eval.cases import load_slice_plan, read_case_directory
from spendguard.eval.freeze import main, read_locked_hash
from spendguard.eval.validator import validate_dataset
from spendguard.policy.loader import load_policy
from spendguard.seed.loader import read_seed_directory

FIXTURES = Path(__file__).parent / 'fixtures'


def freeze_arguments(case_directory: Path, lock_path: Path) -> list[str]:
    return [
        '--cases',
        str(case_directory),
        '--plan',
        str(FIXTURES / 'eval' / 'slice_plan.yaml'),
        '--policy',
        str(FIXTURES / 'policy' / 'valid_policy.yaml'),
        '--seed',
        str(FIXTURES / 'seed'),
        '--lock',
        str(lock_path),
    ]


def test_freeze_refuses_an_invalid_dataset_and_writes_no_lock_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case_directory = tmp_path / 'cases'
    shutil.copytree(FIXTURES / 'eval' / 'cases', case_directory)
    broken_case = yaml.safe_load((case_directory / 'test-alcohol.yaml').read_text())
    broken_case['submission']['claimant'] = 'EMP-999'
    (case_directory / 'test-alcohol.yaml').write_text(yaml.safe_dump(broken_case))
    lock_path = tmp_path / 'dataset.lock.json'

    exit_code = main(freeze_arguments(case_directory, lock_path))

    assert exit_code == 1
    assert not lock_path.exists()
    assert 'test-alcohol.yaml: unknown employee EMP-999' in capsys.readouterr().err


def test_freeze_writes_the_dataset_hash_to_the_lock_file_for_a_valid_dataset(
    tmp_path: Path,
) -> None:
    case_directory = FIXTURES / 'eval' / 'cases'
    lock_path = tmp_path / 'dataset.lock.json'
    expected_hash = validate_dataset(
        read_case_directory(case_directory),
        load_policy(FIXTURES / 'policy' / 'valid_policy.yaml'),
        read_seed_directory(FIXTURES / 'seed'),
        load_slice_plan(FIXTURES / 'eval' / 'slice_plan.yaml'),
    ).dataset_hash

    exit_code = main(freeze_arguments(case_directory, lock_path))

    assert exit_code == 0
    assert expected_hash is not None
    assert read_locked_hash(lock_path) == expected_hash
