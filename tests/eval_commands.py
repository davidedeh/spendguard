from pathlib import Path

from spendguard.eval.freeze import main as freeze

FIXTURES = Path(__file__).parent / 'fixtures'
CASE_DIRECTORY = FIXTURES / 'eval' / 'cases'
PRICE = {
    'input_per_million_tokens_usd': '1.00',
    'output_per_million_tokens_usd': '5.00',
    'priced_on': '2026-11-02',
}
UNPACED_REQUESTS_PER_MINUTE = 600_000


def dataset_arguments(lock_path: Path) -> list[str]:
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
    ]


def locked_fixture_dataset(lock_path: Path) -> Path:
    exit_code = freeze(dataset_arguments(lock_path))
    assert exit_code == 0
    return lock_path


def eval_command_arguments(lock_path: Path, tmp_path: Path, readme_path: Path) -> list[str]:
    return [
        *dataset_arguments(lock_path),
        '--results',
        str(tmp_path / 'results'),
        '--readme',
        str(readme_path),
    ]
