import argparse
import json
import sys
from pathlib import Path

from spendguard.eval.cases import load_slice_plan, read_case_directory
from spendguard.eval.validator import validate_dataset
from spendguard.policy.loader import POLICY_PATH, load_policy
from spendguard.seed.loader import SEED_DIRECTORY, read_seed_directory

CASE_DIRECTORY = Path('evals/cases')
SLICE_PLAN_PATH = Path('evals/slice_plan.yaml')
LOCK_PATH = Path('evals/dataset.lock.json')


def read_locked_hash(lock_path: Path) -> str:
    locked_hash: str = json.loads(lock_path.read_text())['dataset_hash']
    return locked_hash


def dataset_argument_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cases', type=Path, default=CASE_DIRECTORY)
    parser.add_argument('--plan', type=Path, default=SLICE_PLAN_PATH)
    parser.add_argument('--policy', type=Path, default=POLICY_PATH)
    parser.add_argument('--seed', type=Path, default=SEED_DIRECTORY)
    parser.add_argument('--lock', type=Path, default=LOCK_PATH)
    return parser


def main(arguments: list[str]) -> int:
    parser = dataset_argument_parser('Validate the eval dataset and write its lock file')
    options = parser.parse_args(arguments)
    report = validate_dataset(
        read_case_directory(options.cases),
        load_policy(options.policy),
        read_seed_directory(options.seed),
        load_slice_plan(options.plan),
    )
    if report.dataset_hash is None:
        for problem in report.problems:
            print(problem, file=sys.stderr)
        print('Not frozen: fix the problems above and run it again', file=sys.stderr)
        return 1
    options.lock.write_text(json.dumps({'dataset_hash': report.dataset_hash}, indent=2) + '\n')
    print(f'Frozen {options.cases} as {report.dataset_hash} in {options.lock}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
