import asyncio
import datetime
import sys
from pathlib import Path

from spendguard.db.database import async_connection_pool, branch_connection_info
from spendguard.eval.case_runner import EvalResources, run_eval
from spendguard.eval.cases import EvalCase, SlicePlan, load_slice_plan, read_case_directory
from spendguard.eval.freeze import dataset_argument_parser, read_locked_hash
from spendguard.eval.report import write_results
from spendguard.eval.validator import validate_dataset
from spendguard.models.factory import build_chat_model, build_embedder
from spendguard.policy.loader import Policy, load_policy
from spendguard.retrieval.policy_index import index_policy
from spendguard.seed.loader import read_seed_directory
from spendguard.seed.schema import SeedData
from spendguard.settings import load_settings

RESULTS_DIRECTORY = Path('evals/results')
README_PATH = Path('README.md')
REPORTED_SPLIT = 'test'


class DatasetRefusedError(Exception):
    pass


def frozen_cases(
    case_directory: Path, plan: SlicePlan, policy: Policy, seed_data: SeedData, lock_path: Path
) -> tuple[EvalCase, ...]:
    if not lock_path.is_file():
        raise DatasetRefusedError(f'There is no lock file at {lock_path}: run make freeze first')
    cases = read_case_directory(case_directory)
    report = validate_dataset(cases, policy, seed_data, plan)
    locked_hash = read_locked_hash(lock_path)
    if report.dataset_hash != locked_hash:
        raise DatasetRefusedError(
            f'{case_directory} hashes to {report.dataset_hash}, which does not match the lock'
            f' {locked_hash} in {lock_path}'
        )
    return tuple(EvalCase.model_validate(content) for content in cases.values())


async def _measure(
    cases: tuple[EvalCase, ...],
    policy: Policy,
    seed_data: SeedData,
    dataset_hash: str,
    output: tuple[Path, Path],
) -> Path:
    settings = load_settings()
    connection_pool = async_connection_pool(branch_connection_info(settings))
    await connection_pool.open(wait=True)
    try:
        embedder = build_embedder(settings)
        async with connection_pool.connection() as connection:
            await index_policy(connection, policy, embedder)
        results = await run_eval(
            cases,
            EvalResources(settings, policy, seed_data, connection_pool, embedder),
            lambda row: build_chat_model(settings, row.model_key),
            measured_on=datetime.date.today(),
            dataset_hash=dataset_hash,
        )
    finally:
        await connection_pool.close()
    results_directory, readme_path = output
    return write_results(results, results_directory, readme_path)


def main(arguments: list[str]) -> int:
    parser = dataset_argument_parser('Run the eval rows against the frozen dataset')
    parser.add_argument('--results', type=Path, default=RESULTS_DIRECTORY)
    parser.add_argument('--readme', type=Path, default=README_PATH)
    options = parser.parse_args(arguments)
    policy = load_policy(options.policy)
    seed_data = read_seed_directory(options.seed)
    try:
        cases = frozen_cases(
            options.cases, load_slice_plan(options.plan), policy, seed_data, options.lock
        )
    except DatasetRefusedError as error:
        print(f'Not reporting: {error}', file=sys.stderr)
        return 1
    results_path = asyncio.run(
        _measure(
            tuple(case for case in cases if case.split == REPORTED_SPLIT),
            policy,
            seed_data,
            read_locked_hash(options.lock),
            (options.results, options.readme),
        )
    )
    print(f'Wrote {results_path} and the results table in {options.readme}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
