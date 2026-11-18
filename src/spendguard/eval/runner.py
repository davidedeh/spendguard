import asyncio
import datetime
import sys
import uuid
from pathlib import Path

from spendguard import settings
from spendguard.db.database import async_connection_pool, branch_connection_info, opened_pool
from spendguard.eval.case_runner import EvalResources, EvalRunOptions, run_eval
from spendguard.eval.cases import EvalCase, SlicePlan, load_slice_plan, read_case_directory
from spendguard.eval.freeze import dataset_argument_parser, read_locked_hash
from spendguard.eval.report import EvalResults, dated_results_name, write_results
from spendguard.eval.validator import validate_dataset
from spendguard.models.factory import build_chat_model, build_embedder
from spendguard.observability.tracing import create_tracing
from spendguard.policy.loader import Policy, load_policy
from spendguard.retrieval.policy_index import index_policy
from spendguard.seed.loader import read_seed_directory
from spendguard.seed.schema import SeedData
from spendguard.settings import Settings

RESULTS_DIRECTORY = Path('evals/results')
README_PATH = Path('README.md')
REPORTED_SPLIT = 'test'


class EvalRefusedError(Exception):
    pass


class DatasetRefusedError(EvalRefusedError):
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


def refuse_unsafe_settings(loaded_settings: Settings) -> None:
    if loaded_settings.database_url is not None:
        raise EvalRefusedError(
            'DATABASE_URL is set: the eval deletes every precedent in the database it runs'
            ' against, so it runs only against the local branch database'
        )
    unpriced_rows = [
        row
        for row in loaded_settings.eval_rows
        if row.model_key not in loaded_settings.model_prices
    ]
    if unpriced_rows:
        raise EvalRefusedError(
            '; '.join(
                f'Eval row {row.label!r} uses chat model {row.model_key!r}, which has no price in'
                ' MODEL_PRICES'
                for row in unpriced_rows
            )
        )


async def _measure(
    loaded_settings: Settings,
    cases: tuple[EvalCase, ...],
    policy: Policy,
    seed_data: SeedData,
    options: EvalRunOptions,
) -> EvalResults:
    async with opened_pool(
        async_connection_pool(branch_connection_info(loaded_settings))
    ) as connection_pool:
        embedder = build_embedder(loaded_settings)
        async with connection_pool.connection() as connection:
            await index_policy(
                connection, policy, embedder, loaded_settings.embedding_model.model_id
            )
        return await run_eval(
            cases,
            EvalResources(
                loaded_settings,
                policy,
                seed_data,
                connection_pool,
                embedder,
                create_tracing(loaded_settings),
            ),
            lambda row: build_chat_model(loaded_settings, row.model_key),
            options,
        )


def main(arguments: list[str]) -> int:
    parser = dataset_argument_parser('Run the eval rows against the frozen dataset')
    parser.add_argument('--results', type=Path, default=RESULTS_DIRECTORY)
    parser.add_argument('--readme', type=Path, default=README_PATH)
    parser.add_argument('--cases-per-row', type=int)
    options = parser.parse_args(arguments)
    policy = load_policy(options.policy)
    seed_data = read_seed_directory(options.seed)
    try:
        cases = frozen_cases(
            options.cases, load_slice_plan(options.plan), policy, seed_data, options.lock
        )
        loaded_settings = settings.load_settings()
        refuse_unsafe_settings(loaded_settings)
    except EvalRefusedError as error:
        print(f'Not reporting: {error}', file=sys.stderr)
        return 1
    measured_on = datetime.date.today()
    run_identifier = uuid.uuid4().hex
    results = asyncio.run(
        _measure(
            loaded_settings,
            tuple(case for case in cases if case.split == REPORTED_SPLIT),
            policy,
            seed_data,
            EvalRunOptions(
                measured_on=measured_on,
                dataset_hash=read_locked_hash(options.lock),
                run_identifier=run_identifier,
                rows_directory=options.results
                / dated_results_name(measured_on, run_identifier, options.cases_per_row),
                cases_per_row=options.cases_per_row,
            ),
        )
    )
    results_path = write_results(results, options.results, options.readme)
    if options.cases_per_row is None:
        print(f'Wrote {results_path} and the results table in {options.readme}')
    else:
        print(f'Wrote {results_path}; the README is unchanged')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
