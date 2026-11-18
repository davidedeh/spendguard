import asyncio
import dataclasses
import datetime
import json
import sys
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

from spendguard import settings
from spendguard.eval.case_runner import (
    EvalResources,
    SeededCases,
    eval_graph,
    first_completed_results,
    run_cases,
    seeded_cases,
)
from spendguard.eval.cases import EvalCase, load_slice_plan
from spendguard.eval.dataset_runs import DatasetRun, dataset_name, upload_cases
from spendguard.eval.freeze import dataset_argument_parser, read_locked_hash
from spendguard.eval.metrics import (
    CaseFailure,
    CaseOutcome,
    CaseResult,
    judgement_positive_count,
    row_metrics,
    rule_metrics,
)
from spendguard.eval.report import (
    DATASET_NAME,
    CarriedMemorySection,
    EvalResults,
    RerunRecord,
    RowResult,
    RowsDirectory,
    case_result_from_record,
    dated_results_name,
    results_file_text,
    rewrite_results_after_rerun,
    write_row_results,
)
from spendguard.eval.runner import (
    README_PATH,
    REPORTED_SPLIT,
    RESULTS_DIRECTORY,
    EvalRefusedError,
    frozen_cases,
    indexed_eval_resources,
    paced_chat_models,
    refuse_unsafe_settings,
)
from spendguard.policy.loader import Policy, load_policy
from spendguard.seed.loader import read_seed_directory
from spendguard.settings import Settings
from spendguard.store import precedents

RERUNNABLE_ERROR_TYPES = frozenset(
    {
        'InternalServerException',
        'ModelNotReadyException',
        'ModelTimeoutException',
        'ServiceUnavailableException',
        'ThrottlingException',
    }
)
RERUNS_FILE_NAME = 'reruns.json'
MEMORY_SECTION_START = '## Memory on versus off\n\n'
MEMORY_SECTION_END = '\n\n## Failed cases'
RERUN_RECORDS = TypeAdapter(tuple[RerunRecord, ...])
ISO_DATE_LENGTH = len('YYYY-MM-DD')


class RerunRefusedError(EvalRefusedError):
    pass


@dataclass(frozen=True)
class FinishedRun:
    results_path: Path
    rows_directory: Path
    measured_on: datetime.date
    run_identifier: str
    results_by_row: tuple[tuple[CaseResult, ...], ...]
    memory_section: CarriedMemorySection
    reruns: tuple[RerunRecord, ...]


def _run_date(run_name: str) -> datetime.date:
    date_text = run_name.removeprefix(f'{DATASET_NAME}-')[:ISO_DATE_LENGTH]
    try:
        return datetime.date.fromisoformat(date_text)
    except ValueError as error:
        raise RerunRefusedError(f'{run_name} is not named like a {DATASET_NAME} run') from error


def read_finished_run(results_directory: Path, run_name: str, row_count: int) -> FinishedRun:
    results_path = results_directory / f'{run_name}.md'
    rows_directory = results_directory / run_name
    if not results_path.is_file():
        raise RerunRefusedError(f'There is no results file at {results_path}')
    row_file_count = len(list(rows_directory.glob('row-*.json')))
    if row_file_count != row_count:
        raise RerunRefusedError(
            f'{rows_directory} has {row_file_count} row files, but the settings have'
            f' {row_count} eval rows'
        )
    records = [
        json.loads((rows_directory / f'row-{row_number}.json').read_text())
        for row_number in range(1, row_count + 1)
    ]
    run_identifier: str = records[0]['run_identifier']
    measured_on = _run_date(run_name)
    if dated_results_name(measured_on, run_identifier, None) != run_name:
        raise RerunRefusedError(f'{run_name} is not a full {DATASET_NAME} run')
    reruns_path = rows_directory / RERUNS_FILE_NAME
    _, _, after_memory_heading = results_path.read_text().partition(MEMORY_SECTION_START)
    memory_text, _, _ = after_memory_heading.partition(MEMORY_SECTION_END)
    return FinishedRun(
        results_path=results_path,
        rows_directory=rows_directory,
        measured_on=measured_on,
        run_identifier=run_identifier,
        results_by_row=tuple(
            tuple(case_result_from_record(entry) for entry in record['cases']) for record in records
        ),
        memory_section=CarriedMemorySection(lines=tuple(memory_text.split('\n'))),
        reruns=(
            RERUN_RECORDS.validate_json(reruns_path.read_text()) if reruns_path.is_file() else ()
        ),
    )


@dataclass(frozen=True)
class FrozenDataset:
    loaded_settings: Settings
    cases: Sequence[EvalCase]
    policy: Policy
    dataset_hash: str


def rebuilt_results(run: FinishedRun, dataset: FrozenDataset) -> EvalResults:
    loaded_settings, cases, policy = dataset.loaded_settings, dataset.cases, dataset.policy
    return EvalResults(
        measured_on=run.measured_on,
        dataset_hash=dataset.dataset_hash,
        run_identifier=run.run_identifier,
        case_count=len(cases),
        judgement_positive_count=judgement_positive_count(cases, policy),
        cases_per_row=None,
        retrieval_top_k=loaded_settings.retrieval_top_k,
        judgement_max_tool_calls=loaded_settings.judgement_max_tool_calls,
        rows=tuple(
            RowResult(
                row=row,
                model=loaded_settings.chat_models[row.model_key],
                price=loaded_settings.model_prices[row.model_key],
                metrics=row_metrics(
                    cases, results, policy, loaded_settings.model_prices[row.model_key]
                ),
            )
            for row, results in zip(loaded_settings.eval_rows, run.results_by_row, strict=True)
        ),
        rule_metrics=rule_metrics(cases, first_completed_results(run.results_by_row), policy),
        memory_comparison=run.memory_section,
        reruns=run.reruns,
    )


def refuse_an_unfaithful_rebuild(rebuilt: EvalResults, results_path: Path) -> None:
    expected_lines = results_file_text(rebuilt).split('\n')
    actual_lines = results_path.read_text().split('\n')
    if expected_lines == actual_lines:
        return
    first_difference = next(
        (
            line_number
            for line_number, (expected, actual) in enumerate(
                zip(expected_lines, actual_lines, strict=False), start=1
            )
            if expected != actual
        ),
        min(len(expected_lines), len(actual_lines)) + 1,
    )
    raise RerunRefusedError(
        f'{results_path} does not match a rebuild from its row files and the current settings'
        f' (first difference on line {first_difference}), so a rerun cannot merge into it'
    )


def rerunnable_failures(run: FinishedRun) -> dict[int, list[CaseFailure]]:
    failures_by_row = {
        row_number: [
            result
            for result in results
            if isinstance(result, CaseFailure) and result.error_type in RERUNNABLE_ERROR_TYPES
        ]
        for row_number, results in enumerate(run.results_by_row, start=1)
    }
    return {row_number: failures for row_number, failures in failures_by_row.items() if failures}


def _rerun_identifier(run: FinishedRun, row_number: int) -> str:
    return f'{run.run_identifier}:row-{row_number}:rerun-{uuid.uuid4().hex}'


async def _rerun_rows(
    run: FinishedRun,
    failures_by_row: dict[int, list[CaseFailure]],
    resources: EvalResources,
    dataset: FrozenDataset,
    rerun_on: datetime.date,
) -> AsyncIterator[tuple[int, FinishedRun]]:
    loaded_settings, cases = resources.settings, dataset.cases
    tracing = resources.tracing
    chat_model_for = paced_chat_models(loaded_settings)
    chat_models = {
        row_number: chat_model_for(loaded_settings.eval_rows[row_number - 1])
        for row_number in failures_by_row
    }
    planned = seeded_cases(cases, resources.policy).planned
    for row_number, failures in failures_by_row.items():
        row = loaded_settings.eval_rows[row_number - 1]
        failed_case_ids = {failure.case_id for failure in failures}
        rerun_identifier = _rerun_identifier(run, row_number)
        rerun_cases = [case for case in cases if case.id in failed_case_ids]
        dataset_run = (
            None
            if tracing is None
            else DatasetRun(
                client=tracing.client,
                name=f'{run.measured_on} {row.label} {rerun_identifier}',
                items_by_case_id=await asyncio.to_thread(
                    upload_cases, tracing.client, dataset_name(dataset.dataset_hash), rerun_cases
                ),
            )
        )
        await precedents.clear_precedents(resources.connection_pool)
        rerun = await run_cases(
            SeededCases(cases=rerun_cases, planned=planned),
            eval_graph(row, chat_models[row_number], resources, precedent_memory=True),
            resources,
            rerun_identifier,
            dataset_run,
        )
        rerun_by_case_id = {result.case_id: result for result in rerun.results}
        results_by_row = list(run.results_by_row)
        results_by_row[row_number - 1] = tuple(
            rerun_by_case_id.get(result.case_id, result)
            for result in run.results_by_row[row_number - 1]
        )
        run = dataclasses.replace(
            run,
            results_by_row=tuple(results_by_row),
            reruns=(
                *run.reruns,
                *(
                    RerunRecord(
                        row_label=row.label,
                        case_id=failure.case_id,
                        replaced_error_type=failure.error_type,
                        replaced_error_message=failure.error_message,
                        rerun_identifier=rerun_identifier,
                        rerun_on=rerun_on,
                        requests_per_minute=loaded_settings.eval_requests_per_minute[row.model_key],
                        result=_rerun_result(rerun_by_case_id[failure.case_id]),
                    )
                    for failure in failures
                ),
            ),
        )
        yield row_number, run


def _save_rerun_row(
    merged_run: FinishedRun, row_number: int, dataset: FrozenDataset, readme_path: Path
) -> None:
    loaded_settings = dataset.loaded_settings
    row = loaded_settings.eval_rows[row_number - 1]
    write_row_results(
        RowsDirectory(merged_run.rows_directory, merged_run.run_identifier),
        row_number,
        row,
        loaded_settings.chat_models[row.model_key],
        merged_run.results_by_row[row_number - 1],
    )
    (merged_run.rows_directory / RERUNS_FILE_NAME).write_text(
        RERUN_RECORDS.dump_json(merged_run.reruns, indent=2).decode() + '\n'
    )
    rewrite_results_after_rerun(
        rebuilt_results(merged_run, dataset), merged_run.results_path.parent, readme_path
    )


def _rerun_result(result: CaseResult) -> str:
    return 'completed' if isinstance(result, CaseOutcome) else result.reason


def main(arguments: list[str]) -> int:
    parser = dataset_argument_parser(
        "Rerun the cases of a finished eval run that failed on the provider's side"
    )
    parser.add_argument('--results', type=Path, default=RESULTS_DIRECTORY)
    parser.add_argument('--readme', type=Path, default=README_PATH)
    parser.add_argument('--run', required=True)
    options = parser.parse_args(arguments)
    policy = load_policy(options.policy)
    seed_data = read_seed_directory(options.seed)
    try:
        cases = frozen_cases(
            options.cases, load_slice_plan(options.plan), policy, seed_data, options.lock
        )
        loaded_settings = settings.load_settings()
        refuse_unsafe_settings(loaded_settings)
        dataset = FrozenDataset(
            loaded_settings=loaded_settings,
            cases=[case for case in cases if case.split == REPORTED_SPLIT],
            policy=policy,
            dataset_hash=read_locked_hash(options.lock),
        )
        run = read_finished_run(options.results, options.run, len(loaded_settings.eval_rows))
        refuse_an_unfaithful_rebuild(rebuilt_results(run, dataset), run.results_path)
        failures_by_row = rerunnable_failures(run)
        if not failures_by_row:
            raise RerunRefusedError(f"No case in {options.run} failed on the provider's side")
    except EvalRefusedError as error:
        print(f'Not rerunning: {error}', file=sys.stderr)
        return 1

    async def rerun() -> None:
        async with indexed_eval_resources(loaded_settings, policy, seed_data) as resources:
            async for row_number, merged_run in _rerun_rows(
                run, failures_by_row, resources, dataset, datetime.date.today()
            ):
                _save_rerun_row(merged_run, row_number, dataset, options.readme)

    asyncio.run(rerun())
    print(f'Rewrote {run.results_path}, its row files and the results table in {options.readme}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
