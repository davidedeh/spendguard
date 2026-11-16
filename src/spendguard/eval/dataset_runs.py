import asyncio
import contextvars
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langfuse import Langfuse
from langfuse.api import DatasetItem, NotFoundError
from langfuse.experiment import ExperimentItem

from spendguard.eval.cases import EvalCase
from spendguard.eval.metrics import CaseOutcome

type CaseRun = Callable[[EvalCase], Coroutine[Any, Any, CaseOutcome]]


def dataset_name(dataset_hash: str) -> str:
    return f'spendguard-eval-{dataset_hash}'


@dataclass(frozen=True)
class DatasetRun:
    client: Langfuse
    name: str
    items_by_case_id: Mapping[str, DatasetItem]


def upload_cases(client: Langfuse, name: str, cases: Sequence[EvalCase]) -> dict[str, DatasetItem]:
    try:
        client.get_dataset(name)
    except NotFoundError:
        client.create_dataset(name=name)
    item_ids = {case.id: f'{name}-{case.id}' for case in cases}
    for case in cases:
        client.create_dataset_item(
            dataset_name=name,
            id=item_ids[case.id],
            input=case.submission.model_dump(mode='json'),
            expected_output=case.model_dump(
                mode='json',
                include={
                    'expected_findings',
                    'expected_route',
                    'expected_recommendation',
                    'expected_clearance',
                },
            ),
            metadata={'case_id': case.id, 'slice': case.slice},
        )
    items_by_id = {item.id: item for item in client.get_dataset(name).items}
    return {case_id: items_by_id[item_id] for case_id, item_id in item_ids.items()}


async def run_as_dataset_run(
    run: DatasetRun, cases: Sequence[EvalCase], run_case: CaseRun
) -> list[CaseOutcome]:
    loop = asyncio.get_running_loop()
    cases_by_item_id = {run.items_by_case_id[case.id].id: case for case in cases}
    outcomes: dict[str, CaseOutcome] = {}
    failures: list[Exception] = []

    def task(*, item: ExperimentItem, **keyword_arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(item, DatasetItem):
            raise TypeError('a dataset run is given dataset items only')
        case = cases_by_item_id[item.id]
        item_context = contextvars.copy_context()

        async def run_in_item_context() -> CaseOutcome:
            return await loop.create_task(run_case(case), context=item_context)

        try:
            outcome = asyncio.run_coroutine_threadsafe(run_in_item_context(), loop).result()
        except Exception as error:
            failures.append(error)
            raise
        outcomes[case.id] = outcome
        return {
            'rule_codes': sorted(outcome.rule_codes),
            'judgement_codes': sorted(outcome.judgement_codes),
            'route': outcome.route,
            'recommendation': outcome.recommendation,
        }

    await asyncio.to_thread(
        run.client.run_experiment,
        name=run.name,
        run_name=run.name,
        data=[run.items_by_case_id[case.id] for case in cases],
        task=task,
        max_concurrency=1,
    )
    if failures:
        raise failures[0]
    return [outcomes[case.id] for case in cases]
