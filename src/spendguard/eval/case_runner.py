import asyncio
import datetime
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from spendguard.db.database import ConnectionPool
from spendguard.eval.cases import EvalCase, ScriptedReview
from spendguard.eval.dataset_runs import (
    DatasetRun,
    case_failure,
    dataset_name,
    run_as_dataset_run,
    upload_cases,
)
from spendguard.eval.metrics import (
    CaseOutcome,
    CaseResult,
    MemoryMetrics,
    judgement_positive_count,
    memory_metrics,
    row_metrics,
    rule_metrics,
)
from spendguard.eval.report import (
    EvalResults,
    MemoryComparison,
    RowResult,
    RowsDirectory,
    write_row_results,
)
from spendguard.eval.validator import precedent_subject
from spendguard.graph.build import ReviewGraph, build_graph
from spendguard.graph.config import GraphConfig, thread_config
from spendguard.graph.nodes.execute import DecisionEvent
from spendguard.graph.state import (
    Decision,
    Disposition,
    DispositionVerdict,
    RecordedFinding,
    initial_review_state,
    precedent_clearances,
)
from spendguard.observability.tracing import Tracing, traced_advance
from spendguard.policy.loader import FindingCode, Policy
from spendguard.queue.messages import QueueMessage
from spendguard.seed.schema import SeedData
from spendguard.settings import EvalRow, Settings
from spendguard.store import precedents, submissions
from spendguard.worker import WorkerRuntime, process_message


class _DiscardedDecisionEvents:
    async def send(self, event: DecisionEvent) -> None:
        return None


class _TokenCounter(BaseCallbackHandler):
    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                if not isinstance(generation, ChatGeneration):
                    continue
                message = generation.message
                if isinstance(message, AIMessage) and message.usage_metadata is not None:
                    self.input_tokens += message.usage_metadata['input_tokens']
                    self.output_tokens += message.usage_metadata['output_tokens']


@dataclass(frozen=True)
class EvalResources:
    settings: Settings
    policy: Policy
    seed_data: SeedData
    connection_pool: ConnectionPool
    embedder: Embeddings
    tracing: Tracing | None


def eval_graph(
    row: EvalRow, chat_model: BaseChatModel, resources: EvalResources, *, precedent_memory: bool
) -> ReviewGraph:
    return build_graph(
        GraphConfig(
            settings=resources.settings.model_copy(update={'primary_model_key': row.model_key}),
            policy=resources.policy,
            seed_data=resources.seed_data,
            chat_model=chat_model,
            embedder=resources.embedder,
            connection_pool=resources.connection_pool,
            checkpointer=AsyncPostgresSaver(resources.connection_pool),
            decision_event_sender=_DiscardedDecisionEvents(),
            precedent_memory=precedent_memory,
            record_precedents=False,
            policy_context=row.policy_context,
        )
    )


def eval_idempotency_key(case: EvalCase, run_identifier: str) -> str:
    return f'eval:{run_identifier}:{case.id}'


@dataclass(frozen=True)
class PlannedPrecedent:
    pair: EvalCase
    subject: str
    code: FindingCode
    clause_reference: str
    clause_hash: str
    reason: str
    approver: str


def planned_precedents(cases: Sequence[EvalCase], policy: Policy) -> dict[str, PlannedPrecedent]:
    pairs = {case.id: case for case in cases if case.scripted_review is not None}
    planned: dict[str, PlannedPrecedent] = {}
    for follow_up in cases:
        if follow_up.memory is not None:
            planned[follow_up.id] = _planned_precedent(
                pairs[follow_up.memory.precedent_from], follow_up, policy
            )
    return planned


def _planned_precedent(pair: EvalCase, follow_up: EvalCase, policy: Policy) -> PlannedPrecedent:
    memory = follow_up.memory
    review = pair.scripted_review
    if memory is None or review is None:
        raise ValueError(f'{follow_up.id} does not follow up a memory pair')
    reason = next(
        disposition.reason
        for disposition in review.dispositions
        if disposition.code == memory.code and disposition.disposition == 'dismissed'
    )
    clause_reference = next(
        finding.gold_clauses[0] for finding in pair.expected_findings if finding.code == memory.code
    )
    clause = next(
        (clause for clause in policy.clauses if clause.reference == clause_reference), None
    )
    if clause is None:
        raise ValueError(
            f'{pair.id} gives {clause_reference} for {memory.code}, and {clause_reference} is not'
            ' in the current policy'
        )
    scope = policy.finding_codes[memory.code].precedent_scope
    subject = None if scope is None else precedent_subject(follow_up.submission, scope)
    if reason is None or subject is None:
        raise ValueError(f'{pair.id} gives no precedent for {memory.code}')
    return PlannedPrecedent(
        pair=pair,
        subject=subject,
        code=memory.code,
        clause_reference=clause.reference,
        clause_hash=clause.hash,
        reason=reason,
        approver=review.approver,
    )


@dataclass(frozen=True)
class SeededCases:
    cases: Sequence[EvalCase]
    planned: Mapping[str, PlannedPrecedent]

    def follow_ups_only(self) -> 'SeededCases':
        return SeededCases(
            cases=[case for case in self.cases if case.memory is not None], planned=self.planned
        )


def seeded_cases(cases: Sequence[EvalCase], policy: Policy) -> SeededCases:
    return SeededCases(cases=cases, planned=planned_precedents(cases, policy))


async def record_planned_precedents(
    planned: Mapping[str, PlannedPrecedent], connection_pool: ConnectionPool, run_identifier: str
) -> dict[str, str]:
    return {
        follow_up_id: await _record_planned_precedent(plan, connection_pool, run_identifier)
        for follow_up_id, plan in planned.items()
    }


async def _record_planned_precedent(
    plan: PlannedPrecedent, connection_pool: ConnectionPool, run_identifier: str
) -> str:
    pair = plan.pair
    recording = await submissions.record_submission(
        connection_pool,
        idempotency_key=eval_idempotency_key(pair, run_identifier),
        submission_type=pair.submission.type,
        payload=pair.submission.model_dump(mode='json'),
    )
    await precedents.record_precedent(
        connection_pool,
        precedents.NewPrecedent(
            subject=plan.subject,
            code=plan.code,
            clause_reference=plan.clause_reference,
            clause_hash=plan.clause_hash,
            reason=plan.reason,
            approver=plan.approver,
            submission_id=recording.submission_id,
        ),
    )
    return next(
        recorded.precedent_id
        for recorded in await precedents.precedents_recorded_from(
            connection_pool, recording.submission_id
        )
        if recorded.code == plan.code
    )


async def run_case(
    case: EvalCase,
    graph: ReviewGraph,
    connection_pool: ConnectionPool,
    run_identifier: str,
    tracing: Tracing | None,
) -> CaseOutcome:
    payload = case.submission.model_dump(mode='json')
    idempotency_key = eval_idempotency_key(case, run_identifier)
    recording = await submissions.record_submission(
        connection_pool,
        idempotency_key=idempotency_key,
        submission_type=case.submission.type,
        payload=payload,
    )
    initial_state = initial_review_state(
        recording.submission_id, idempotency_key, case.submission.type, payload
    )
    token_counter = _TokenCounter()
    thread: RunnableConfig = thread_config(idempotency_key) | {'callbacks': [token_counter]}
    started = time.perf_counter()
    await traced_advance(tracing, graph, initial_state, thread, str(recording.submission_id))
    latency_seconds = time.perf_counter() - started
    state = (await graph.aget_state(thread)).values
    findings = state.get('findings', [])
    dropped = sum(entry['event'] == 'finding_dropped' for entry in state['audit'])
    kept_clearances = precedent_clearances(state['judgement_outcomes'])
    judgement_findings = [finding for finding in findings if finding['source'] == 'judgement']
    return CaseOutcome(
        case_id=case.id,
        rule_codes=frozenset(
            finding['code'] for finding in findings if finding['source'] == 'rule'
        ),
        judgement_codes=frozenset(finding['code'] for finding in judgement_findings),
        route=state['route'],
        recommendation=state['recommendation'],
        proposed_judgement_results=len(judgement_findings) + len(kept_clearances) + dropped,
        dropped_judgement_results=dropped,
        retrieved_clause_references=frozenset(
            clause['clause_reference'] for clause in state['retrieved_clauses']
        ),
        clearances=frozenset(
            (clearance['code'], clearance['precedent_id']) for clearance in kept_clearances
        ),
        unanswered_judgement_codes=frozenset(state['unanswered_checks']),
        latency_seconds=latency_seconds,
        input_tokens=token_counter.input_tokens,
        output_tokens=token_counter.output_tokens,
    )


UPHELD_WITHOUT_A_SCRIPTED_REASON = 'Upheld by the scripted review.'


def _scripted_dispositions(
    review: ScriptedReview, findings: list[RecordedFinding]
) -> list[Disposition]:
    scripted_by_code = {disposition.code: disposition for disposition in review.dispositions}
    dispositions: list[Disposition] = []
    for finding in findings:
        scripted = scripted_by_code.get(finding['code'])
        if scripted is None or scripted.disposition == 'upheld':
            verdict: DispositionVerdict = 'upheld'
            reason = (
                UPHELD_WITHOUT_A_SCRIPTED_REASON
                if scripted is None or scripted.reason is None
                else scripted.reason
            )
        elif scripted.reason is not None:
            verdict = 'dismissed'
            reason = scripted.reason
        else:
            raise ValueError(f'dismissing {scripted.code} needs a reason')
        dispositions.append(
            {'finding_id': finding['finding_id'], 'verdict': verdict, 'reason': reason}
        )
    return dispositions


async def _apply_scripted_review(
    case: EvalCase,
    review: ScriptedReview,
    graph: ReviewGraph,
    resources: EvalResources,
    run_identifier: str,
) -> None:
    connection_pool = resources.connection_pool
    idempotency_key = eval_idempotency_key(case, run_identifier)
    snapshot = await graph.aget_state(thread_config(idempotency_key))
    if not snapshot.interrupts:
        return
    submission_id = uuid.UUID(snapshot.values['submission_id'])
    dispositions = _scripted_dispositions(review, snapshot.values['findings'])
    upheld_finding_ids = [
        disposition['finding_id']
        for disposition in dispositions
        if disposition['verdict'] == 'upheld'
    ]
    decision: Decision = {
        'outcome': 'reject' if upheld_finding_ids else 'approve',
        'decided_by': review.approver,
        'reason': f'Upheld {", ".join(upheld_finding_ids)}' if upheld_finding_ids else None,
        'dispositions': dispositions,
    }
    if not await submissions.record_decision(
        connection_pool, submission_id, decision, review.approver
    ):
        raise RuntimeError(f'{case.id} was not awaiting a decision')
    await process_message(
        QueueMessage(type='resume', submission_id=submission_id, thread_id=idempotency_key),
        WorkerRuntime(graph=graph, connection_pool=connection_pool, tracing=resources.tracing),
    )


@dataclass(frozen=True)
class PairedRun:
    results: list[CaseResult]
    precedent_id_by_follow_up: dict[str, str]


async def run_cases(
    seeded: SeededCases,
    graph: ReviewGraph,
    resources: EvalResources,
    run_identifier: str,
    dataset_run: DatasetRun | None,
) -> PairedRun:
    follow_ups = [case for case in seeded.cases if case.memory is not None]
    others = [case for case in seeded.cases if case.memory is None]
    precedent_id_by_follow_up: dict[str, str] = {}

    async def run_paired_case(case: EvalCase) -> CaseOutcome:
        if case.memory is not None and not precedent_id_by_follow_up:
            precedent_id_by_follow_up.update(
                await record_planned_precedents(
                    seeded.planned, resources.connection_pool, run_identifier
                )
            )
        outcome = await run_case(
            case, graph, resources.connection_pool, run_identifier, resources.tracing
        )
        if case.scripted_review is not None:
            await _apply_scripted_review(
                case, case.scripted_review, graph, resources, run_identifier
            )
        return outcome

    async def run_or_record_failure(case: EvalCase) -> CaseResult:
        try:
            return await run_paired_case(case)
        except Exception as error:
            return case_failure(case, error)

    if dataset_run is None:
        results = [await run_or_record_failure(case) for case in others + follow_ups]
    else:
        results = await run_as_dataset_run(dataset_run, others + follow_ups, run_paired_case)
    return PairedRun(
        results=results,
        precedent_id_by_follow_up=precedent_id_by_follow_up,
    )


def _memory_comparison_row(settings: Settings) -> EvalRow | None:
    return next(
        (
            row
            for row in settings.eval_rows
            if row.model_key == settings.primary_model_key
            and row.policy_context == 'retrieved_clauses'
        ),
        None,
    )


async def _compare_memory(
    seeded: SeededCases,
    resources: EvalResources,
    row: EvalRow,
    chat_model: BaseChatModel,
    run_identifier: str,
) -> MemoryComparison:
    follow_ups = seeded.follow_ups_only()
    metrics_by_setting: dict[bool, MemoryMetrics] = {}
    for precedent_memory in (True, False):
        await precedents.clear_precedents(resources.connection_pool)
        run = await run_cases(
            follow_ups,
            eval_graph(row, chat_model, resources, precedent_memory=precedent_memory),
            resources,
            f'{run_identifier}:memory-{"on" if precedent_memory else "off"}',
            dataset_run=None,
        )
        metrics_by_setting[precedent_memory] = memory_metrics(
            follow_ups.cases, run.results, run.precedent_id_by_follow_up
        )
    return MemoryComparison(
        row=row,
        model=resources.settings.chat_models[row.model_key],
        memory_on=metrics_by_setting[True],
        memory_off=metrics_by_setting[False],
    )


def _first_completed_results(results_by_row: Sequence[Sequence[CaseResult]]) -> list[CaseResult]:
    first_results: dict[str, CaseResult] = {}
    for results in results_by_row:
        for result in results:
            earlier = first_results.get(result.case_id)
            if earlier is None or (
                isinstance(result, CaseOutcome) and not isinstance(earlier, CaseOutcome)
            ):
                first_results[result.case_id] = result
    return list(first_results.values())


def pre_flight_cases(cases: Sequence[EvalCase], minimum_case_count: int) -> list[EvalCase]:
    cases_by_slice: dict[str, list[EvalCase]] = {}
    for case in cases:
        cases_by_slice.setdefault(case.slice, []).append(case)
    chosen_case_ids: set[str] = set()
    for round_index in range(max(len(slice_cases) for slice_cases in cases_by_slice.values())):
        for slice_cases in cases_by_slice.values():
            if round_index >= len(slice_cases):
                continue
            case = slice_cases[round_index]
            chosen_case_ids.add(case.id)
            if case.memory is not None:
                chosen_case_ids.add(case.memory.precedent_from)
        if len(chosen_case_ids) >= minimum_case_count:
            break
    return [case for case in cases if case.id in chosen_case_ids]


@dataclass(frozen=True)
class EvalRunOptions:
    measured_on: datetime.date
    dataset_hash: str
    run_identifier: str
    rows_directory: Path
    cases_per_row: int | None


async def run_eval(
    cases: Sequence[EvalCase],
    resources: EvalResources,
    chat_model_for: Callable[[EvalRow], BaseChatModel],
    options: EvalRunOptions,
) -> EvalResults:
    settings = resources.settings
    measured_on = options.measured_on
    row_cases = (
        list(cases)
        if options.cases_per_row is None
        else pre_flight_cases(cases, options.cases_per_row)
    )
    chat_models = [chat_model_for(row) for row in settings.eval_rows]
    run_identifier = options.run_identifier
    rows_directory = RowsDirectory(options.rows_directory, run_identifier)
    row_results: list[RowResult] = []
    results_by_row: list[list[CaseResult]] = []
    tracing = resources.tracing
    items_by_case_id = (
        None
        if tracing is None
        else await asyncio.to_thread(
            upload_cases, tracing.client, dataset_name(options.dataset_hash), row_cases
        )
    )
    run_name_prefix = (
        f'{measured_on}' if options.cases_per_row is None else f'{measured_on} pre-flight'
    )
    seeded = seeded_cases(row_cases, resources.policy)
    for row_number, (row, chat_model) in enumerate(
        zip(settings.eval_rows, chat_models, strict=True), start=1
    ):
        await precedents.clear_precedents(resources.connection_pool)
        graph = eval_graph(row, chat_model, resources, precedent_memory=True)
        row_run_identifier = f'{run_identifier}:row-{row_number}'
        dataset_run = (
            None
            if tracing is None or items_by_case_id is None
            else DatasetRun(
                client=tracing.client,
                name=f'{run_name_prefix} {row.label} {row_run_identifier}',
                items_by_case_id=items_by_case_id,
            )
        )
        results = (
            await run_cases(seeded, graph, resources, row_run_identifier, dataset_run)
        ).results
        model = settings.chat_models[row.model_key]
        write_row_results(rows_directory, row_number, row, model, results)
        results_by_row.append(results)
        price = settings.model_prices[row.model_key]
        row_results.append(
            RowResult(
                row=row,
                model=model,
                price=price,
                metrics=row_metrics(row_cases, results, resources.policy, price),
            )
        )
    memory_row = _memory_comparison_row(settings)
    return EvalResults(
        measured_on=measured_on,
        dataset_hash=options.dataset_hash,
        run_identifier=run_identifier,
        case_count=len(row_cases),
        judgement_positive_count=judgement_positive_count(row_cases, resources.policy),
        cases_per_row=options.cases_per_row,
        retrieval_top_k=settings.retrieval_top_k,
        judgement_max_tool_calls=settings.judgement_max_tool_calls,
        rows=tuple(row_results),
        rule_metrics=rule_metrics(
            row_cases, _first_completed_results(results_by_row), resources.policy
        ),
        memory_comparison=(
            None
            if memory_row is None
            else await _compare_memory(
                seeded,
                resources,
                memory_row,
                chat_models[settings.eval_rows.index(memory_row)],
                run_identifier,
            )
        ),
    )
