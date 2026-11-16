import datetime
import hashlib
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
from spendguard.eval.metrics import (
    CaseOutcome,
    MemoryMetrics,
    memory_metrics,
    row_metrics,
    rule_metrics,
)
from spendguard.eval.report import EvalResults, MemoryComparison, RowResult
from spendguard.graph.build import ReviewGraph, build_graph
from spendguard.graph.config import GraphConfig
from spendguard.graph.nodes.execute import DecisionEvent
from spendguard.graph.state import Decision, Disposition, RecordedFinding, ReviewState
from spendguard.policy.loader import Policy
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
            policy_context=row.policy_context,
        )
    )


def _thread(idempotency_key: str) -> RunnableConfig:
    return {'configurable': {'thread_id': idempotency_key}}


def _idempotency_key(case: EvalCase, run_identifier: str) -> str:
    return f'eval:{run_identifier}:{case.id}'


async def run_case(
    case: EvalCase, graph: ReviewGraph, connection_pool: ConnectionPool, run_identifier: str
) -> CaseOutcome:
    payload = case.submission.model_dump(mode='json')
    idempotency_key = _idempotency_key(case, run_identifier)
    recording = await submissions.record_submission(
        connection_pool,
        idempotency_key=idempotency_key,
        submission_type=case.submission.type,
        payload=payload,
        payload_hash=hashlib.sha256(idempotency_key.encode()).hexdigest(),
    )
    initial_state: ReviewState = {
        'submission_id': str(recording.submission_id),
        'idempotency_key': idempotency_key,
        'submission_type': case.submission.type,
        'submission': payload,
    }
    token_counter = _TokenCounter()
    thread: RunnableConfig = _thread(idempotency_key) | {'callbacks': [token_counter]}
    started = time.perf_counter()
    await graph.ainvoke(initial_state, thread)
    latency_seconds = time.perf_counter() - started
    state = (await graph.aget_state(thread)).values
    findings = state.get('findings', [])
    dropped = sum(entry['event'] == 'finding_dropped' for entry in state['audit'])
    kept_clearances = sum(
        outcome['outcome'] == 'cleared_by_precedent' for outcome in state['judgement_outcomes']
    )
    judgement_findings = [finding for finding in findings if finding['source'] == 'judgement']
    return CaseOutcome(
        case_id=case.id,
        rule_codes=frozenset(
            finding['code'] for finding in findings if finding['source'] == 'rule'
        ),
        judgement_codes=frozenset(finding['code'] for finding in judgement_findings),
        route=state['route'],
        recommendation=state['recommendation'],
        proposed_judgement_results=len(judgement_findings) + kept_clearances + dropped,
        dropped_judgement_results=dropped,
        retrieved_clause_references=frozenset(
            clause['clause_reference'] for clause in state['retrieved_clauses']
        ),
        clearances=frozenset(
            (outcome['code'], outcome['precedent_id'])
            for outcome in state['judgement_outcomes']
            if outcome['outcome'] == 'cleared_by_precedent' and outcome['precedent_id'] is not None
        ),
        latency_seconds=latency_seconds,
        input_tokens=token_counter.input_tokens,
        output_tokens=token_counter.output_tokens,
    )


def _scripted_dispositions(
    review: ScriptedReview, findings: list[RecordedFinding]
) -> list[Disposition]:
    scripted_by_code = {disposition.code: disposition for disposition in review.dispositions}
    dispositions: list[Disposition] = []
    for finding in findings:
        scripted = scripted_by_code.get(finding['code'])
        dispositions.append(
            {
                'finding_id': finding['finding_id'],
                'verdict': 'upheld' if scripted is None else scripted.disposition,
                'reason': None if scripted is None else scripted.reason,
            }
        )
    return dispositions


async def _apply_scripted_review(
    case: EvalCase,
    review: ScriptedReview,
    graph: ReviewGraph,
    connection_pool: ConnectionPool,
    run_identifier: str,
) -> frozenset[str]:
    idempotency_key = _idempotency_key(case, run_identifier)
    snapshot = await graph.aget_state(_thread(idempotency_key))
    if not snapshot.interrupts:
        return frozenset()
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
        WorkerRuntime(graph=graph, connection_pool=connection_pool, tracing=None),
    )
    return await precedents.precedent_ids_recorded_from(connection_pool, submission_id)


@dataclass(frozen=True)
class PairedRun:
    outcomes: list[CaseOutcome]
    precedent_ids_by_pair: dict[str, frozenset[str]]


async def run_cases(
    cases: Sequence[EvalCase], graph: ReviewGraph, resources: EvalResources, run_identifier: str
) -> PairedRun:
    follow_ups = [case for case in cases if case.expected_clearance is not None]
    others = [case for case in cases if case.expected_clearance is None]
    outcomes: list[CaseOutcome] = []
    precedent_ids_by_pair: dict[str, frozenset[str]] = {}
    for case in others + follow_ups:
        outcomes.append(await run_case(case, graph, resources.connection_pool, run_identifier))
        if case.scripted_review is not None:
            precedent_ids_by_pair[case.id] = await _apply_scripted_review(
                case, case.scripted_review, graph, resources.connection_pool, run_identifier
            )
    return PairedRun(outcomes=outcomes, precedent_ids_by_pair=precedent_ids_by_pair)


async def _compare_memory(
    cases: Sequence[EvalCase],
    resources: EvalResources,
    chat_model_for: Callable[[EvalRow], BaseChatModel],
    run_identifier: str,
) -> MemoryComparison | None:
    settings = resources.settings
    row = next(
        (
            row
            for row in settings.eval_rows
            if row.model_key == settings.primary_model_key
            and row.policy_context == 'retrieved_clauses'
        ),
        None,
    )
    if row is None:
        return None
    paired_cases = [
        case
        for case in cases
        if case.scripted_review is not None or case.expected_clearance is not None
    ]
    chat_model = chat_model_for(row)
    metrics_by_setting: dict[bool, MemoryMetrics] = {}
    for precedent_memory in (True, False):
        await precedents.clear_precedents(resources.connection_pool)
        run = await run_cases(
            paired_cases,
            eval_graph(row, chat_model, resources, precedent_memory=precedent_memory),
            resources,
            f'{run_identifier}:memory-{"on" if precedent_memory else "off"}',
        )
        metrics_by_setting[precedent_memory] = memory_metrics(
            paired_cases, run.outcomes, run.precedent_ids_by_pair
        )
    return MemoryComparison(
        row=row,
        model=settings.chat_models[row.model_key],
        memory_on=metrics_by_setting[True],
        memory_off=metrics_by_setting[False],
    )


async def run_eval(
    cases: Sequence[EvalCase],
    resources: EvalResources,
    chat_model_for: Callable[[EvalRow], BaseChatModel],
    measured_on: datetime.date,
    dataset_hash: str,
) -> EvalResults:
    settings = resources.settings
    run_identifier = uuid.uuid4().hex
    row_results: list[RowResult] = []
    first_row_outcomes: list[CaseOutcome] = []
    for row_number, row in enumerate(settings.eval_rows, start=1):
        await precedents.clear_precedents(resources.connection_pool)
        graph = eval_graph(row, chat_model_for(row), resources, precedent_memory=True)
        outcomes = (
            await run_cases(cases, graph, resources, f'{run_identifier}:row-{row_number}')
        ).outcomes
        first_row_outcomes = first_row_outcomes or outcomes
        price = settings.model_prices.get(row.model_key)
        row_results.append(
            RowResult(
                row=row,
                model=settings.chat_models[row.model_key],
                price=price,
                metrics=row_metrics(cases, outcomes, resources.policy, price),
            )
        )
    return EvalResults(
        measured_on=measured_on,
        dataset_hash=dataset_hash,
        case_count=len(cases),
        retrieval_top_k=settings.retrieval_top_k,
        judgement_max_tool_calls=settings.judgement_max_tool_calls,
        rows=tuple(row_results),
        rule_metrics=rule_metrics(cases, first_row_outcomes, resources.policy),
        memory_comparison=await _compare_memory(cases, resources, chat_model_for, run_identifier),
    )
