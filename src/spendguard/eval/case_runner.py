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
from spendguard.eval.cases import EvalCase
from spendguard.eval.metrics import CaseOutcome, row_metrics, rule_metrics
from spendguard.eval.report import EvalResults, RowResult
from spendguard.graph.build import ReviewGraph, build_graph
from spendguard.graph.config import GraphConfig
from spendguard.graph.nodes.execute import DecisionEvent
from spendguard.graph.state import ReviewState
from spendguard.policy.loader import Policy
from spendguard.seed.schema import SeedData
from spendguard.settings import EvalRow, Settings
from spendguard.store import precedents, submissions


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


def eval_graph(row: EvalRow, chat_model: BaseChatModel, resources: EvalResources) -> ReviewGraph:
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
            policy_context=row.policy_context,
        )
    )


async def run_case(
    case: EvalCase, graph: ReviewGraph, connection_pool: ConnectionPool, run_identifier: str
) -> CaseOutcome:
    payload = case.submission.model_dump(mode='json')
    idempotency_key = f'eval:{run_identifier}:{case.id}'
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
    thread: RunnableConfig = {
        'configurable': {'thread_id': idempotency_key},
        'callbacks': [token_counter],
    }
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
        latency_seconds=latency_seconds,
        input_tokens=token_counter.input_tokens,
        output_tokens=token_counter.output_tokens,
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
        graph = eval_graph(row, chat_model_for(row), resources)
        outcomes = [
            await run_case(
                case, graph, resources.connection_pool, f'{run_identifier}:row-{row_number}'
            )
            for case in cases
        ]
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
    )
