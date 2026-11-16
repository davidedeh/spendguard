import datetime
import json
import uuid
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langfuse._client.resource_manager import LangfuseResourceManager
from langfuse.api import (
    Dataset,
    DatasetItem,
    LangfuseAPI,
    NotFoundError,
    PaginatedDatasetItems,
    Projects,
)
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import Field, SecretStr

from settings_factory import make_test_settings
from spendguard.db.database import ConnectionPool
from spendguard.eval.case_runner import (
    EvalResources,
    eval_graph,
    run_case,
    run_cases,
    run_eval,
)
from spendguard.eval.cases import EvalCase
from spendguard.eval.metrics import Proportion
from spendguard.graph.nodes.plan import RetrievalPlan
from spendguard.graph.nodes.run_judgement_checks import CheckVerdict, JudgementReport, QuotedText
from spendguard.models.factory import build_embedder
from spendguard.models.fake import FakeChatModel, structured_response
from spendguard.observability.tracing import Tracing, create_tracing
from spendguard.policy.loader import POLICY_PATH, load_policy
from spendguard.retrieval.policy_index import index_policy
from spendguard.seed.loader import SEED_DIRECTORY, read_seed_directory
from spendguard.settings import ChatModelSettings, EvalRow, Settings
from spendguard.store import precedents, submissions

EMBEDDING_MODEL_ID = make_test_settings().embedding_model.model_id

POLICY = load_policy(POLICY_PATH)
SEED_DATA = read_seed_directory(SEED_DIRECTORY)
CASE_DIRECTORY = Path(__file__).parent.parent / 'evals' / 'cases'
INPUT_TOKENS_PER_CALL = 100
OUTPUT_TOKENS_PER_CALL = 7


def dataset_case(case_id: str) -> EvalCase:
    return EvalCase.model_validate(yaml.safe_load((CASE_DIRECTORY / f'{case_id}.yaml').read_text()))


def clause_text(reference: str) -> str:
    return next(clause.text for clause in POLICY.clauses if clause.reference == reference)


class ScriptedEvalModel(FakeChatModel):
    retrieval_queries: list[str] = Field(default_factory=list)
    verdicts: list[CheckVerdict] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        bound_tool_names = [tool['function']['name'] for tool in kwargs.get('tools', [])]
        if bound_tool_names == [RetrievalPlan.__name__]:
            reply = structured_response(
                RetrievalPlan(retrieval_queries=self.retrieval_queries, check_guidance=[])
            )
        elif bound_tool_names == [JudgementReport.__name__]:
            reply = structured_response(JudgementReport(verdicts=self.verdicts))
        else:
            reply = AIMessage(content='Nothing to look up.')
        reply.usage_metadata = {
            'input_tokens': INPUT_TOKENS_PER_CALL,
            'output_tokens': OUTPUT_TOKENS_PER_CALL,
            'total_tokens': INPUT_TOKENS_PER_CALL + OUTPUT_TOKENS_PER_CALL,
        }
        return ChatResult(generations=[ChatGeneration(message=reply)])


async def test_a_case_run_through_the_graph_records_what_the_row_found_and_spent(
    connection_pool: ConnectionPool,
) -> None:
    await precedents.clear_precedents(connection_pool)
    case = dataset_case('dev-judgement-positive-01')
    settings = make_test_settings()
    embedder = build_embedder(settings)
    async with connection_pool.connection() as connection:
        await index_policy(connection, POLICY, embedder, EMBEDDING_MODEL_ID)
    chat_model = ScriptedEvalModel(
        model_id='fake-chat',
        retrieval_queries=[clause_text('§2.4'), 'receipts for stationery'],
        verdicts=[
            CheckVerdict(
                code='WEEKEND_SPEND_UNJUSTIFIED',
                outcome='flagged',
                clause_reference='§2.4',
                quotes=[QuotedText(field='business_purpose', text='needed on Monday morning')],
                cited_precedent_ids=[],
                precedent_id=None,
            ),
            CheckVerdict(
                code='PURPOSE_UNCLEAR',
                outcome='flagged',
                clause_reference='§2.4',
                quotes=[QuotedText(field='business_purpose', text='words nobody wrote')],
                cited_precedent_ids=[],
                precedent_id=None,
            ),
        ],
    )
    graph = eval_graph(
        EvalRow(model_key='fake_primary', label='Fake primary', policy_context='retrieved_clauses'),
        chat_model,
        EvalResources(settings, POLICY, SEED_DATA, connection_pool, embedder, None),
        precedent_memory=True,
    )

    outcome = await run_case(case, graph, connection_pool, uuid.uuid4().hex, tracing=None)

    assert outcome.case_id == 'dev-judgement-positive-01'
    assert outcome.judgement_codes == {'WEEKEND_SPEND_UNJUSTIFIED'}
    assert outcome.rule_codes == frozenset()
    assert (outcome.route, outcome.recommendation) == ('human_review', 'none')
    assert (outcome.proposed_judgement_results, outcome.dropped_judgement_results) == (2, 1)
    assert '§2.4' in outcome.retrieved_clause_references
    assert POLICY.rendered_text not in judgement_prompt(chat_model)
    model_calls = len(chat_model.received)
    assert model_calls == 3
    assert (outcome.input_tokens, outcome.output_tokens) == (
        model_calls * INPUT_TOKENS_PER_CALL,
        model_calls * OUTPUT_TOKENS_PER_CALL,
    )
    assert outcome.latency_seconds > 0


class RefusingEmbedder(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError('the whole-policy row never embeds')

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError('the whole-policy row never retrieves')


def judgement_prompt(chat_model: ScriptedEvalModel) -> str:
    judgement_call = chat_model.received[1]
    return str(judgement_call[0].content)


async def test_the_whole_policy_row_gives_judgement_the_rendered_policy_and_retrieves_nothing(
    connection_pool: ConnectionPool,
) -> None:
    await precedents.clear_precedents(connection_pool)
    settings = make_test_settings()
    chat_model = ScriptedEvalModel(
        model_id='fake-chat',
        retrieval_queries=['weekend stationery', 'receipts for stationery'],
        verdicts=[weekend_verdict()],
    )
    graph = eval_graph(
        EvalRow(model_key='fake_primary', label='Full context', policy_context='whole_policy'),
        chat_model,
        EvalResources(settings, POLICY, SEED_DATA, connection_pool, RefusingEmbedder(), None),
        precedent_memory=True,
    )

    outcome = await run_case(
        dataset_case('dev-judgement-positive-01'),
        graph,
        connection_pool,
        run_identifier=uuid.uuid4().hex,
        tracing=None,
    )

    assert POLICY.rendered_text in judgement_prompt(chat_model)
    assert outcome.judgement_codes == {'WEEKEND_SPEND_UNJUSTIFIED'}
    assert outcome.dropped_judgement_results == 0


def counts(proportion: Proportion) -> tuple[int, int]:
    return proportion.successes, proportion.trials


def weekend_verdict() -> CheckVerdict:
    return CheckVerdict(
        code='WEEKEND_SPEND_UNJUSTIFIED',
        outcome='flagged',
        clause_reference='§2.4',
        quotes=[QuotedText(field='business_purpose', text='needed on Monday morning')],
        cited_precedent_ids=[],
        precedent_id=None,
    )


async def leftover_precedent(connection_pool: ConnectionPool, subject: str) -> None:
    recording = await submissions.record_submission(
        connection_pool,
        idempotency_key=f'leftover:{uuid.uuid4().hex}',
        submission_type='expense_claim',
        payload={},
    )
    weekend_clause = POLICY.finding_codes['WEEKEND_SPEND_UNJUSTIFIED'].clause
    await precedents.record_precedent(
        connection_pool,
        precedents.NewPrecedent(
            subject=subject,
            code='WEEKEND_SPEND_UNJUSTIFIED',
            clause_reference=weekend_clause.reference,
            clause_hash=weekend_clause.hash,
            reason='Left by an earlier run.',
            approver='EMP-002',
            submission_id=recording.submission_id,
        ),
    )


async def test_an_eval_run_reports_each_configured_row_in_order_with_memory_cleared(
    connection_pool: ConnectionPool,
) -> None:
    settings = make_test_settings(
        chat_models={
            'fake_primary': ChatModelSettings(
                provider='fake',
                model_id='fake-chat',
                structured_output_method='function_calling',
                temperature=0.0,
            ),
            'fake_second': ChatModelSettings(
                provider='fake',
                model_id='fake-second',
                structured_output_method='function_calling',
                temperature=None,
            ),
        },
        eval_rows=[
            {
                'model_key': 'fake_primary',
                'label': 'Primary',
                'policy_context': 'retrieved_clauses',
            },
            {'model_key': 'fake_second', 'label': 'Second', 'policy_context': 'retrieved_clauses'},
        ],
        model_prices={
            'fake_primary': {
                'input_per_million_tokens_usd': '1.00',
                'output_per_million_tokens_usd': '5.00',
                'priced_on': '2026-11-02',
            }
        },
    )
    embedder = build_embedder(settings)
    async with connection_pool.connection() as connection:
        await index_policy(connection, POLICY, embedder, EMBEDDING_MODEL_ID)
    await leftover_precedent(connection_pool, 'EMP-015')
    queries = [clause_text('§2.4'), 'receipts for stationery']
    models = {
        'fake_primary': ScriptedEvalModel(
            model_id='fake-chat', retrieval_queries=queries, verdicts=[weekend_verdict()]
        ),
        'fake_second': ScriptedEvalModel(model_id='fake-second', retrieval_queries=queries),
    }

    results = await run_eval(
        (dataset_case('dev-judgement-positive-01'), dataset_case('dev-clean-01')),
        EvalResources(settings, POLICY, SEED_DATA, connection_pool, embedder, None),
        lambda row: models[row.model_key],
        measured_on=datetime.date(2026, 11, 20),
        dataset_hash='a-dataset-hash',
    )

    assert [result.row.label for result in results.rows] == ['Primary', 'Second']
    assert [result.model.model_id for result in results.rows] == ['fake-chat', 'fake-second']
    primary, second = results.rows
    assert counts(primary.metrics.judgement_recall) == (1, 1)
    assert counts(second.metrics.judgement_recall) == (0, 1)
    assert primary.price is not None
    assert second.price is None
    assert (results.case_count, results.dataset_hash) == (2, 'a-dataset-hash')
    remaining = await precedents.recent_precedents(
        connection_pool, 'EMP-015', 'WEEKEND_SPEND_UNJUSTIFIED', POLICY
    )
    assert 'Left by an earlier run.' not in [precedent['reason'] for precedent in remaining]


class PrecedentAwareEvalModel(FakeChatModel):
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        bound_tool_names = [tool['function']['name'] for tool in kwargs.get('tools', [])]
        tool_results = [message for message in messages if isinstance(message, ToolMessage)]
        reply: AIMessage
        if bound_tool_names == [RetrievalPlan.__name__]:
            reply = structured_response(
                RetrievalPlan(
                    retrieval_queries=[clause_text('§2.4'), 'receipts for stationery'],
                    check_guidance=[],
                )
            )
        elif bound_tool_names == [JudgementReport.__name__]:
            supplied_ids = [
                precedent['precedent_id']
                for result in tool_results
                for precedent in json.loads(str(result.content))
            ]
            reply = structured_response(
                JudgementReport(verdicts=[weekend_clearance(supplied_ids[0])])
                if supplied_ids
                else JudgementReport(verdicts=[weekend_flag_quoting('Monday')])
            )
        elif not tool_results:
            reply = AIMessage(
                content='',
                tool_calls=[
                    {
                        'name': 'get_precedents',
                        'args': {'code': 'WEEKEND_SPEND_UNJUSTIFIED'},
                        'id': 'weekend-lookup',
                    }
                ],
            )
        else:
            reply = AIMessage(content='Nothing more to look up.')
        return ChatResult(generations=[ChatGeneration(message=reply)])


def weekend_flag_quoting(text: str) -> CheckVerdict:
    return CheckVerdict(
        code='WEEKEND_SPEND_UNJUSTIFIED',
        outcome='flagged',
        clause_reference='§2.4',
        quotes=[QuotedText(field='business_purpose', text=text)],
        cited_precedent_ids=[],
        precedent_id=None,
    )


def weekend_clearance(precedent_id: str) -> CheckVerdict:
    return CheckVerdict(
        code='WEEKEND_SPEND_UNJUSTIFIED',
        outcome='cleared_by_precedent',
        clause_reference=None,
        quotes=[],
        cited_precedent_ids=[precedent_id],
        precedent_id=precedent_id,
    )


async def indexed_resources(connection_pool: ConnectionPool) -> EvalResources:
    await precedents.clear_precedents(connection_pool)
    settings = make_test_settings()
    embedder = build_embedder(settings)
    async with connection_pool.connection() as connection:
        await index_policy(connection, POLICY, embedder, EMBEDDING_MODEL_ID)
    return EvalResources(settings, POLICY, SEED_DATA, connection_pool, embedder, None)


PAIR_ID = 'dev-judgement-positive-01'
FOLLOW_UP_ID = 'dev-memory-follow-up-01'
PRIMARY_ROW = EvalRow(
    model_key='fake_primary', label='Fake primary', policy_context='retrieved_clauses'
)


async def test_a_follow_up_runs_after_its_pair_and_is_cleared_by_the_precedent_its_dismissal_wrote(
    connection_pool: ConnectionPool,
) -> None:
    resources = await indexed_resources(connection_pool)
    graph = eval_graph(
        PRIMARY_ROW, PrecedentAwareEvalModel(model_id='fake-chat'), resources, precedent_memory=True
    )

    run = await run_cases(
        (dataset_case(FOLLOW_UP_ID), dataset_case(PAIR_ID)),
        graph,
        resources,
        uuid.uuid4().hex,
        None,
    )

    assert [outcome.case_id for outcome in run.outcomes] == [PAIR_ID, FOLLOW_UP_ID]
    pair, follow_up = run.outcomes
    assert pair.judgement_codes == {'WEEKEND_SPEND_UNJUSTIFIED'}
    assert follow_up.judgement_codes == frozenset()
    [pair_precedent_id] = run.precedent_ids_by_pair[PAIR_ID]
    assert follow_up.clearances == {('WEEKEND_SPEND_UNJUSTIFIED', pair_precedent_id)}


async def test_with_memory_off_a_follow_up_sees_no_precedent_although_its_pair_wrote_one(
    connection_pool: ConnectionPool,
) -> None:
    resources = await indexed_resources(connection_pool)
    graph = eval_graph(
        PRIMARY_ROW,
        PrecedentAwareEvalModel(model_id='fake-chat'),
        resources,
        precedent_memory=False,
    )

    run = await run_cases(
        (dataset_case(PAIR_ID), dataset_case(FOLLOW_UP_ID)),
        graph,
        resources,
        uuid.uuid4().hex,
        None,
    )

    assert len(run.precedent_ids_by_pair[PAIR_ID]) == 1
    _, follow_up = run.outcomes
    assert follow_up.judgement_codes == {'WEEKEND_SPEND_UNJUSTIFIED'}
    assert follow_up.clearances == frozenset()


async def test_a_whole_policy_row_sharing_the_primary_model_runs_every_case_afresh(
    connection_pool: ConnectionPool,
) -> None:
    settings = make_test_settings(
        eval_rows=[
            {
                'model_key': 'fake_primary',
                'label': 'Primary',
                'policy_context': 'retrieved_clauses',
            },
            {
                'model_key': 'fake_primary',
                'label': 'Whole policy',
                'policy_context': 'whole_policy',
            },
        ]
    )
    embedder = build_embedder(settings)
    async with connection_pool.connection() as connection:
        await index_policy(connection, POLICY, embedder, EMBEDDING_MODEL_ID)
    queries = ['receipts for stationery', 'travel booking']
    models = {
        'Primary': ScriptedEvalModel(
            model_id='fake-chat',
            retrieval_queries=queries,
            verdicts=[
                CheckVerdict(
                    code='PURPOSE_UNCLEAR',
                    outcome='flagged',
                    clause_reference='§2.4',
                    quotes=[QuotedText(field='business_purpose', text='words nobody wrote')],
                    cited_precedent_ids=[],
                    precedent_id=None,
                )
            ],
        ),
        'Whole policy': ScriptedEvalModel(
            model_id='fake-chat', retrieval_queries=queries, verdicts=[weekend_verdict()]
        ),
    }

    results = await run_eval(
        (dataset_case('dev-judgement-positive-01'),),
        EvalResources(settings, POLICY, SEED_DATA, connection_pool, embedder, None),
        lambda row: models[row.label],
        measured_on=datetime.date(2026, 11, 20),
        dataset_hash='a-dataset-hash',
    )

    primary, whole_policy = results.rows
    assert counts(primary.metrics.judgement_recall) == (0, 1)
    assert counts(whole_policy.metrics.judgement_recall) == (1, 1)
    assert counts(primary.metrics.verify_drop_rate) == (1, 1)
    assert counts(whole_policy.metrics.verify_drop_rate) == (0, 1)
    assert POLICY.rendered_text in judgement_prompt(models['Whole policy'])


async def test_an_eval_run_compares_memory_on_and_off_on_the_primary_model_only(
    connection_pool: ConnectionPool,
) -> None:
    resources = await indexed_resources(connection_pool)
    settings = resources.settings.model_copy(
        update={
            'eval_rows': (
                EvalRow(
                    model_key='fake_second', label='Second', policy_context='retrieved_clauses'
                ),
                PRIMARY_ROW,
            ),
            'chat_models': resources.settings.chat_models
            | {
                'fake_second': ChatModelSettings(
                    provider='fake',
                    model_id='fake-second',
                    structured_output_method='function_calling',
                    temperature=None,
                )
            },
        }
    )
    models: dict[str, FakeChatModel] = {
        'fake_primary': PrecedentAwareEvalModel(model_id='fake-chat'),
        'fake_second': ScriptedEvalModel(
            model_id='fake-second',
            retrieval_queries=[clause_text('§2.4'), 'receipts for stationery'],
        ),
    }

    results = await run_eval(
        (dataset_case(PAIR_ID), dataset_case(FOLLOW_UP_ID)),
        EvalResources(settings, POLICY, SEED_DATA, connection_pool, resources.embedder, None),
        lambda row: models[row.model_key],
        measured_on=datetime.date(2026, 11, 20),
        dataset_hash='a-dataset-hash',
    )

    comparison = results.memory_comparison
    assert comparison is not None
    assert comparison.row == PRIMARY_ROW
    assert counts(comparison.memory_on.false_positive_rate) == (0, 1)
    assert counts(comparison.memory_off.false_positive_rate) == (1, 1)
    assert counts(comparison.memory_on.recommendation_agreement) == (1, 1)
    assert counts(comparison.memory_off.recommendation_agreement) == (0, 1)
    _, primary = results.rows
    assert counts(primary.metrics.recommendation_accuracy) == (2, 2)


class FakeLangfuseProjects:
    def get(self, **keyword_arguments: Any) -> Projects:
        return Projects.model_validate(
            {
                'data': [
                    {
                        'id': 'project-1',
                        'name': 'SpendGuard',
                        'organization': {'id': 'organisation-1', 'name': 'Halden'},
                        'metadata': {},
                    }
                ]
            }
        )


class FakeLangfuseDatasets:
    def __init__(self) -> None:
        self.created: dict[str, Dataset] = {}

    def get(self, dataset_name: str, **keyword_arguments: Any) -> Dataset:
        if dataset_name not in self.created:
            raise NotFoundError(body={'message': 'Dataset not found'})
        return self.created[dataset_name]

    def create(self, *, name: str, **keyword_arguments: Any) -> Dataset:
        moment = datetime.datetime(2026, 11, 20, tzinfo=datetime.UTC).isoformat()
        self.created[name] = Dataset.model_validate(
            {
                'id': f'dataset-{len(self.created) + 1}',
                'name': name,
                'metadata': None,
                'project_id': 'project-1',
                'created_at': moment,
                'updated_at': moment,
            }
        )
        return self.created[name]


class FakeLangfuseDatasetItems:
    def __init__(self, datasets: FakeLangfuseDatasets) -> None:
        self.datasets = datasets
        self.items: dict[str, DatasetItem] = {}

    def create(
        self, *, dataset_name: str, id: str, input: Any, metadata: Any, **keyword_arguments: Any
    ) -> DatasetItem:
        moment = datetime.datetime(2026, 11, 20, tzinfo=datetime.UTC).isoformat()
        self.items[id] = DatasetItem.model_validate(
            {
                'id': id,
                'status': 'ACTIVE',
                'input': input,
                'expected_output': keyword_arguments.get('expected_output'),
                'metadata': metadata,
                'dataset_id': self.datasets.get(dataset_name).id,
                'dataset_name': dataset_name,
                'media_references': [],
                'created_at': moment,
                'updated_at': moment,
            }
        )
        return self.items[id]

    def list(self, *, dataset_name: str, **keyword_arguments: Any) -> PaginatedDatasetItems:
        items = [item for item in self.items.values() if item.dataset_name == dataset_name]
        return PaginatedDatasetItems.model_validate(
            {
                'data': [item.model_dump() for item in items],
                'meta': {'page': 1, 'limit': 50, 'total_items': len(items), 'total_pages': 1},
            }
        )


class FakeLangfuseApi:
    def __init__(self) -> None:
        self.projects = FakeLangfuseProjects()
        self.datasets = FakeLangfuseDatasets()
        self.dataset_items = FakeLangfuseDatasetItems(self.datasets)


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    yield InMemorySpanExporter()
    LangfuseResourceManager.reset()


def attribute(span: ReadableSpan, name: str) -> str:
    return str((span.attributes or {})[name])


def trace_id_of(span: ReadableSpan) -> int:
    assert span.context is not None
    return int(span.context.trace_id)


def tracing_into(
    exporter: InMemorySpanExporter, langfuse_api: FakeLangfuseApi, settings: Settings
) -> Tracing:
    tracing = create_tracing(settings, span_exporter=exporter, tracer_provider=TracerProvider())
    assert tracing is not None
    tracing.client.api = cast(LangfuseAPI, langfuse_api)
    return tracing


class FailingEvalModel(FakeChatModel):
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError('the provider is unavailable')


async def test_a_case_that_fails_inside_a_dataset_run_fails_the_eval(
    connection_pool: ConnectionPool, exporter: InMemorySpanExporter
) -> None:
    resources = await indexed_resources(connection_pool)
    settings = resources.settings.model_copy(
        update={
            'langfuse_public_key': SecretStr('pk-lf-test'),
            'langfuse_secret_key': SecretStr('sk-lf-test'),
            'eval_rows': (PRIMARY_ROW,),
        }
    )
    tracing = tracing_into(exporter, FakeLangfuseApi(), settings)

    with pytest.raises(RuntimeError, match='the provider is unavailable'):
        await run_eval(
            (dataset_case('dev-clean-01'),),
            EvalResources(
                settings, POLICY, SEED_DATA, connection_pool, resources.embedder, tracing
            ),
            lambda row: FailingEvalModel(model_id='fake-chat'),
            measured_on=datetime.date(2026, 11, 20),
            dataset_hash='a-dataset-hash',
        )


async def test_each_row_becomes_a_langfuse_dataset_run_with_one_traced_item_per_case(
    connection_pool: ConnectionPool, exporter: InMemorySpanExporter
) -> None:
    settings = make_test_settings(
        langfuse_public_key='pk-lf-test',
        langfuse_secret_key='sk-lf-test',
        eval_rows=[
            {
                'model_key': 'fake_primary',
                'label': 'Primary',
                'policy_context': 'retrieved_clauses',
            },
            {
                'model_key': 'fake_primary',
                'label': 'Whole policy',
                'policy_context': 'whole_policy',
            },
        ],
        model_prices={
            'fake_primary': {
                'input_per_million_tokens_usd': '1.00',
                'output_per_million_tokens_usd': '5.00',
                'priced_on': '2026-11-02',
            }
        },
    )
    langfuse_api = FakeLangfuseApi()
    tracing = tracing_into(exporter, langfuse_api, settings)
    embedder = build_embedder(settings)
    async with connection_pool.connection() as connection:
        await index_policy(connection, POLICY, embedder, EMBEDDING_MODEL_ID)
    queries = [clause_text('§2.4'), 'receipts for stationery']
    built_models: list[ScriptedEvalModel] = []

    def scripted_model(row: EvalRow) -> ScriptedEvalModel:
        built_models.append(
            ScriptedEvalModel(
                model_id='fake-chat', retrieval_queries=queries, verdicts=[weekend_verdict()]
            )
        )
        return built_models[-1]

    cases = (dataset_case(PAIR_ID), dataset_case('dev-clean-01'))

    results = await run_eval(
        cases,
        EvalResources(settings, POLICY, SEED_DATA, connection_pool, embedder, tracing),
        scripted_model,
        measured_on=datetime.date(2026, 11, 20),
        dataset_hash='a-dataset-hash',
    )

    tracing.client.flush()
    spans = exporter.get_finished_spans()
    item_runs = sorted(
        (span for span in spans if span.name == 'experiment-item-run'),
        key=lambda span: span.start_time or 0,
    )
    case_ids_by_item_id = {
        item.id: item.metadata['case_id'] for item in langfuse_api.dataset_items.items.values()
    }
    runs = [attribute(span, 'langfuse.experiment.id') for span in item_runs]
    assert len(set(runs)) == 2
    assert [
        case_ids_by_item_id[attribute(span, 'langfuse.experiment.item.id')] for span in item_runs
    ] == [PAIR_ID, 'dev-clean-01'] * 2
    assert runs[0] == runs[1] != runs[2] == runs[3]
    item_traces = [trace_id_of(span) for span in item_runs]
    assert len(set(item_traces)) == 4
    intake_traces = {trace_id_of(span) for span in spans if span.name == 'intake'}
    assert set(item_traces) <= intake_traces
    finalize_traces = {trace_id_of(span) for span in spans if span.name == 'finalize'}
    assert {item_traces[0], item_traces[2]} <= finalize_traces
    primary_row_calls = len(built_models[0].received)
    assert results.rows[0].metrics.cost_per_submission_usd == Decimal(
        primary_row_calls * (INPUT_TOKENS_PER_CALL * 1 + OUTPUT_TOKENS_PER_CALL * 5)
    ) / Decimal(1_000_000) / len(cases)
