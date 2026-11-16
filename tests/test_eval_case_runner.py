import datetime
import uuid
from pathlib import Path
from typing import Any

import yaml
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from settings_factory import make_test_settings
from spendguard.db.database import ConnectionPool
from spendguard.eval.case_runner import EvalResources, eval_graph, run_case, run_eval
from spendguard.eval.cases import EvalCase
from spendguard.eval.metrics import Proportion
from spendguard.graph.nodes.plan import RetrievalPlan
from spendguard.graph.nodes.run_judgement_checks import CheckVerdict, JudgementReport, QuotedText
from spendguard.models.factory import build_embedder
from spendguard.models.fake import FakeChatModel, structured_response
from spendguard.policy.loader import POLICY_PATH, load_policy
from spendguard.retrieval.policy_index import index_policy
from spendguard.seed.loader import SEED_DIRECTORY, read_seed_directory
from spendguard.settings import ChatModelSettings, EvalRow
from spendguard.store import precedents, submissions

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
        await index_policy(connection, POLICY, embedder)
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
        EvalRow(model_key='fake_primary', label='Fake primary'),
        chat_model,
        EvalResources(settings, POLICY, SEED_DATA, connection_pool, embedder),
    )

    outcome = await run_case(case, graph, connection_pool, run_identifier=uuid.uuid4().hex)

    assert outcome.case_id == 'dev-judgement-positive-01'
    assert outcome.judgement_codes == {'WEEKEND_SPEND_UNJUSTIFIED'}
    assert outcome.rule_codes == frozenset()
    assert (outcome.route, outcome.recommendation) == ('human_review', 'none')
    assert (outcome.proposed_judgement_results, outcome.dropped_judgement_results) == (2, 1)
    assert '§2.4' in outcome.retrieved_clause_references
    model_calls = len(chat_model.received)
    assert model_calls == 3
    assert (outcome.input_tokens, outcome.output_tokens) == (
        model_calls * INPUT_TOKENS_PER_CALL,
        model_calls * OUTPUT_TOKENS_PER_CALL,
    )
    assert outcome.latency_seconds > 0


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
        payload_hash='leftover',
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
            {'model_key': 'fake_primary', 'label': 'Primary'},
            {'model_key': 'fake_second', 'label': 'Second'},
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
        await index_policy(connection, POLICY, embedder)
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
        EvalResources(settings, POLICY, SEED_DATA, connection_pool, embedder),
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
    assert (
        await precedents.recent_precedents(
            connection_pool, 'EMP-015', 'WEEKEND_SPEND_UNJUSTIFIED', POLICY
        )
        == []
    )
