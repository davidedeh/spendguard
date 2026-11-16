import io
import json
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse._client.resource_manager import LangfuseResourceManager
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from settings_factory import make_test_settings
from spendguard.graph.build import ReviewGraph
from spendguard.graph.state import ReviewState
from spendguard.models.fake import FakeChatModel
from spendguard.observability.tracing import (
    Tracing,
    create_tracing,
    flush_traces_before_return,
    mask_personal_data,
    traced_advance,
)

SUBMISSION_ID = '5f0c8a52-6a3e-4d55-9a8f-2f3c1d9e7b10'


def intake(state: ReviewState) -> ReviewState:
    return {
        'audit': [{'node': 'intake', 'event': 'field_redacted', 'detail': {'field': 'purpose'}}]
    }


def plan(state: ReviewState) -> ReviewState:
    chat_model = FakeChatModel(
        model_id='fake-chat', responses=[AIMessage('Ask ana.ruiz@halden.example for a receipt')]
    )
    chat_model.invoke([HumanMessage(str(state.get('submission', {})))])
    return {}


def human_review(state: ReviewState) -> ReviewState:
    interrupt({'submission_id': state['submission_id']})
    return {}


def finalize(state: ReviewState) -> ReviewState:
    return {}


def pausing_graph() -> ReviewGraph:
    builder: StateGraph[ReviewState, None, ReviewState, ReviewState] = StateGraph(ReviewState)
    builder.add_node('intake', intake)
    builder.add_node('plan', plan)
    builder.add_node('human_review', human_review)
    builder.add_node('finalize', finalize)
    builder.add_edge(START, 'intake')
    builder.add_edge('intake', 'plan')
    builder.add_edge('plan', 'human_review')
    builder.add_edge('human_review', 'finalize')
    builder.add_edge('finalize', END)
    return builder.compile(checkpointer=InMemorySaver())


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    yield InMemorySpanExporter()
    LangfuseResourceManager.reset()


@pytest.fixture
def tracing(exporter: InMemorySpanExporter) -> Tracing:
    settings = make_test_settings(
        langfuse_public_key='pk-lf-test', langfuse_secret_key='sk-lf-test'
    )
    tracing = create_tracing(settings, span_exporter=exporter, tracer_provider=TracerProvider())
    assert tracing is not None
    return tracing


def exported_spans(tracing: Tracing, exporter: InMemorySpanExporter) -> tuple[ReadableSpan, ...]:
    tracing.client.flush()
    return exporter.get_finished_spans()


async def test_a_start_and_a_resume_are_two_traces_in_the_submission_session(
    tracing: Tracing, exporter: InMemorySpanExporter
) -> None:
    graph = pausing_graph()
    thread: RunnableConfig = {'configurable': {'thread_id': 'expense-claims:C-1'}}

    await traced_advance(
        tracing, graph, {'submission_id': SUBMISSION_ID, 'audit': []}, thread, SUBMISSION_ID
    )
    await traced_advance(tracing, graph, Command(resume='decided'), thread, SUBMISSION_ID)

    spans = exported_spans(tracing, exporter)
    trace_ids = {span.context.trace_id for span in spans if span.context is not None}
    session_ids = {(span.attributes or {}).get('session.id') for span in spans}
    assert len(trace_ids) == 2
    assert session_ids == {SUBMISSION_ID}


async def test_traced_payloads_carry_no_card_numbers_emails_or_phone_numbers(
    tracing: Tracing, exporter: InMemorySpanExporter
) -> None:
    graph = pausing_graph()
    thread: RunnableConfig = {'configurable': {'thread_id': 'expense-claims:C-2'}}
    submission = {
        'purpose': 'Paid with 4111 1111 1111 1111, receipt to ana.ruiz@halden.example',
        'notes': 'Call me on +44 20 7946 0958',
    }

    await traced_advance(
        tracing,
        graph,
        {'submission_id': SUBMISSION_ID, 'submission': submission, 'audit': []},
        thread,
        SUBMISSION_ID,
    )

    exported_text = repr([span.attributes for span in exported_spans(tracing, exporter)])
    assert '[CARD]' in exported_text
    for personal_data in ('4111 1111 1111 1111', 'ana.ruiz@halden.example', '7946 0958'):
        assert personal_data not in exported_text


async def test_a_handler_exports_its_traces_before_it_returns(
    tracing: Tracing, exporter: InMemorySpanExporter
) -> None:
    @flush_traces_before_return
    async def handler(event: dict[str, object]) -> dict[str, object]:
        tracing.client.start_observation(name='advance').end()
        return {'batchItemFailures': []}

    await handler({})

    assert [span.name for span in exporter.get_finished_spans()] == ['advance']


async def test_a_failing_handler_still_exports_its_traces(
    tracing: Tracing, exporter: InMemorySpanExporter
) -> None:
    @flush_traces_before_return
    async def handler(event: dict[str, object]) -> dict[str, object]:
        tracing.client.start_observation(name='advance').end()
        raise RuntimeError('advance failed')

    with pytest.raises(RuntimeError):
        await handler({})

    assert [span.name for span in exporter.get_finished_spans()] == ['advance']


async def test_without_configured_keys_nothing_is_traced_even_when_the_environment_has_keys(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('LANGFUSE_PUBLIC_KEY', 'pk-lf-from-the-shell')
    monkeypatch.setenv('LANGFUSE_SECRET_KEY', 'sk-lf-from-the-shell')
    tracing = create_tracing(
        make_test_settings(), span_exporter=exporter, tracer_provider=TracerProvider()
    )
    thread: RunnableConfig = {'configurable': {'thread_id': 'expense-claims:C-5'}}

    await traced_advance(
        tracing,
        pausing_graph(),
        {'submission_id': SUBMISSION_ID, 'audit': []},
        thread,
        SUBMISSION_ID,
    )

    assert LangfuseResourceManager._instances == {}


class ToolResult(BaseModel):
    contact: str
    grades: tuple[str, ...]


def test_masking_redacts_text_inside_objects_and_tuples() -> None:
    masked = mask_personal_data(
        data={'result': ToolResult(contact='ana.ruiz@halden.example', grades=('+44 20 7946 0958',))}
    )

    assert masked == {'result': {'contact': '[EMAIL]', 'grades': ['[PHONE]']}}


async def test_guardrail_events_are_recorded_once_with_metadata_in_the_advance_that_raised_them(
    tracing: Tracing, exporter: InMemorySpanExporter
) -> None:
    graph = pausing_graph()
    thread: RunnableConfig = {'configurable': {'thread_id': 'expense-claims:C-3'}}

    await traced_advance(
        tracing, graph, {'submission_id': SUBMISSION_ID, 'audit': []}, thread, SUBMISSION_ID
    )
    await traced_advance(tracing, graph, Command(resume='decided'), thread, SUBMISSION_ID)

    spans = exported_spans(tracing, exporter)
    [intake_span] = [span for span in spans if span.name == 'intake']
    [guardrail_span] = [span for span in spans if span.name == 'field_redacted']
    attributes = guardrail_span.attributes or {}
    assert guardrail_span.context is not None
    assert intake_span.context is not None
    assert guardrail_span.context.trace_id == intake_span.context.trace_id
    assert attributes['langfuse.observation.metadata.node'] == '"intake"'
    assert attributes['langfuse.observation.metadata.field'] == '"purpose"'


async def test_log_lines_from_an_advance_carry_the_submission_node_and_trace_ids_only(
    tracing: Tracing, exporter: InMemorySpanExporter, log_stream: io.StringIO
) -> None:
    graph = pausing_graph()
    thread: RunnableConfig = {'configurable': {'thread_id': 'expense-claims:C-4'}}
    submission = {'purpose': 'Receipt to ana.ruiz@halden.example'}

    await traced_advance(
        tracing,
        graph,
        {'submission_id': SUBMISSION_ID, 'submission': submission, 'audit': []},
        thread,
        SUBMISSION_ID,
    )

    [intake_span] = [span for span in exported_spans(tracing, exporter) if span.name == 'intake']
    assert intake_span.context is not None
    lines = [json.loads(line) for line in log_stream.getvalue().splitlines()]
    [guardrail_line] = [line for line in lines if line['event'] == 'field_redacted']
    assert {key: value for key, value in guardrail_line.items() if key != 'timestamp'} == {
        'level': 'info',
        'event': 'field_redacted',
        'submission_id': SUBMISSION_ID,
        'node': 'intake',
        'trace_id': format(intake_span.context.trace_id, '032x'),
    }
    assert 'ana.ruiz' not in log_stream.getvalue()


class ChatModelCallCounter(BaseCallbackHandler):
    def __init__(self) -> None:
        self.calls = 0

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[BaseMessage]], **kwargs: Any
    ) -> None:
        self.calls += 1


async def test_a_traced_advance_still_reports_to_the_callbacks_the_caller_supplied(
    tracing: Tracing, exporter: InMemorySpanExporter
) -> None:
    counter = ChatModelCallCounter()
    thread: RunnableConfig = {
        'configurable': {'thread_id': 'expense-claims:C-6'},
        'callbacks': [counter],
    }

    await traced_advance(
        tracing,
        pausing_graph(),
        {'submission_id': SUBMISSION_ID, 'audit': []},
        thread,
        SUBMISSION_ID,
    )

    assert counter.calls == 1
    assert any(span.name == 'plan' for span in exported_spans(tracing, exporter))
