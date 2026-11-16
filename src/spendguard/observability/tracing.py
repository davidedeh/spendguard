import functools
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse, get_client, propagate_attributes
from langfuse._utils.serializer import EventSerializer
from langfuse.langchain import CallbackHandler
from langgraph.types import Command
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter

from spendguard.graph.build import ReviewGraph
from spendguard.graph.state import ReviewState
from spendguard.privacy.redaction import redact
from spendguard.settings import Settings
from spendguard.structured_logging import log_event


def _redacted(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: _redacted(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redacted(item) for item in value]
    return value


def mask_personal_data(*, data: Any, **keyword_arguments: Any) -> Any:
    return _redacted(json.loads(json.dumps(data, cls=EventSerializer)))


@dataclass(frozen=True)
class Tracing:
    client: Langfuse
    public_key: str


def create_tracing(
    settings: Settings,
    span_exporter: SpanExporter | None = None,
    tracer_provider: TracerProvider | None = None,
) -> Tracing | None:
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        return None
    public_key = settings.langfuse_public_key.get_secret_value()
    client = Langfuse(
        public_key=public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_host,
        mask=mask_personal_data,
        span_exporter=span_exporter,
        tracer_provider=tracer_provider,
    )
    return Tracing(client=client, public_key=public_key)


def flush_traces_before_return[**Parameters, Result](
    function: Callable[Parameters, Awaitable[Result]],
) -> Callable[Parameters, Awaitable[Result]]:
    @functools.wraps(function)
    async def flushing(
        *arguments: Parameters.args, **keyword_arguments: Parameters.kwargs
    ) -> Result:
        try:
            return await function(*arguments, **keyword_arguments)
        finally:
            get_client().flush()

    return flushing


async def traced_advance(
    tracing: Tracing | None,
    graph: ReviewGraph,
    graph_input: ReviewState | Command[str] | None,
    thread: RunnableConfig,
    submission_id: str,
) -> None:
    audit_count_before = len((await graph.aget_state(thread)).values.get('audit', []))
    if tracing is None:
        await graph.ainvoke(graph_input, thread)
        trace_identifiers: dict[str, str] = {}
    else:
        handler = CallbackHandler(public_key=tracing.public_key)
        supplied_callbacks = thread.get('callbacks')
        if supplied_callbacks is not None and not isinstance(supplied_callbacks, list):
            raise TypeError('traced_advance merges callbacks given as a list only')
        traced_thread: RunnableConfig = {
            **thread,
            'callbacks': [*(supplied_callbacks or []), handler],
            'metadata': {'langfuse_session_id': submission_id},
        }
        await graph.ainvoke(graph_input, traced_thread)
        trace_identifiers = {'trace_id': handler.last_trace_id} if handler.last_trace_id else {}
    audit = (await graph.aget_state(thread)).values.get('audit', [])
    for entry in audit[audit_count_before:]:
        if tracing is not None and trace_identifiers:
            with propagate_attributes(session_id=submission_id):
                tracing.client.create_event(
                    trace_context={'trace_id': trace_identifiers['trace_id']},
                    name=entry['event'],
                    metadata={'node': entry['node'], **entry['detail']},
                )
        log_event(
            entry['event'], submission_id=submission_id, node=entry['node'], **trace_identifiers
        )
