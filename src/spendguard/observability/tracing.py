import base64
import functools
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse, get_client, propagate_attributes
from langfuse._client.environment_variables import (
    LANGFUSE_OTEL_TRACES_EXPORT_PATH,
    LANGFUSE_TIMEOUT,
)
from langfuse._client.span_processor import (
    _resolve_compression,
    _resolve_max_batch_size_bytes,
)
from langfuse._version import __version__ as langfuse_version
from langfuse.langchain import CallbackHandler
from langfuse.types import (
    MaskOtelSpansParams,
    MaskOtelSpansResult,
    OtelSpanIdentifier,
    OtelSpanPatch,
)
from langgraph.types import Command
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Status

from spendguard.graph.build import ReviewGraph
from spendguard.graph.state import ReviewState
from spendguard.privacy.redaction import redact
from spendguard.settings import Settings
from spendguard.structured_logging import log_event


def _redacted_text(attributes: Mapping[str, object]) -> dict[str, str | list[str]]:
    redacted: dict[str, str | list[str]] = {}
    for key, value in attributes.items():
        if isinstance(value, str) and redact(value) != value:
            redacted[key] = redact(value)
        elif isinstance(value, Sequence) and not isinstance(value, str):
            texts = [item for item in value if isinstance(item, str)]
            if len(texts) == len(value) and [redact(text) for text in texts] != texts:
                redacted[key] = [redact(text) for text in texts]
    return redacted


def mask_span_attributes(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult:
    patches: dict[OtelSpanIdentifier, OtelSpanPatch | None] = {}
    for identifier, span in params.spans.items():
        redacted = _redacted_text(span.attributes)
        patches[identifier] = OtelSpanPatch(set_attributes=redacted) if redacted else None
    return MaskOtelSpansResult(span_patches=patches)


class RedactingSpanExporter(SpanExporter):
    def __init__(self, exporter: SpanExporter) -> None:
        self._exporter = exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._exporter.export([_with_redacted_status_and_events(span) for span in spans])

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._exporter.force_flush(timeout_millis)


def _with_redacted_status_and_events(span: ReadableSpan) -> ReadableSpan:
    description = span.status.description
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=span.attributes,
        events=[
            Event(
                name=event.name,
                attributes={
                    **(event.attributes or {}),
                    **_redacted_text(event.attributes or {}),
                },
                timestamp=event.timestamp,
            )
            for event in span.events
        ],
        links=span.links,
        kind=span.kind,
        status=Status(span.status.status_code, redact(description) if description else None),
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def _langfuse_otlp_exporter(public_key: str, secret_key: str, base_url: str) -> SpanExporter:
    # Langfuse skips building its own exporter once one is supplied, so this mirrors its defaults.
    credentials = base64.b64encode(f'{public_key}:{secret_key}'.encode()).decode('ascii')
    traces_export_path = os.environ.get(LANGFUSE_OTEL_TRACES_EXPORT_PATH)
    return OTLPSpanExporter(
        endpoint=f'{base_url}/{traces_export_path}'
        if traces_export_path
        else f'{base_url}/api/public/otel/v1/traces',
        headers={
            'Authorization': f'Basic {credentials}',
            'x-langfuse-sdk-name': 'python',
            'x-langfuse-sdk-version': langfuse_version,
            'x-langfuse-public-key': public_key,
            'x-langfuse-ingestion-version': '4',
        },
        timeout=int(os.environ.get(LANGFUSE_TIMEOUT, '5')),
        compression=_resolve_compression(None),
        max_request_size=_resolve_max_batch_size_bytes(),
    )


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
    secret_key = settings.langfuse_secret_key.get_secret_value()
    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=settings.langfuse_host,
        mask_otel_spans=mask_span_attributes,
        span_exporter=RedactingSpanExporter(
            span_exporter or _langfuse_otlp_exporter(public_key, secret_key, settings.langfuse_host)
        ),
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
