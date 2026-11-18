import datetime
import json
import secrets
import sys
from pathlib import Path

import psycopg
from langfuse import Langfuse
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from spendguard.db.database import branch_connection_info
from spendguard.observability.tracing import langfuse_otlp_exporter
from spendguard.settings import load_settings

SHARED_SPAN_NAME = 'shared_for_demo'
TRACE_PUBLIC_ATTRIBUTE = 'langfuse.trace.public'


def first_logged_times(worker_log: Path) -> dict[str, dict[str, int]]:
    found: dict[str, dict[str, int]] = {}
    for line in worker_log.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        submission_id, trace_id = record.get('submission_id'), record.get('trace_id')
        if submission_id and trace_id and trace_id not in found.setdefault(submission_id, {}):
            logged_at = datetime.datetime.fromisoformat(record['timestamp'])
            found[submission_id][trace_id] = int(logged_at.timestamp() * 1_000_000_000)
    return found


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit('usage: share_traces.py <worker log>')
    settings = load_settings()
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        sys.exit('set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env')
    provider = TracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(
            langfuse_otlp_exporter(
                settings.langfuse_public_key.get_secret_value(),
                settings.langfuse_secret_key.get_secret_value(),
                settings.langfuse_host,
            )
        )
    )
    tracer = provider.get_tracer('spendguard.record-demo')
    traces = first_logged_times(Path(sys.argv[1]))
    for logged in traces.values():
        for trace_id, logged_at in logged.items():
            parent = SpanContext(
                trace_id=int(trace_id, 16),
                span_id=secrets.randbits(64),
                is_remote=True,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
            span = tracer.start_span(
                SHARED_SPAN_NAME,
                context=trace.set_span_in_context(NonRecordingSpan(parent)),
                start_time=logged_at,
                attributes={TRACE_PUBLIC_ATTRIBUTE: True},
            )
            span.end(end_time=logged_at)
    provider.shutdown()
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key.get_secret_value(),
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_host,
        tracing_enabled=False,
    )
    with psycopg.connect(branch_connection_info(settings)) as connection:
        idempotency_keys = dict(
            connection.execute(
                'select id::text, idempotency_key from submissions where id::text = any(%s)',
                (list(traces),),
            ).fetchall()
        )
    print(
        json.dumps(
            {
                idempotency_keys[submission_id]: [
                    {
                        'url': langfuse.get_trace_url(trace_id=trace_id),
                        'logged_at': logged_at / 1_000_000_000,
                    }
                    for trace_id, logged_at in logged.items()
                ]
                for submission_id, logged in traces.items()
            }
        )
    )


if __name__ == '__main__':
    main()
