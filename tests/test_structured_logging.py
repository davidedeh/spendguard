import io
import json
import logging
from collections.abc import Iterator

import pytest

from spendguard.structured_logging import configure_logging, log_event


@pytest.fixture
def log_stream() -> Iterator[io.StringIO]:
    root = logging.getLogger()
    original_handlers, original_level = root.handlers[:], root.level
    stream = io.StringIO()
    configure_logging('INFO', stream)
    yield stream
    root.handlers, root.level = original_handlers, original_level


def test_an_event_is_one_json_line_carrying_its_identifiers(log_stream: io.StringIO) -> None:
    log_event('node_finished', submission_id='sub-42', node='verify', trace_id='tr-9')

    [line] = log_stream.getvalue().splitlines()
    record = json.loads(line)
    assert record['event'] == 'node_finished'
    assert record['level'] == 'info'
    assert (record['submission_id'], record['node'], record['trace_id']) == (
        'sub-42',
        'verify',
        'tr-9',
    )
    assert set(record) == {'timestamp', 'level', 'event', 'submission_id', 'node', 'trace_id'}


@pytest.mark.parametrize(
    ('event', 'identifiers'),
    [
        ('node_finished', {'submission_id': 'Dinner with the client at Rosa'}),
        ('node_finished', {'purpose': 'x' * 200}),
        ('Receipt looked fine', {'submission_id': 'sub-42'}),
    ],
)
def test_free_text_is_refused_in_events_and_identifiers(
    log_stream: io.StringIO, event: str, identifiers: dict[str, str]
) -> None:
    with pytest.raises(ValueError, match='free text'):
        log_event(event, **identifiers)

    assert log_stream.getvalue() == ''


def test_a_library_log_record_is_json_without_its_message_text(log_stream: io.StringIO) -> None:
    try:
        raise KeyError('card 4111 1111 1111 1111')
    except KeyError:
        logging.getLogger('uvicorn.error').exception('Request body %s', 'Dinner at Rosa')

    [line] = log_stream.getvalue().splitlines()
    assert json.loads(line) | {'timestamp': None} == {
        'timestamp': None,
        'level': 'error',
        'event': 'library_record',
        'logger': 'uvicorn.error',
        'exception_type': 'KeyError',
    }
