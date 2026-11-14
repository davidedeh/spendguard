import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any, TextIO

LOGGER_NAME = 'spendguard'
EVENT_ATTRIBUTE = 'spendguard_event'
IDENTIFIERS_ATTRIBUTE = 'spendguard_identifiers'
LIBRARY_RECORD_EVENT = 'library_record'

logger = logging.getLogger(LOGGER_NAME)
event_name_pattern = re.compile(r'[a-z][a-z0-9_]{0,63}')
identifier_value_pattern = re.compile(r'[A-Za-z0-9_.:-]{1,128}')


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            'timestamp': datetime.fromtimestamp(record.created, UTC).isoformat(),
            'level': record.levelname.lower(),
        }
        if hasattr(record, EVENT_ATTRIBUTE):
            line['event'] = getattr(record, EVENT_ATTRIBUTE)
            line.update(getattr(record, IDENTIFIERS_ATTRIBUTE))
        else:
            line['event'] = LIBRARY_RECORD_EVENT
            line['logger'] = record.name
            if record.exc_info and record.exc_info[0] is not None:
                line['exception_type'] = record.exc_info[0].__name__
        return json.dumps(line)


def configure_logging(level: str, stream: TextIO = sys.stderr) -> None:
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def log_event(event: str, **identifiers: str | int) -> None:
    if not event_name_pattern.fullmatch(event):
        raise ValueError(f'Log event names are snake_case identifiers, not free text: {event!r}')
    for key, value in identifiers.items():
        if isinstance(value, str) and not identifier_value_pattern.fullmatch(value):
            raise ValueError(f'Log identifier {key} holds free text, not an ID')
    logger.info(
        event,
        extra={EVENT_ATTRIBUTE: event, IDENTIFIERS_ATTRIBUTE: identifiers},
    )
