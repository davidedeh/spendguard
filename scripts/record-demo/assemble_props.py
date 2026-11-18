import datetime
import json
import sys
from pathlib import Path

import httpx

from spendguard.demo import demo_samples
from spendguard.eval.freeze import CASE_DIRECTORY
from spendguard.settings import load_settings

OBSERVATIONS_PATH = '/api/public/v2/observations'


def review_seconds(trace_id: str, host: str, auth: tuple[str, str]) -> float:
    response = httpx.get(
        f'{host}{OBSERVATIONS_PATH}',
        params={'traceId': trace_id, 'fields': 'core,basic', 'limit': 100},
        auth=auth,
        timeout=30,
    )
    response.raise_for_status()
    [root] = [row for row in response.json()['data'] if row['parentObservationId'] is None]
    started, ended = (
        datetime.datetime.fromisoformat(root[field]) for field in ('startTime', 'endTime')
    )
    return round((ended - started).total_seconds(), 1)


def main() -> None:
    if len(sys.argv) != 6:
        sys.exit('usage: assemble_props.py <capture> <traces> <trace stills> <repository> <props>')
    capture, traces, stills = (json.loads(Path(path).read_text()) for path in sys.argv[1:4])
    settings = load_settings()
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        sys.exit('set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env')
    auth = (
        settings.langfuse_public_key.get_secret_value(),
        settings.langfuse_secret_key.get_secret_value(),
    )
    first_trace_ids = [
        traces[sample.idempotency_key][0]['url'].rsplit('/', 1)[1]
        for sample in demo_samples(CASE_DIRECTORY)
    ]
    props = {
        'lines': capture['lines'],
        'reviewSeconds': [
            review_seconds(trace_id, settings.langfuse_host, auth) for trace_id in first_trace_ids
        ],
        'primaryModelId': settings.chat_models[settings.primary_model_key].model_id,
        'recordedOn': datetime.date.today().isoformat(),
        'repositoryUrl': sys.argv[4],
        'trace': stills,
    }
    Path(sys.argv[5]).write_text(json.dumps(props, indent=2))


if __name__ == '__main__':
    main()
