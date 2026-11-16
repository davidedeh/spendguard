import asyncio
import io
import json
import os
import sys
import textwrap
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, TextIO

import httpx
import yaml
from dotenv import dotenv_values

from spendguard import settings
from spendguard.eval.cases import EvalCase
from spendguard.eval.freeze import CASE_DIRECTORY
from spendguard.security.signatures import SIGNATURE_HEADER, sign
from spendguard.submission import ExpenseClaim

ENV_FILE = Path('.env')
REQUEST_TIMEOUT_SECONDS = 30
POLL_INTERVAL_SECONDS = 2
PROCESSING_TIMEOUT_SECONDS = 300

DEMO_SOURCE = 'demo'
DEMO_CASES = (
    ('clean small claim', 'test-clean-05'),
    ('claim with an unclear purpose', 'test-judgement-positive-05'),
    ('claim with injected instructions', 'test-injection-01'),
)

Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class DemoCredentials:
    webhook_key: bytes
    approver_token: str


@dataclass(frozen=True)
class DemoSample:
    label: str
    webhook_path: str
    delivery: Mapping[str, Any]


def demo_samples(case_directory: Path) -> list[DemoSample]:
    return [
        _expense_claim_sample(label, _frozen_case(case_directory, case_id))
        for label, case_id in DEMO_CASES
    ]


def _frozen_case(case_directory: Path, case_id: str) -> EvalCase:
    return EvalCase.model_validate(yaml.safe_load((case_directory / f'{case_id}.yaml').read_text()))


def _expense_claim_sample(label: str, case: EvalCase) -> DemoSample:
    if not isinstance(case.submission, ExpenseClaim):
        raise TypeError(f'{case.id} is not an expense claim')
    return DemoSample(
        label=label,
        webhook_path='/webhooks/expense-claims',
        delivery={
            'source': DEMO_SOURCE,
            'claim_id': case.id,
            'claim': case.submission.model_dump(mode='json'),
        },
    )


@dataclass(frozen=True)
class _PostedSample:
    sample: DemoSample
    submission_id: str


async def run_demo(
    client: httpx.AsyncClient,
    credentials: DemoCredentials,
    samples: Sequence[DemoSample],
    sleep: Sleep,
    output: TextIO,
) -> bool:
    authorization = {'Authorization': f'Bearer {credentials.approver_token}'}

    print(f'Posting {len(samples)} signed sample submissions', file=output)
    posted: list[_PostedSample] = []
    for sample in samples:
        raw_body = json.dumps(sample.delivery).encode()
        response = await client.post(
            sample.webhook_path,
            content=raw_body,
            headers={
                SIGNATURE_HEADER: sign(credentials.webhook_key, int(time.time()), raw_body),
                'Content-Type': 'application/json',
            },
        )
        if response.status_code != 202:
            print(f'  {sample.label}: {response.status_code} {response.text}', file=output)
            return False
        submission_id = response.json()['submission_id']
        print(f'  {sample.label}: accepted as {submission_id}', file=output)
        posted.append(_PostedSample(sample, submission_id))

    print('Waiting for the worker', file=output)
    details: dict[str, dict[str, Any]] = {}
    for entry in posted:
        detail = await _processed_submission(client, authorization, entry.submission_id, sleep)
        if detail is None:
            print(
                f'  {entry.sample.label}: not processed after {PROCESSING_TIMEOUT_SECONDS}s;'
                ' is make run-worker running?',
                file=output,
            )
            return False
        details[entry.submission_id] = detail
        print(f'  {entry.sample.label}: {detail["status"]}', file=output)

    listed = await client.get(
        '/submissions', params={'status': 'pending_review'}, headers=authorization
    )
    listed.raise_for_status()
    pending_by_id = {review['submission_id']: review for review in listed.json()['submissions']}
    pending = [
        (entry.sample.label, pending_by_id[entry.submission_id])
        for entry in posted
        if entry.submission_id in pending_by_id
    ]
    print(f'Pending reviews from this demo: {len(pending)}', file=output)
    for label, review in pending:
        codes = ', '.join(finding['code'] for finding in review['findings']) or 'no findings'
        amount = Decimal(review['base_currency_amount']).quantize(Decimal('0.01'))
        print(
            f'  {label}: {review["submission_id"]}, requester {review["requester"]},'
            f' {amount} in base currency, recommendation {review["recommendation"]},'
            f' findings {codes}',
            file=output,
        )

    shown = _finding_to_show(posted, details)
    if shown is None:
        print('No finding was raised on the samples', file=output)
        return False
    label, finding = shown
    print(f'Finding on {label}:', file=output)
    print(
        f'  {finding["code"]}, {finding["severity"]} severity, from {finding["source"]}',
        file=output,
    )
    clause = finding['clause']
    print(f'  Clause {clause["reference"]}: {" > ".join(clause["heading_path"])}', file=output)
    print(textwrap.indent(clause['text'], '    '), file=output)
    for quote in finding['quotes']:
        print(f'  Quote from {quote["field"]}: "{quote["text"]}"', file=output)
    if not finding['quotes']:
        print('  No quotes: this finding comes from a rule', file=output)
    return True


async def _processed_submission(
    client: httpx.AsyncClient, authorization: Mapping[str, str], submission_id: str, sleep: Sleep
) -> dict[str, Any] | None:
    for _ in range(PROCESSING_TIMEOUT_SECONDS // POLL_INTERVAL_SECONDS):
        await sleep(POLL_INTERVAL_SECONDS)
        response = await client.get(f'/submissions/{submission_id}', headers=authorization)
        response.raise_for_status()
        detail: dict[str, Any] = response.json()
        if detail['status'] != 'received':
            return detail
    return None


def _finding_to_show(
    posted: Sequence[_PostedSample], details: Mapping[str, Mapping[str, Any]]
) -> tuple[str, Mapping[str, Any]] | None:
    raised = [
        (entry.sample.label, finding)
        for entry in posted
        for finding in details[entry.submission_id]['review']['findings']
    ]
    quoted = [(label, finding) for label, finding in raised if finding['quotes']]
    return next(iter(quoted or raised), None)


def _webhook_signing_key() -> bytes:
    if os.environ.get('SSM_PARAMETER_PREFIX') or dotenv_values(ENV_FILE).get(
        'SSM_PARAMETER_PREFIX'
    ):
        sys.exit('make demo reads its keys from .env only: unset SSM_PARAMETER_PREFIX')
    key = settings.load_settings(ENV_FILE).webhook_signing_key
    if key is None:
        sys.exit('set WEBHOOK_SIGNING_KEY in .env to the key the API verifies with')
    return key.get_secret_value().encode()


async def _demo(base_url: str, token: str) -> bool:
    async with httpx.AsyncClient(base_url=base_url, timeout=REQUEST_TIMEOUT_SECONDS) as client:
        return await run_demo(
            client,
            DemoCredentials(webhook_key=_webhook_signing_key(), approver_token=token),
            samples=demo_samples(CASE_DIRECTORY),
            sleep=asyncio.sleep,
            output=sys.stdout,
        )


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit('usage: python -m spendguard.demo <api base URL> <approver token>')
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    if not asyncio.run(_demo(sys.argv[1], sys.argv[2])):
        sys.exit(1)


if __name__ == '__main__':
    main()
