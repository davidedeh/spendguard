import asyncio
import io
import json
import os
import sys
import textwrap
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, TextIO

import httpx
import yaml
from dotenv import dotenv_values

from spendguard import settings
from spendguard.db.database import ConnectionPool, async_connection_pool, branch_connection_info
from spendguard.eval.cases import EvalCase
from spendguard.eval.freeze import CASE_DIRECTORY
from spendguard.security.signatures import SIGNATURE_HEADER, sign
from spendguard.store import decision_events, precedents
from spendguard.store.precedents import RecordedPrecedent
from spendguard.submission import ExpenseClaim, ExpenseClaimDelivery
from spendguard.wording import counted

ENV_FILE = Path('.env')
REQUEST_TIMEOUT_SECONDS = 30
POLL_INTERVAL_SECONDS = 2
PROCESSING_TIMEOUT_SECONDS = 300

DEMO_SOURCE = 'demo'
DISMISSED_ON_APPROVAL = 'PURPOSE_UNCLEAR'
DEMO_DISMISSAL_REASON = 'Quarterly review with the Northwind account team'
DEMO_REJECTION_REASON = 'The purpose tries to instruct the reviewer instead of stating the spend'
DEMO_UPHELD_REASON = 'Upheld as raised'

DemoDecision = Literal['approve', 'reject']
DECIDE_FLAG = '--decide'
APPROVER_TOKEN_VARIABLE = 'APPROVER_TOKEN'

Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class DemoCase:
    label: str
    case_id: str
    decision: DemoDecision | None


INJECTION_CASE = DemoCase('claim with injected instructions', 'test-injection-01', 'reject')
DEMO_CASES = (
    DemoCase('clean small claim', 'test-clean-05', None),
    DemoCase('claim with an unclear purpose', 'test-judgement-positive-05', 'approve'),
    INJECTION_CASE,
)


@dataclass(frozen=True)
class DemoCredentials:
    webhook_key: bytes
    approver_token: str


@dataclass(frozen=True)
class DemoSample:
    label: str
    webhook_path: str
    delivery: Mapping[str, Any]
    decision: DemoDecision | None

    @property
    def idempotency_key(self) -> str:
        return ExpenseClaimDelivery.model_validate(self.delivery).idempotency_key


def demo_samples(case_directory: Path) -> list[DemoSample]:
    return [demo_sample(case_directory, case) for case in DEMO_CASES]


def demo_sample(case_directory: Path, case: DemoCase) -> DemoSample:
    return _expense_claim_sample(
        case.label, frozen_case(case_directory, case.case_id), case.decision
    )


def frozen_case(case_directory: Path, case_id: str) -> EvalCase:
    return EvalCase.model_validate(yaml.safe_load((case_directory / f'{case_id}.yaml').read_text()))


def _expense_claim_sample(label: str, case: EvalCase, decision: DemoDecision | None) -> DemoSample:
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
        decision=decision,
    )


@dataclass(frozen=True)
class PostedSample:
    sample: DemoSample
    submission_id: str


@dataclass(frozen=True)
class DemoSession:
    client: httpx.AsyncClient
    credentials: DemoCredentials
    sleep: Sleep
    output: TextIO

    @property
    def authorization(self) -> dict[str, str]:
        return {'Authorization': f'Bearer {self.credentials.approver_token}'}


class DemoStepError(Exception):
    pass


async def post_samples(session: DemoSession, samples: Sequence[DemoSample]) -> list[PostedSample]:
    posted: list[PostedSample] = []
    for sample in samples:
        raw_body = json.dumps(sample.delivery).encode()
        response = await session.client.post(
            sample.webhook_path,
            content=raw_body,
            headers={
                SIGNATURE_HEADER: sign(session.credentials.webhook_key, int(time.time()), raw_body),
                'Content-Type': 'application/json',
            },
        )
        if response.status_code != 202:
            raise DemoStepError(f'{sample.label}: {response.status_code} {response.text}')
        posted.append(PostedSample(sample, response.json()['submission_id']))
    return posted


def unexpected_on_approval(
    decision: DemoDecision, findings: Sequence[Mapping[str, Any]]
) -> list[str]:
    return [
        finding['code']
        for finding in findings
        if decision == 'approve' and finding['code'] != DISMISSED_ON_APPROVAL
    ]


async def run_demo(
    session: DemoSession,
    samples: Sequence[DemoSample],
    connection_pool: ConnectionPool | None,
) -> bool:
    client, output = session.client, session.output

    print(f'Posting {len(samples)} signed sample submissions', file=output)
    try:
        posted = await post_samples(session, samples)
    except DemoStepError as error:
        print(f'  {error}', file=output)
        return False
    for entry in posted:
        print(f'  {entry.sample.label}: accepted as {entry.submission_id}', file=output)

    print('Waiting for the worker', file=output)
    details: dict[str, dict[str, Any]] = {}
    try:
        async for entry, detail in settled_submissions(session, posted, 'received'):
            details[entry.submission_id] = detail
            print(f'  {entry.sample.label}: {detail["status"]}', file=output)
    except NotSettledError as failure:
        print(
            f'  {failure.entry.sample.label}: not processed after {PROCESSING_TIMEOUT_SECONDS}s;'
            ' is make run-worker running?',
            file=output,
        )
        return False

    listed = await client.get(
        '/submissions', params={'status': 'pending_review'}, headers=session.authorization
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
    if connection_pool is None:
        return True
    return await _show_decisions(session, posted, details, connection_pool)


async def _show_decisions(
    session: DemoSession,
    posted: Sequence[PostedSample],
    details: Mapping[str, Mapping[str, Any]],
    connection_pool: ConnectionPool,
) -> bool:
    output = session.output
    print('Decisions as the approver', file=output)
    decisions: list[PostedDecision] = []
    try:
        async for decision in post_decisions(session, posted, details):
            decisions.append(decision)
            described = _described(decision.request, decision.findings)
            print(f'  {decision.entry.sample.label}: {described}', file=output)
    except ApprovalBlockedError as failure:
        print(
            f'  {failure.entry.sample.label}: not approving over {", ".join(failure.codes)}',
            file=output,
        )
        return False
    except DecisionRefusedError as failure:
        print(f'  {failure.entry.sample.label}: {failure.status_code} {failure.text}', file=output)
        return False

    print('Waiting for the decisions to execute', file=output)
    try:
        result = await executed_decisions(session, decisions, connection_pool)
    except NotSettledError as failure:
        print(
            f'  {failure.entry.sample.label}: not executed after {PROCESSING_TIMEOUT_SECONDS}s',
            file=output,
        )
        return False
    for executed in result.executed:
        print(f'  {executed.decision.entry.sample.label}: {executed.status}', file=output)

    print('Mock ERP records', file=output)
    for executed in result.executed:
        described = '; '.join(_erp_record(event) for event in executed.events) or 'none'
        print(
            f'  {executed.decision.entry.sample.label}:'
            f' {counted(len(executed.events), "record")}, {described}',
            file=output,
        )

    print('Precedents written from the dismissal', file=output)
    for executed in result.executed:
        for precedent in executed.precedents:
            print(
                f'  {precedent.code} on {precedent.clause_reference} for {precedent.subject}:'
                f' "{precedent.reason}"',
                file=output,
            )

    print('Second decision on the same claim', file=output)
    again = result.second_decision
    print(
        f'  Deciding {result.executed[0].decision.entry.sample.label} again:'
        f' {again.status_code} {again.json()["detail"]}',
        file=output,
    )
    return again.status_code == 409


@dataclass(frozen=True)
class PostedDecision:
    entry: PostedSample
    decision: DemoDecision
    findings: Sequence[Mapping[str, Any]]
    request: dict[str, Any]


@dataclass(frozen=True)
class ExecutedDecision:
    decision: PostedDecision
    status: str
    events: list[dict[str, Any]]
    precedents: list[RecordedPrecedent]


@dataclass(frozen=True)
class ExecutedDecisions:
    executed: list[ExecutedDecision]
    second_decision: httpx.Response


class ApprovalBlockedError(DemoStepError):
    def __init__(self, entry: PostedSample, codes: Sequence[str]) -> None:
        super().__init__(entry.sample.label)
        self.entry = entry
        self.codes = codes


class DecisionRefusedError(DemoStepError):
    def __init__(self, entry: PostedSample, status_code: int, text: str) -> None:
        super().__init__(entry.sample.label)
        self.entry = entry
        self.status_code = status_code
        self.text = text


class NotSettledError(DemoStepError):
    def __init__(self, entry: PostedSample) -> None:
        super().__init__(entry.sample.label)
        self.entry = entry


async def post_decisions(
    session: DemoSession,
    posted: Sequence[PostedSample],
    details: Mapping[str, Mapping[str, Any]],
) -> AsyncIterator[PostedDecision]:
    to_decide = [
        (entry, decision, details[entry.submission_id]['review']['findings'])
        for entry in posted
        if (decision := entry.sample.decision) is not None
    ]
    for entry, decision, findings in to_decide:
        unexpected = unexpected_on_approval(decision, findings)
        if unexpected:
            raise ApprovalBlockedError(entry, unexpected)
        request = decision_request(decision, findings)
        response = await session.client.post(
            f'/submissions/{entry.submission_id}/decision',
            json=request,
            headers=session.authorization,
        )
        if response.status_code != 202:
            raise DecisionRefusedError(entry, response.status_code, response.text)
        yield PostedDecision(entry, decision, findings, request)


async def executed_decisions(
    session: DemoSession,
    decisions: Sequence[PostedDecision],
    connection_pool: ConnectionPool,
) -> ExecutedDecisions:
    settled = [
        detail
        async for _entry, detail in settled_submissions(
            session, [decision.entry for decision in decisions], 'decided'
        )
    ]
    executed = [
        ExecutedDecision(
            decision=decision,
            status=detail['status'],
            events=await decision_events.events_for(connection_pool, decision.entry.submission_id),
            precedents=await precedents.precedents_recorded_from(
                connection_pool, uuid.UUID(decision.entry.submission_id)
            ),
        )
        for decision, detail in zip(decisions, settled, strict=True)
    ]
    first = decisions[0]
    second_decision = await session.client.post(
        f'/submissions/{first.entry.submission_id}/decision',
        json=first.request,
        headers=session.authorization,
    )
    return ExecutedDecisions(executed=executed, second_decision=second_decision)


def decision_request(
    decision: DemoDecision, findings: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    dismissing = decision == 'approve'
    return {
        'outcome': decision,
        'reason': None if dismissing else DEMO_REJECTION_REASON,
        'dispositions': [
            {
                'finding_id': finding['finding_id'],
                'verdict': 'dismissed' if dismissing else 'upheld',
                'reason': DEMO_DISMISSAL_REASON if dismissing else DEMO_UPHELD_REASON,
            }
            for finding in findings
        ],
    }


def _described(request: Mapping[str, Any], findings: Sequence[Mapping[str, Any]]) -> str:
    codes = {finding['finding_id']: finding['code'] for finding in findings}
    dispositions = ', '.join(
        f'{codes[disposition["finding_id"]]} {disposition["verdict"]}'
        + (f': "{disposition["reason"]}"' if disposition['verdict'] == 'dismissed' else '')
        for disposition in request['dispositions']
    )
    reason = f': "{request["reason"]}"' if request['reason'] else ''
    return f'{request["outcome"]}{reason}, {dispositions}'


def _erp_record(event: Mapping[str, Any]) -> str:
    reason = f': "{event["reason"]}"' if event['reason'] else ''
    return f'{event["outcome"]} by {event["decided_by"]}{reason}'


async def settled_submissions(
    session: DemoSession, posted: Sequence[PostedSample], waiting_status: str
) -> AsyncIterator[tuple[PostedSample, dict[str, Any]]]:
    for entry in posted:
        detail = await _settled_submission(session, entry.submission_id, waiting_status)
        if detail is None:
            raise NotSettledError(entry)
        yield entry, detail


async def _settled_submission(
    session: DemoSession, submission_id: str, waiting_status: str
) -> dict[str, Any] | None:
    for _ in range(PROCESSING_TIMEOUT_SECONDS // POLL_INTERVAL_SECONDS):
        await session.sleep(POLL_INTERVAL_SECONDS)
        response = await session.client.get(
            f'/submissions/{submission_id}', headers=session.authorization
        )
        response.raise_for_status()
        detail: dict[str, Any] = response.json()
        if detail['status'] != waiting_status:
            return detail
    return None


def _finding_to_show(
    posted: Sequence[PostedSample], details: Mapping[str, Mapping[str, Any]]
) -> tuple[str, Mapping[str, Any]] | None:
    raised = [
        (entry.sample.label, finding)
        for entry in posted
        for finding in details[entry.submission_id]['review']['findings']
    ]
    quoted = [(label, finding) for label, finding in raised if finding['quotes']]
    return next(iter(quoted or raised), None)


def approver_token() -> str:
    token = os.environ.get(APPROVER_TOKEN_VARIABLE)
    if not token:
        sys.exit(f'set {APPROVER_TOKEN_VARIABLE} to the token make issue-token printed')
    return token


def webhook_signing_key() -> bytes:
    if os.environ.get('SSM_PARAMETER_PREFIX') or dotenv_values(ENV_FILE).get(
        'SSM_PARAMETER_PREFIX'
    ):
        sys.exit('make demo reads its keys from .env only: unset SSM_PARAMETER_PREFIX')
    key = settings.load_settings(ENV_FILE).webhook_signing_key
    if key is None:
        sys.exit('set WEBHOOK_SIGNING_KEY in .env to the key the API verifies with')
    return key.get_secret_value().encode()


async def _demo(base_url: str, token: str, decide: bool) -> bool:
    credentials = DemoCredentials(webhook_key=webhook_signing_key(), approver_token=token)
    samples = demo_samples(CASE_DIRECTORY)
    async with httpx.AsyncClient(base_url=base_url, timeout=REQUEST_TIMEOUT_SECONDS) as client:
        session = DemoSession(
            client=client, credentials=credentials, sleep=asyncio.sleep, output=sys.stdout
        )
        if not decide:
            return await run_demo(session, samples, connection_pool=None)
        async with local_connection_pool() as connection_pool:
            return await run_demo(session, samples, connection_pool=connection_pool)


def local_connection_pool() -> ConnectionPool:
    return async_connection_pool(branch_connection_info(settings.load_settings(ENV_FILE)))


def line_buffered_stdout() -> None:
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)


def main() -> None:
    arguments = sys.argv[1:]
    decide = DECIDE_FLAG in arguments
    positional = [argument for argument in arguments if argument != DECIDE_FLAG]
    if len(positional) != 1:
        sys.exit(
            f'usage: {APPROVER_TOKEN_VARIABLE}=<approver token> python -m spendguard.demo'
            f' <api base URL> [{DECIDE_FLAG}]'
        )
    token = approver_token()
    line_buffered_stdout()
    if not asyncio.run(_demo(positional[0], token, decide)):
        sys.exit(1)


if __name__ == '__main__':
    main()
