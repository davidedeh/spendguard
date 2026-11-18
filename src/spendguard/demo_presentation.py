import asyncio
import re
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, TextIO

import httpx

from spendguard.db.database import ConnectionPool
from spendguard.demo import (
    DEMO_DISMISSAL_REASON,
    PROCESSING_TIMEOUT_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    ApprovalBlockedError,
    DecisionRefusedError,
    DemoCredentials,
    DemoSample,
    DemoSession,
    DemoStepError,
    NotSettledError,
    PostedDecision,
    PostedSample,
    demo_samples,
    executed_decisions,
    line_buffered_stdout,
    local_connection_pool,
    post_decisions,
    post_samples,
    settled_submissions,
    webhook_signing_key,
)
from spendguard.eval.freeze import CASE_DIRECTORY
from spendguard.store import decision_events
from spendguard.wording import counted

_SENTENCE_END_PATTERN = re.compile(r'(?<=[.!?])\s+')
_WORD_PATTERN = re.compile(r'[a-z]{4,}')

_RESET = '\x1b[0m'
_BOLD = '\x1b[1m'
_DIM = '\x1b[2m'
_GREEN = '\x1b[32m'
_AMBER = '\x1b[33m'
_RED = '\x1b[31m'
_CYAN = '\x1b[36m'


def _words(text: str) -> set[str]:
    return set(_WORD_PATTERN.findall(text.lower()))


def clause_excerpt(clause_text: str, quote_texts: Sequence[str]) -> str:
    sentences = _SENTENCE_END_PATTERN.split(' '.join(clause_text.split()))
    quoted_words = set().union(*(_words(text) for text in quote_texts))
    return max(sentences, key=lambda sentence: len(_words(sentence) & quoted_words))


def quoted(text: str) -> str:
    return '"' + ' / '.join(line.strip() for line in text.splitlines() if line.strip()) + '"'


def _badge(colour: str, text: str) -> str:
    return f'{_BOLD}{colour}{text}{_RESET}'


def _labelled(label: str, text: str) -> str:
    return f'{_DIM}{label}{_RESET}  {text}'


class _Printer:
    def __init__(self, output: TextIO) -> None:
        self.output = output

    def heading(self, *parts: str) -> None:
        print(
            f'{_BOLD}{parts[0]}{_RESET}' + ''.join(f' · {part}' for part in parts[1:]),
            file=self.output,
        )

    def line(self, text: str, indent: int = 2) -> None:
        print(' ' * indent + text, file=self.output)


def _capitalised(label: str) -> str:
    return label[:1].upper() + label[1:]


def _claim_heading(printer: _Printer, entry: PostedSample) -> None:
    claim = entry.sample.delivery['claim']
    amount = Decimal(str(claim['amount'])).quantize(Decimal('0.01'))
    printer.heading(
        _capitalised(entry.sample.label),
        entry.submission_id[:8],
        claim['claimant'],
        f'{amount} {claim["currency"]}',
    )
    printer.line(_labelled('purpose', quoted(claim['business_purpose'])))


def _finding_lines(printer: _Printer, finding: Mapping[str, Any], with_clause: bool) -> None:
    clause = finding['clause']
    printer.line(
        f'{_BOLD}{_CYAN}{finding["code"]}{_RESET}  {finding["severity"]} · '
        f'{clause["reference"]} {" > ".join(clause["heading_path"])}'
    )
    for quote in finding['quotes']:
        printer.line(_labelled('quote ', quoted(quote['text'])), indent=4)
    if with_clause:
        excerpt = clause_excerpt(clause['text'], [quote['text'] for quote in finding['quotes']])
        printer.line(_labelled('clause', quoted(excerpt)), indent=4)


async def _show_outcome(
    session: DemoSession,
    entry: PostedSample,
    detail: Mapping[str, Any],
    connection_pool: ConnectionPool,
) -> None:
    printer = _Printer(session.output)
    _claim_heading(printer, entry)
    if detail['status'] == 'complete':
        events = await decision_events.events_for(connection_pool, entry.submission_id)
        outcomes = ', '.join(event['outcome'] for event in events) or 'none'
        printer.line(
            f'{_badge(_GREEN, "AUTO-APPROVED")}  no findings · '
            f'mock ERP: {counted(len(events), "record")}, {outcomes}'
        )
        return
    review = detail['review']
    printer.line(
        f'{_badge(_AMBER, "NEEDS REVIEW")}  recommendation: {review["recommendation"] or "none"}'
    )
    for index, finding in enumerate(review['findings']):
        _finding_lines(printer, finding, with_clause=index == 0)


async def run_presentation(
    session: DemoSession,
    samples: Sequence[DemoSample],
    connection_pool: ConnectionPool,
) -> bool:
    printer = _Printer(session.output)
    try:
        posted = await post_samples(session, samples)
    except DemoStepError as error:
        printer.line(str(error), indent=0)
        return False
    details: dict[str, dict[str, Any]] = {}
    try:
        async for entry, detail in settled_submissions(session, posted, 'received'):
            details[entry.submission_id] = detail
            await _show_outcome(session, entry, detail, connection_pool)
    except NotSettledError as failure:
        printer.line(
            f'{failure.entry.sample.label}: not processed after {PROCESSING_TIMEOUT_SECONDS}s',
            indent=0,
        )
        return False
    return await _show_decisions(session, posted, details, connection_pool)


async def _show_decisions(
    session: DemoSession,
    posted: Sequence[PostedSample],
    details: Mapping[str, Mapping[str, Any]],
    connection_pool: ConnectionPool,
) -> bool:
    printer = _Printer(session.output)
    printer.heading('Approver decisions')
    decisions: list[PostedDecision] = []
    try:
        async for decision in post_decisions(session, posted, details):
            decisions.append(decision)
            label = decision.entry.sample.label
            if decision.decision == 'approve':
                dismissed = ', '.join(finding['code'] for finding in decision.findings)
                printer.line(
                    f'{_badge(_GREEN, "APPROVE")}  {label} · {dismissed} dismissed:'
                    f' "{DEMO_DISMISSAL_REASON}"'
                )
            else:
                printer.line(f'{_badge(_RED, "REJECT")}  {label} · "{decision.request["reason"]}"')
    except ApprovalBlockedError as failure:
        printer.line(f'not approving {failure.entry.sample.label} over {", ".join(failure.codes)}')
        return False
    except DecisionRefusedError as failure:
        printer.line(f'{failure.entry.sample.label}: {failure.status_code} {failure.text}')
        return False

    try:
        result = await executed_decisions(session, decisions, connection_pool)
    except NotSettledError as failure:
        printer.line(
            f'{failure.entry.sample.label}: not executed after {PROCESSING_TIMEOUT_SECONDS}s'
        )
        return False
    for executed in result.executed:
        if len(executed.events) != 1:
            printer.line(
                f'{executed.decision.entry.sample.label}: the mock ERP holds'
                f' {len(executed.events)} records'
            )
            return False
    events = [event for executed in result.executed for event in executed.events]
    approvers = ', '.join(sorted({event['decided_by'] for event in events}))
    outcomes = ' and '.join(event['outcome'] for event in events)
    printer.line(
        f'{_badge(_GREEN, "EXECUTED")}  mock ERP: 1 record each, {outcomes} by {approvers}'
    )

    for executed in result.executed:
        for precedent in executed.precedents:
            printer.line(
                f'{_badge(_CYAN, "PRECEDENT")}  {precedent.code} {precedent.clause_reference}'
                f' for {precedent.subject}: "{precedent.reason}"'
            )

    again = result.second_decision
    printer.line(
        f'{_badge(_RED, f"{again.status_code} CONFLICT")}  a second decision on the'
        f' {result.executed[0].decision.entry.sample.label}: {again.json()["detail"]}'
    )
    return again.status_code == 409


async def _present(base_url: str, token: str) -> bool:
    credentials = DemoCredentials(webhook_key=webhook_signing_key(), approver_token=token)
    async with (
        httpx.AsyncClient(base_url=base_url, timeout=REQUEST_TIMEOUT_SECONDS) as client,
        local_connection_pool() as connection_pool,
    ):
        session = DemoSession(
            client=client, credentials=credentials, sleep=asyncio.sleep, output=sys.stdout
        )
        return await run_presentation(session, demo_samples(CASE_DIRECTORY), connection_pool)


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit('usage: python -m spendguard.demo_presentation <api base URL> <approver token>')
    line_buffered_stdout()
    if not asyncio.run(_present(sys.argv[1], sys.argv[2])):
        sys.exit(1)


if __name__ == '__main__':
    main()
