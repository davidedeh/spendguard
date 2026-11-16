import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from settings_factory import make_test_settings
from spendguard.db.database import ConnectionPool
from spendguard.graph.config import GraphConfig
from spendguard.graph.nodes.execute import DecisionEvent
from spendguard.graph.nodes.run_judgement_checks import (
    TOOL_CALL_LIMIT_REACHED,
    CheckVerdict,
    JudgementReport,
    QuotedText,
    run_judgement_checks,
)
from spendguard.graph.state import ReviewState, SuppliedPrecedent
from spendguard.models.factory import build_embedder
from spendguard.models.fake import FakeChatModel, structured_response
from spendguard.policy.loader import POLICY_PATH, FindingCode, Policy, load_policy
from spendguard.privacy.prompt_data import DATA_TAG_INSTRUCTION, prompt_fields, untrusted_data
from spendguard.seed.loader import read_seed_directory
from spendguard.settings import Settings
from spendguard.store import precedents

POLICY = load_policy(POLICY_PATH)
SEED_DATA = read_seed_directory(Path(__file__).parent / 'fixtures' / 'seed')
SATURDAY_CLAIM: dict[str, Any] = {
    'type': 'expense_claim',
    'claimant': 'EMP-003',
    'spent_on': '2026-10-10',
    'category': 'subsistence',
    'quantity': 1,
    'merchant': 'Harbour Kitchen',
    'amount': '38.00',
    'currency': 'GBP',
    'claimed_exchange_rate': None,
    'business_purpose': 'Saturday dinner with the Leeds team for Halden Group',
    'receipt_text': 'Harbour Kitchen\n1 x Fish pie 24.00\n1 x Pint of lager 14.00\nTOTAL 38.00',
}


def reference_of(clause_id: str) -> str:
    return next(clause.reference for clause in POLICY.clauses if clause.id == clause_id)


ALCOHOL_CLAUSE = reference_of('no_alcohol')
JUDGEMENT_CODES = POLICY.codes_with_check('judgement')


class UnusedSender:
    async def send(self, event: DecisionEvent) -> None:
        raise AssertionError('judgement checks send no decision events')


def graph_config(
    chat_model: FakeChatModel, connection_pool: ConnectionPool, settings: Settings | None = None
) -> GraphConfig:
    configured_settings = settings or make_test_settings()
    return GraphConfig(
        settings=configured_settings,
        policy=POLICY,
        seed_data=SEED_DATA,
        chat_model=chat_model,
        embedder=build_embedder(configured_settings),
        connection_pool=connection_pool,
        checkpointer=InMemorySaver(),
        decision_event_sender=UnusedSender(),
        policy_context='retrieved_clauses',
    )


def claim_state(**claim_overrides: Any) -> ReviewState:
    claim = SATURDAY_CLAIM | claim_overrides
    return {
        'submission_id': 'submission-1',
        'idempotency_key': 'expense_claim:travel-desk:CLAIM-1',
        'submission_type': 'expense_claim',
        'submission': claim,
        'redacted_submission': claim,
        'requester': 'EMP-003',
        'check_guidance': [],
        'retrieved_clauses': [
            {
                'clause_reference': ALCOHOL_CLAUSE,
                'clause_hash': POLICY.finding_codes['ALCOHOL_NOT_PERMITTED'].clause.hash,
                'text': POLICY.finding_codes['ALCOHOL_NOT_PERMITTED'].clause.text,
            }
        ],
    }


INVOICE: dict[str, Any] = {
    'type': 'supplier_invoice',
    'supplier': 'SUP-002',
    'supplier_invoice_number': 'INV-1',
    'purchase_order': None,
    'cost_centre': 'CC-200',
    'issued_on': '2026-10-21',
    'amount': '85.00',
    'currency': 'GBP',
    'claimed_exchange_rate': None,
    'description': 'Printer toner cartridges for the Leeds office',
}


def invoice_state() -> ReviewState:
    return claim_state() | {
        'idempotency_key': 'supplier_invoice:SUP-002:INV-1',
        'submission_type': 'supplier_invoice',
        'submission': INVOICE,
        'redacted_submission': INVOICE,
        'requester': 'EMP-002',
    }


def verdict(code: str, outcome: str, **fields: Any) -> CheckVerdict:
    return CheckVerdict.model_validate(
        {
            'code': code,
            'outcome': outcome,
            'clause_reference': None,
            'quotes': [],
            'cited_precedent_ids': [],
            'precedent_id': None,
        }
        | fields
    )


def report(*verdicts: CheckVerdict) -> AIMessage:
    return structured_response(JudgementReport(verdicts=list(verdicts)))


async def test_a_flagged_check_becomes_a_candidate_finding_and_a_clear_one_an_outcome_only(
    connection_pool: ConnectionPool,
) -> None:
    model = FakeChatModel(
        model_id='fake-judge',
        responses=[
            AIMessage(content='I have what I need.'),
            report(
                verdict(
                    'ALCOHOL_NOT_PERMITTED',
                    'flagged',
                    clause_reference=ALCOHOL_CLAUSE,
                    quotes=[QuotedText(field='receipt_text', text='Pint of lager')],
                ),
                verdict('PURPOSE_UNCLEAR', 'not_flagged'),
            ),
        ],
    )

    update = await run_judgement_checks(claim_state(), graph_config(model, connection_pool))

    assert update['judgement_outcomes'] == [
        {
            'code': 'ALCOHOL_NOT_PERMITTED',
            'outcome': 'flagged',
            'precedent_id': None,
            'candidate': {
                'code': 'ALCOHOL_NOT_PERMITTED',
                'clause_reference': ALCOHOL_CLAUSE,
                'quotes': [{'field': 'receipt_text', 'text': 'Pint of lager'}],
                'cited_precedent_ids': [],
            },
        },
        {
            'code': 'PURPOSE_UNCLEAR',
            'outcome': 'not_flagged',
            'precedent_id': None,
            'candidate': None,
        },
    ]


def tool_calls(*calls: tuple[str, dict[str, Any]]) -> AIMessage:
    return AIMessage(
        content='',
        tool_calls=[
            {'name': name, 'args': args, 'id': f'call-{index}'}
            for index, (name, args) in enumerate(calls)
        ],
    )


def tool_results(messages: list[BaseMessage]) -> dict[str, Any]:
    return {
        message.tool_call_id: json.loads(message.text)
        for message in messages
        if isinstance(message, ToolMessage)
    }


async def test_lookups_answer_for_this_claim_whatever_identifiers_the_model_passes(
    connection_pool: ConnectionPool,
) -> None:
    model = FakeChatModel(
        model_id='fake-judge',
        responses=[
            tool_calls(
                ('get_employee', {'employee': 'EMP-001'}),
                ('get_cost_centre', {'cost_centre': 'CC-200'}),
                ('get_supplier', {'supplier': 'SUP-002'}),
                ('get_precedents', {'code': 'PURPOSE_UNCLEAR', 'employee': 'EMP-001'}),
            ),
            AIMessage(content='I have what I need.'),
            report(),
        ],
    )

    await run_judgement_checks(claim_state(), graph_config(model, connection_pool))

    assert tool_results(model.received[1]) == {
        'call-0': {
            'reference': 'EMP-003',
            'grade': 'G2',
            'cost_centre': 'CC-100',
            'manager': 'EMP-001',
            'approval_limit_gbp': None,
        },
        'call-1': {
            'reference': 'CC-100',
            'owner': 'EMP-001',
            'budget_gbp': '50000.00',
            'spend_to_date_gbp': '12000.00',
        },
        'call-2': {'error': 'This submission has no supplier.'},
        'call-3': [],
    }


async def test_lookups_for_an_invoice_answer_for_its_supplier_requester_and_cost_centre(
    connection_pool: ConnectionPool,
) -> None:
    state = invoice_state()
    model = FakeChatModel(
        model_id='fake-judge',
        responses=[
            tool_calls(
                ('get_employee', {}),
                ('get_cost_centre', {'cost_centre': 'CC-100'}),
                ('get_supplier', {'supplier': 'SUP-001'}),
            ),
            AIMessage(content='I have what I need.'),
            report(),
        ],
    )

    await run_judgement_checks(state, graph_config(model, connection_pool))

    assert tool_results(model.received[1]) == {
        'call-0': {
            'reference': 'EMP-002',
            'grade': 'G4',
            'cost_centre': 'CC-200',
            'manager': 'EMP-001',
            'approval_limit_gbp': '5000.00',
        },
        'call-1': {
            'reference': 'CC-200',
            'owner': 'EMP-002',
            'budget_gbp': '8000.00',
            'spend_to_date_gbp': '7900.00',
        },
        'call-2': {'reference': 'SUP-002', 'approved': False},
    }


def bound_tool_names(generate_arguments: dict[str, Any]) -> list[str]:
    return [tool['function']['name'] for tool in generate_arguments.get('tools', [])]


class ToolHungryChatModel(FakeChatModel):
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        reply = (
            report(verdict('PURPOSE_UNCLEAR', 'not_flagged'))
            if bound_tool_names(kwargs) == ['JudgementReport']
            else tool_calls(('get_employee', {}), ('get_cost_centre', {}))
        )
        return ChatResult(generations=[ChatGeneration(message=reply)])


@pytest.mark.parametrize(
    'settings', [make_test_settings(judgement_max_tool_calls=3)], ids=['a cap of three']
)
async def test_the_step_cap_stops_a_model_that_keeps_calling_tools_and_asks_for_its_verdict(
    connection_pool: ConnectionPool, settings: Settings
) -> None:
    model = ToolHungryChatModel(model_id='fake-judge')

    update = await run_judgement_checks(
        claim_state(), graph_config(model, connection_pool, settings)
    )

    assert len(model.received) == 3
    answered = [message for message in model.received[2] if isinstance(message, ToolMessage)]
    assert [message.text == TOOL_CALL_LIMIT_REACHED for message in answered] == [
        False,
        False,
        False,
        True,
    ]
    assert [outcome['code'] for outcome in update['judgement_outcomes']] == ['PURPOSE_UNCLEAR']
    assert update['audit'] == [
        {
            'node': 'run_judgement_checks',
            'event': 'tool_call_limit_reached',
            'detail': {'tool_calls': 3},
        }
    ]


@pytest.mark.parametrize(
    ('state', 'expected_checks'),
    [
        (
            claim_state(),
            [
                'WEEKEND_SPEND_UNJUSTIFIED',
                'PURPOSE_UNCLEAR',
                'RECEIPT_MISMATCH',
                'ALCOHOL_NOT_PERMITTED',
            ],
        ),
        (
            claim_state(spent_on='2026-10-09'),
            ['PURPOSE_UNCLEAR', 'RECEIPT_MISMATCH', 'ALCOHOL_NOT_PERMITTED'],
        ),
        (invoice_state(), ['PURPOSE_UNCLEAR']),
    ],
    ids=['a claim spent on a Saturday', 'a claim spent on a Friday', 'a supplier invoice'],
)
async def test_the_model_is_asked_only_the_judgement_checks_that_apply_to_the_submission(
    connection_pool: ConnectionPool, state: ReviewState, expected_checks: list[str]
) -> None:
    model = FakeChatModel(
        model_id='fake-judge', responses=[AIMessage(content='Nothing to look up.'), report()]
    )

    update = await run_judgement_checks(state, graph_config(model, connection_pool))

    assert update['judgement_checks'] == expected_checks
    system_prompt = model.received[0][0].text
    assert [code for code in JUDGEMENT_CODES if code in system_prompt] == [
        code for code in JUDGEMENT_CODES if code in expected_checks
    ]


async def test_the_judge_sees_the_retrieved_clauses_the_guidance_and_only_the_redacted_submission(
    connection_pool: ConnectionPool,
) -> None:
    model = FakeChatModel(
        model_id='fake-judge', responses=[AIMessage(content='Nothing to look up.'), report()]
    )
    state = claim_state(business_purpose='Team dinner, ask pat@example.com')
    state['redacted_submission'] = state['submission'] | {
        'business_purpose': 'Team dinner, ask [EMAIL]'
    }
    state['check_guidance'] = [{'code': 'ALCOHOL_NOT_PERMITTED', 'look_for': 'a pint on the bill'}]

    await run_judgement_checks(state, graph_config(model, connection_pool))

    [system_message, human_message] = model.received[0]
    assert DATA_TAG_INSTRUCTION in system_message.text
    assert f'**{ALCOHOL_CLAUSE}** {POLICY.finding_codes["ALCOHOL_NOT_PERMITTED"].clause.text}' in (
        system_message.text
    )
    assert 'a pint on the bill' in system_message.text
    assert human_message.text == untrusted_data(prompt_fields(state['redacted_submission']))
    assert 'pat@example.com' not in str(model.received)


async def test_a_call_to_a_tool_that_does_not_exist_is_answered_with_an_error(
    connection_pool: ConnectionPool,
) -> None:
    model = FakeChatModel(
        model_id='fake-judge',
        responses=[
            tool_calls(('search_policy', {'query': 'alcohol'})),
            AIMessage(content='I have what I need.'),
            report(),
        ],
    )

    await run_judgement_checks(claim_state(), graph_config(model, connection_pool))

    assert tool_results(model.received[1]) == {'call-0': {'error': 'No tool named search_policy.'}}


async def test_the_tool_supplies_the_requester_precedents_for_the_requested_code_only(
    connection_pool: ConnectionPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    purpose_precedent: SuppliedPrecedent = {
        'precedent_id': 'PRECEDENT-1',
        'code': 'PURPOSE_UNCLEAR',
        'clause_reference': reference_of('business_purpose'),
        'reason': 'Standing client visit',
    }
    weekend_precedent: SuppliedPrecedent = {
        'precedent_id': 'PRECEDENT-2',
        'code': 'WEEKEND_SPEND_UNJUSTIFIED',
        'clause_reference': reference_of('business_purpose'),
        'reason': 'Weekend trade fair',
    }
    stored = {
        ('EMP-003', 'PURPOSE_UNCLEAR'): [purpose_precedent],
        ('EMP-003', 'WEEKEND_SPEND_UNJUSTIFIED'): [weekend_precedent],
    }

    async def recent_precedents(
        connection_pool: ConnectionPool, subject: str, code: FindingCode, policy: Policy
    ) -> list[SuppliedPrecedent]:
        return stored.get((subject, code), [])

    monkeypatch.setattr(precedents, 'recent_precedents', recent_precedents)
    model = FakeChatModel(
        model_id='fake-judge',
        responses=[
            tool_calls(
                ('get_precedents', {'code': 'PURPOSE_UNCLEAR', 'subject': 'EMP-001'}),
                ('get_precedents', {'code': 'ALCOHOL_NOT_PERMITTED'}),
            ),
            AIMessage(content='I have what I need.'),
            report(),
        ],
    )

    update = await run_judgement_checks(claim_state(), graph_config(model, connection_pool))

    assert tool_results(model.received[1]) == {'call-0': [purpose_precedent], 'call-1': []}
    assert update['supplied_precedents'] == [purpose_precedent]
