import asyncio
import json
from collections.abc import Sequence
from typing import Any, assert_never

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from spendguard.graph.config import GraphConfig
from spendguard.graph.judgement_tools import submission_tools
from spendguard.graph.nodes.intake import SUBMISSION_ADAPTER
from spendguard.graph.precedents import current_precedents
from spendguard.graph.state import (
    AuditEntry,
    CheckGuidance,
    JudgementOutcome,
    JudgementOutcomeKind,
    RetrievedClause,
    ReviewState,
    SuppliedPrecedent,
)
from spendguard.models.structured_output import output_schema, structured_reply
from spendguard.policy.loader import FindingCode, Policy
from spendguard.privacy.prompt_data import DATA_TAG_INSTRUCTION, prompt_fields, untrusted_data
from spendguard.rules.checks import judgement_checks
from spendguard.strict_model import StrictModel
from spendguard.submission import Submission

VERDICT_REQUEST = 'Now give your verdict for every check.'
TOOL_CALL_LIMIT_REACHED = 'Not run: the tool call limit for this submission is reached.'


class QuotedText(StrictModel):
    field: str
    text: str


class CheckVerdict(StrictModel):
    code: FindingCode
    outcome: JudgementOutcomeKind
    clause_reference: str | None
    quotes: list[QuotedText]
    cited_precedent_ids: list[str]
    precedent_id: str | None


class JudgementReport(StrictModel):
    verdicts: list[CheckVerdict]


JUDGEMENT_REPORT_SCHEMA: dict[str, Any] = output_schema(JudgementReport)


def _outcome(verdict: CheckVerdict) -> JudgementOutcome:
    return {
        'code': verdict.code,
        'outcome': verdict.outcome,
        'precedent_id': verdict.precedent_id,
        'candidate': {
            'code': verdict.code,
            'clause_reference': verdict.clause_reference or '',
            'quotes': [{'field': quote.field, 'text': quote.text} for quote in verdict.quotes],
            'cited_precedent_ids': verdict.cited_precedent_ids,
        }
        if verdict.outcome == 'flagged'
        else None,
    }


def judgement_instructions(
    policy: Policy,
    checks: Sequence[FindingCode],
    guidance: Sequence[CheckGuidance],
    policy_text: str,
) -> str:
    look_for = {item['code']: item['look_for'] for item in guidance}
    check_lines = '\n'.join(
        f'- {code} ({policy.finding_codes[code].clause.reference}'
        f' {" > ".join(policy.finding_codes[code].clause.heading_path)})'
        + (f': look for {look_for[code]}' if look_for.get(code) else '')
        for code in checks
    )
    return (
        f'You review one submission against the {policy.title}. Run each of these judgement'
        ' checks and give every one a verdict: flagged, not_flagged, or cleared_by_precedent'
        ' with the precedent_id that clears it.\n\n'
        f'{check_lines}\n\n'
        'You may call the tools to look up company records for this submission and earlier'
        ' precedents for a finding code. Tool calls are limited, so call only what you need.\n\n'
        'For a flagged check, give the clause_reference of the policy clause it breaks, chosen'
        ' from the clauses below, and quote the evidence: each quote names the submission'
        ' field and copies text from that field exactly, character for character. Cite by'
        ' precedent_id any precedent you rely on.\n\n'
        f'Policy clauses:\n\n{policy_text}\n\n'
        f'{DATA_TAG_INSTRUCTION}'
    )


def _clause_blocks(retrieved_clauses: Sequence[RetrievedClause]) -> str:
    return '\n\n'.join(
        f'**{clause["clause_reference"]}** {clause["text"]}' for clause in retrieved_clauses
    )


def _policy_text(state: ReviewState, graph_config: GraphConfig) -> str:
    match graph_config.policy_context:
        case 'retrieved_clauses':
            return _clause_blocks(state['retrieved_clauses'])
        case 'whole_policy':
            return graph_config.policy.rendered_text
        case _:
            assert_never(graph_config.policy_context)


async def run_judgement_checks(state: ReviewState, graph_config: GraphConfig) -> ReviewState:
    requester = state['requester']
    if requester is None:
        raise ValueError('Judgement checks need the requester that intake derived')
    submission = SUBMISSION_ADAPTER.validate_python(state['submission'])
    available_precedents = await current_precedents(graph_config, submission, requester)
    judged = await asyncio.to_thread(
        _judged, state, graph_config, submission, requester, available_precedents
    )
    return judged | {'current_precedents': available_precedents}


def _judged(
    state: ReviewState,
    graph_config: GraphConfig,
    submission: Submission,
    requester: str,
    available_precedents: list[SuppliedPrecedent],
) -> ReviewState:
    model_settings = graph_config.settings.chat_models[graph_config.settings.primary_model_key]
    checks = judgement_checks(submission, graph_config.seed_data, graph_config.policy)
    messages: list[BaseMessage] = [
        SystemMessage(
            content=judgement_instructions(
                graph_config.policy,
                checks,
                state['check_guidance'],
                _policy_text(state, graph_config),
            )
        ),
        HumanMessage(content=untrusted_data(prompt_fields(state['redacted_submission']))),
    ]
    bound = submission_tools(submission, requester, graph_config.seed_data, available_precedents)
    tools_by_name = {tool.name: tool for tool in bound.tools}
    model_with_tools = graph_config.chat_model.bind_tools(bound.tools)
    maximum_tool_calls = graph_config.settings.judgement_max_tool_calls
    tool_calls_made = 0
    audit: list[AuditEntry] = []
    while True:
        reply = model_with_tools.invoke(messages)
        messages.append(reply)
        if not isinstance(reply, AIMessage) or not reply.tool_calls:
            break
        for call in reply.tool_calls:
            if tool_calls_made >= maximum_tool_calls:
                messages.append(
                    ToolMessage(content=TOOL_CALL_LIMIT_REACHED, tool_call_id=str(call['id']))
                )
            elif call['name'] in tools_by_name:
                messages.append(tools_by_name[call['name']].invoke(call))
                tool_calls_made += 1
            else:
                messages.append(
                    ToolMessage(
                        content=json.dumps({'error': f'No tool named {call["name"]}.'}),
                        tool_call_id=str(call['id']),
                    )
                )
                tool_calls_made += 1
        if tool_calls_made >= maximum_tool_calls:
            audit.append(
                {
                    'node': 'run_judgement_checks',
                    'event': 'tool_call_limit_reached',
                    'detail': {'tool_calls': tool_calls_made},
                }
            )
            break
    messages.append(HumanMessage(content=VERDICT_REQUEST))
    output = structured_reply(
        graph_config.chat_model,
        model_settings.structured_output_method,
        JUDGEMENT_REPORT_SCHEMA,
        messages,
    )
    judgement_report = JudgementReport.model_validate(output)
    return {
        'judgement_checks': checks,
        'supplied_precedents': bound.supplied_precedents,
        'judgement_outcomes': [_outcome(verdict) for verdict in judgement_report.verdicts],
        'audit': audit,
    }
