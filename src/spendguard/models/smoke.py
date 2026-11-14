import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from spendguard import settings as settings_module
from spendguard.models.factory import EMBEDDING_DIMENSIONS, build_chat_model, build_embedder
from spendguard.settings import ChatModelSettings, Settings

REASONING_BLOCK_TYPES = frozenset({'reasoning', 'reasoning_content', 'thinking'})


class ClauseCitation(BaseModel):
    clause_reference: str
    severity: str


@tool
def look_up_clause(clause_reference: str) -> str:
    """Return the text of a policy clause, given its reference such as 3.4."""
    return f'Clause {clause_reference}: meals over the limit need a manager approval.'


@dataclass(frozen=True)
class SmokeResult:
    check: str
    model_key: str
    model_id: str
    passed: bool
    detail: str


def _reasoning_block_count(message: AIMessage) -> int:
    if isinstance(message.content, str):
        return 0
    return sum(
        1
        for block in message.content
        if isinstance(block, dict) and block.get('type') in REASONING_BLOCK_TYPES
    )


def check_structured_output(model: BaseChatModel, model_settings: ChatModelSettings) -> str:
    structured_model = model.with_structured_output(
        ClauseCitation, include_raw=True, method=model_settings.structured_output_method
    )
    result = structured_model.invoke(
        'A policy clause with reference 3.4 carries severity medium. '
        'Report its clause reference and severity.'
    )
    if not isinstance(result, dict):
        raise TypeError(f'Expected a dict with include_raw, got {type(result).__name__}')
    if result['parsing_error'] is not None:
        raise result['parsing_error']
    parsed = result['parsed']
    if not isinstance(parsed, ClauseCitation):
        raise TypeError(f'Expected ClauseCitation, got {type(parsed).__name__}')
    raw = result['raw']
    reasoning_blocks = _reasoning_block_count(raw) if isinstance(raw, AIMessage) else 0
    return (
        f'parsed {parsed.clause_reference}/{parsed.severity}, reasoning blocks {reasoning_blocks}'
    )


def check_tool_call(model: BaseChatModel, model_settings: ChatModelSettings) -> str:
    response = model.bind_tools([look_up_clause]).invoke(
        'Use the look_up_clause tool to fetch policy clause 3.4. Do not answer from memory.'
    )
    if not isinstance(response, AIMessage):
        raise TypeError(f'Expected an AIMessage, got {type(response).__name__}')
    tool_calls = [call for call in response.tool_calls if call['name'] == look_up_clause.name]
    if not tool_calls:
        raise AssertionError(f'No {look_up_clause.name} call in the response')
    look_up_clause.invoke(tool_calls[0]['args'])
    return (
        f'called with {tool_calls[0]["args"]}, reasoning blocks {_reasoning_block_count(response)}'
    )


def _run_check(check: str, model_key: str, model_id: str, run: Callable[[], str]) -> SmokeResult:
    try:
        detail = run()
    except Exception as error:
        first_line = str(error).splitlines()[0] if str(error) else ''
        return SmokeResult(
            check, model_key, model_id, passed=False, detail=f'{type(error).__name__}: {first_line}'
        )
    return SmokeResult(check, model_key, model_id, passed=True, detail=detail)


def _check_embedding(settings: Settings) -> str:
    vector = build_embedder(settings).embed_query('Meals over the limit need a manager approval.')
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise AssertionError(f'Expected {EMBEDDING_DIMENSIONS} dimensions, got {len(vector)}')
    return f'{len(vector)} dimensions'


def _build_and_check(
    check: Callable[[BaseChatModel, ChatModelSettings], str], settings: Settings, model_key: str
) -> str:
    return check(build_chat_model(settings, model_key), settings.chat_models[model_key])


def run_smoke_checks(settings: Settings) -> list[SmokeResult]:
    chat_checks = (
        ('structured output', check_structured_output),
        ('tool call', check_tool_call),
    )
    results = [
        _run_check(
            check_name,
            model_key,
            model_settings.model_id,
            partial(_build_and_check, check, settings, model_key),
        )
        for model_key, model_settings in settings.chat_models.items()
        for check_name, check in chat_checks
    ]
    results.append(
        _run_check(
            'embedding',
            'embedding_model',
            settings.embedding_model.model_id,
            partial(_check_embedding, settings),
        )
    )
    return results


def format_results(results: list[SmokeResult]) -> str:
    rows = [('result', 'check', 'model key', 'model id', 'detail')] + [
        (
            'pass' if result.passed else 'FAIL',
            result.check,
            result.model_key,
            result.model_id,
            result.detail,
        )
        for result in results
    ]
    widths = [max(len(row[column]) for row in rows) for column in range(4)]
    return '\n'.join(
        '  '.join(cell.ljust(width) for cell, width in zip(row[:4], widths, strict=True))
        + '  '
        + row[4]
        for row in rows
    )


def main() -> None:
    results = run_smoke_checks(settings_module.load_settings())
    print(format_results(results))
    sys.exit(0 if all(result.passed for result in results) else 1)


if __name__ == '__main__':
    main()
