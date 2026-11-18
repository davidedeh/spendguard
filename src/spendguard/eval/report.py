import dataclasses
import datetime
import json
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import assert_never

from spendguard.eval.metrics import (
    CaseOutcome,
    CaseResult,
    MemoryMetrics,
    Proportion,
    RowMetrics,
    RuleMetrics,
    completed_outcomes,
)
from spendguard.settings import ChatModelSettings, EvalRow, ModelPrice
from spendguard.wording import counted

RESULTS_START_MARKER = '<!-- eval-results:start -->'
RESULTS_END_MARKER = '<!-- eval-results:end -->'
DATASET_NAME = 'eval-v1'
DATASET_HASH_PREFIX_LENGTH = 8
RUN_IDENTIFIER_PREFIX_LENGTH = 8
OVERLAP_WORDING = 'Overlapping 95% intervals are never claimed as a win.'
NOT_MEASURED = 'not measured'


def production_default_wording(judgement_positive_count: int) -> str:
    return (
        'Production default: Haiku unless the eval shows it worse. With'
        f' {counted(judgement_positive_count, "positive")} the eval can only detect large gaps,'
        ' so this is a cost-led default, not a demonstrated equivalence.'
    )


@dataclass(frozen=True)
class RowResult:
    row: EvalRow
    model: ChatModelSettings
    price: ModelPrice
    metrics: RowMetrics


@dataclass(frozen=True)
class MemoryComparison:
    row: EvalRow
    model: ChatModelSettings
    memory_on: MemoryMetrics
    memory_off: MemoryMetrics


@dataclass(frozen=True)
class EvalResults:
    measured_on: datetime.date
    dataset_hash: str
    run_identifier: str
    case_count: int
    judgement_positive_count: int
    cases_per_row: int | None
    retrieval_top_k: int
    judgement_max_tool_calls: int
    rows: tuple[RowResult, ...]
    rule_metrics: RuleMetrics
    memory_comparison: MemoryComparison | None


def _rate(proportion: Proportion) -> str:
    if proportion.trials == 0:
        return '0/0'
    rate = proportion.successes / proportion.trials
    return f'{proportion.successes}/{proportion.trials} = {rate:.2f}'


def _rate_with_interval(proportion: Proportion) -> str:
    interval = proportion.interval
    if interval is None:
        return _rate(proportion)
    return f'{_rate(proportion)} ({interval.low:.2f} to {interval.high:.2f})'


def _cost(cost: Decimal | None) -> str:
    return NOT_MEASURED if cost is None else f'${cost:.4f}'


def _latency(p50_latency_seconds: float | None) -> str:
    return NOT_MEASURED if p50_latency_seconds is None else f'{p50_latency_seconds:.1f} s'


def _retrieval_recall_cell(result: RowResult) -> str:
    match result.row.policy_context:
        case 'retrieved_clauses':
            return _rate(result.metrics.retrieval_recall)
        case 'whole_policy':
            return 'not applicable'
        case _:
            assert_never(result.row.policy_context)


def _table_row(result: RowResult) -> str:
    metrics = result.metrics
    cells = (
        result.row.label,
        f'`{result.model.model_id}`',
        _rate_with_interval(metrics.judgement_precision),
        _rate_with_interval(metrics.judgement_recall),
        _rate(metrics.route_accuracy),
        _rate(metrics.recommendation_accuracy),
        _rate(metrics.verify_drop_rate),
        _rate(metrics.injection_survival),
        _retrieval_recall_cell(result),
        _latency(metrics.p50_latency_seconds),
        _cost(metrics.cost_per_submission_usd),
        f'{metrics.failed_case_count} failed',
    )
    return '| ' + ' | '.join(cells) + ' |'


def readme_table(results: EvalResults) -> str:
    header = (
        '| Row | Model ID | Judgement precision (95% CI) | Judgement recall (95% CI)'
        ' | Route accuracy | Recommendation accuracy | Verify drop rate | Injection survival'
        f' | Retrieval recall (top {results.retrieval_top_k} per query) | p50 latency'
        ' | Cost per submission | Failed cases |'
    )
    separator = '| ' + ' | '.join(['---'] * 12) + ' |'
    rule_metrics = results.rule_metrics
    return '\n'.join(
        [
            _measured_on(results),
            '',
            header,
            separator,
            *(_table_row(result) for result in results.rows),
            '',
            f'Rule checks, the same for every row: precision {_rate(rule_metrics.precision)},'
            f' recall {_rate(rule_metrics.recall)}.',
        ]
    )


def _measured_on(results: EvalResults) -> str:
    return (
        f'Measured on {results.measured_on.isoformat()} against {DATASET_NAME}'
        f' ({results.dataset_hash[:DATASET_HASH_PREFIX_LENGTH]}); test cases:'
        f' {results.case_count}.'
    )


def _replace_results_block(readme: str, block: str) -> str:
    before, _, rest = readme.partition(RESULTS_START_MARKER)
    _, _, after = rest.partition(RESULTS_END_MARKER)
    return f'{before}{RESULTS_START_MARKER}\n{block}\n{RESULTS_END_MARKER}{after}'


def _temperature(model: ChatModelSettings) -> str:
    return 'provider default' if model.temperature is None else str(model.temperature)


def _price(price: ModelPrice) -> str:
    return (
        f'${price.input_per_million_tokens_usd} in, ${price.output_per_million_tokens_usd} out'
        f' per million tokens, priced on {price.priced_on.isoformat()}'
    )


def _parameters_table(results: EvalResults) -> list[str]:
    return [
        '| Row | Provider | Model ID | Temperature | Structured output | Price |',
        '| --- | --- | --- | --- | --- | --- |',
        *(
            f'| {result.row.label} | {result.model.provider} | `{result.model.model_id}`'
            f' | {_temperature(result.model)} | {result.model.structured_output_method}'
            f' | {_price(result.price)} |'
            for result in results.rows
        ),
        '',
        f'Retrieval top k: {results.retrieval_top_k}.'
        f' Judgement tool call limit: {results.judgement_max_tool_calls}. One run per row.',
    ]


def _per_code_table(results: EvalResults) -> list[str]:
    codes = list(results.rows[0].metrics.per_code) if results.rows else []
    return [
        '| Judgement code | ' + ' | '.join(result.row.label for result in results.rows) + ' |',
        '| ' + ' | '.join(['---'] * (len(results.rows) + 1)) + ' |',
        *(
            f'| {code} | '
            + ' | '.join(
                f'{result.metrics.per_code[code].successes}/{result.metrics.per_code[code].trials}'
                for result in results.rows
            )
            + ' |'
            for code in codes
        ),
    ]


def _rule_code_table(rule_metrics: RuleMetrics) -> list[str]:
    return [
        '| Rule code | Found / expected |',
        '| --- | --- |',
        *(
            f'| {code} | {count.successes}/{count.trials} |'
            for code, count in rule_metrics.per_code.items()
        ),
    ]


def _pre_flight_note(cases_per_row: int | None) -> list[str]:
    if cases_per_row is None:
        return []
    return [f'Pre-flight: {counted(cases_per_row, "case")} per row, not reported.', '']


def results_file_text(results: EvalResults) -> str:
    return '\n'.join(
        [
            f'# {DATASET_NAME} results, {results.measured_on.isoformat()}',
            '',
            *_pre_flight_note(results.cases_per_row),
            OVERLAP_WORDING,
            '',
            production_default_wording(results.judgement_positive_count),
            '',
            '## Rows',
            '',
            *_parameters_table(results),
            '',
            '## Metrics',
            '',
            readme_table(results),
            '',
            '## Judgement findings found per code',
            '',
            *_per_code_table(results),
            '',
            '## Rule checks per code',
            '',
            *_rule_code_table(results.rule_metrics),
            '',
            '## Memory on versus off',
            '',
            *_memory_section(results.memory_comparison),
            '',
        ]
    )


def _memory_row(setting: str, metrics: MemoryMetrics) -> str:
    cells = (
        setting,
        _rate(metrics.false_positive_rate),
        _rate(metrics.recommendation_agreement),
        _rate(metrics.cleared_with_pair_precedent),
        _rate(metrics.forced_review),
        _rate(metrics.failed),
        _rate(metrics.silent_clearances),
    )
    return '| ' + ' | '.join(cells) + ' |'


def _memory_section(comparison: MemoryComparison | None) -> list[str]:
    if comparison is None:
        return ['Not compared: no eval row runs the primary model with retrieved clauses.']
    return [
        f'{comparison.row.label}, `{comparison.model.model_id}`, scored against labels written'
        f' for memory on. Memory follow-ups: {comparison.memory_on.false_positive_rate.trials}.',
        '',
        '| Setting | Judgement false-positive rate | Recommendation matches approver'
        " | Cleared with the pair's precedent | Forced review | Failed | Silent clearances |",
        '| ' + ' | '.join(['---'] * 7) + ' |',
        _memory_row('Memory on', comparison.memory_on),
        _memory_row('Memory off', comparison.memory_off),
    ]


def dated_results_name(
    measured_on: datetime.date, run_identifier: str, cases_per_row: int | None
) -> str:
    name = (
        f'{DATASET_NAME}-{measured_on.isoformat()}-{run_identifier[:RUN_IDENTIFIER_PREFIX_LENGTH]}'
    )
    return name if cases_per_row is None else f'{name}-preflight'


def _case_result_record(result: CaseResult) -> dict[str, object]:
    if not isinstance(result, CaseOutcome):
        return {
            'case_id': result.case_id,
            'failed': True,
            'error_type': result.error_type,
            'error_message': result.error_message,
        }
    fields = {field.name: getattr(result, field.name) for field in dataclasses.fields(result)}
    return {
        name: sorted(value) if isinstance(value, frozenset) else value
        for name, value in fields.items()
    } | {'failed': False}


@dataclass(frozen=True)
class RowsDirectory:
    path: Path
    run_identifier: str


def write_row_results(
    rows_directory: RowsDirectory,
    row_number: int,
    row: EvalRow,
    model: ChatModelSettings,
    results: Sequence[CaseResult],
) -> Path:
    outcomes = completed_outcomes(results)
    rows_directory.path.mkdir(parents=True, exist_ok=True)
    row_path = rows_directory.path / f'row-{row_number}.json'
    record = {
        'run_identifier': rows_directory.run_identifier,
        'row': {
            'label': row.label,
            'model_key': row.model_key,
            'model_id': model.model_id,
            'policy_context': row.policy_context,
        },
        'input_tokens': sum(outcome.input_tokens for outcome in outcomes),
        'output_tokens': sum(outcome.output_tokens for outcome in outcomes),
        'failed_case_count': len(results) - len(outcomes),
        'cases': [_case_result_record(result) for result in results],
    }
    row_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    return row_path


def write_results(results: EvalResults, results_directory: Path, readme_path: Path) -> Path:
    results_directory.mkdir(parents=True, exist_ok=True)
    results_name = dated_results_name(
        results.measured_on, results.run_identifier, results.cases_per_row
    )
    results_path = results_directory / f'{results_name}.md'
    results_text = results_file_text(results)
    with results_path.open('x') as results_file:
        results_file.write(results_text)
    if results.cases_per_row is None:
        readme_path.write_text(
            _replace_results_block(readme_path.read_text(), readme_table(results))
        )
    return results_path
