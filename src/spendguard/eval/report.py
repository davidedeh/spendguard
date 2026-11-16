import datetime
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import assert_never

from spendguard.eval.metrics import Proportion, RowMetrics, RuleMetrics
from spendguard.settings import ChatModelSettings, EvalRow, ModelPrice

RESULTS_START_MARKER = '<!-- eval-results:start -->'
RESULTS_END_MARKER = '<!-- eval-results:end -->'
DATASET_NAME = 'eval-v1'
OVERLAP_WORDING = 'Overlapping 95% intervals are never claimed as a win.'
PRODUCTION_DEFAULT_WORDING = (
    'Production default: Haiku unless the eval shows it worse. With 16 positives the eval can'
    ' only detect large gaps, so this is a cost-led default, not a demonstrated equivalence.'
)


@dataclass(frozen=True)
class RowResult:
    row: EvalRow
    model: ChatModelSettings
    price: ModelPrice | None
    metrics: RowMetrics


@dataclass(frozen=True)
class EvalResults:
    measured_on: datetime.date
    dataset_hash: str
    case_count: int
    retrieval_top_k: int
    judgement_max_tool_calls: int
    rows: tuple[RowResult, ...]
    rule_metrics: RuleMetrics


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
    return 'not priced' if cost is None else f'${cost:.4f}'


def _retrieval_recall(result: RowResult) -> str:
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
        _retrieval_recall(result),
        f'{metrics.p50_latency_seconds:.1f} s',
        _cost(metrics.cost_per_submission_usd),
    )
    return '| ' + ' | '.join(cells) + ' |'


def readme_table(results: EvalResults) -> str:
    header = (
        '| Row | Model ID | Judgement precision (95% CI) | Judgement recall (95% CI)'
        ' | Route accuracy | Recommendation accuracy | Verify drop rate | Injection survival'
        f' | Retrieval recall@{results.retrieval_top_k} | p50 latency | Cost per submission |'
    )
    separator = '| ' + ' | '.join(['---'] * 11) + ' |'
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
        f' ({results.dataset_hash[:8]}); test cases: {results.case_count}.'
    )


def _replace_results_block(readme: str, block: str) -> str:
    before, _, rest = readme.partition(RESULTS_START_MARKER)
    _, _, after = rest.partition(RESULTS_END_MARKER)
    return f'{before}{RESULTS_START_MARKER}\n{block}\n{RESULTS_END_MARKER}{after}'


def _temperature(model: ChatModelSettings) -> str:
    return 'provider default' if model.temperature is None else str(model.temperature)


def _price(price: ModelPrice | None) -> str:
    if price is None:
        return 'not priced'
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


def results_file_text(results: EvalResults) -> str:
    return '\n'.join(
        [
            f'# {DATASET_NAME} results, {results.measured_on.isoformat()}',
            '',
            OVERLAP_WORDING,
            '',
            PRODUCTION_DEFAULT_WORDING,
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
        ]
    )


def write_results(results: EvalResults, results_directory: Path, readme_path: Path) -> Path:
    results_directory.mkdir(parents=True, exist_ok=True)
    results_path = results_directory / f'{DATASET_NAME}-{results.measured_on.isoformat()}.md'
    results_path.write_text(results_file_text(results))
    readme_path.write_text(_replace_results_block(readme_path.read_text(), readme_table(results)))
    return results_path
