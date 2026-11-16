import datetime
from decimal import Decimal
from pathlib import Path

from spendguard.eval.metrics import Proportion, RowMetrics, RuleMetrics
from spendguard.eval.report import EvalResults, RowResult, write_results
from spendguard.settings import ChatModelSettings, EvalRow, ModelPrice

README_BEFORE = """# A project

## Eval results

Fixed wording stays.

<!-- eval-results:start -->
No results yet.
<!-- eval-results:end -->

## After the results

Kept as it was.
"""


def metrics(precision: Proportion, cost: Decimal | None) -> RowMetrics:
    return RowMetrics(
        judgement_precision=precision,
        judgement_recall=Proportion(1, 3),
        per_code={'ALCOHOL_NOT_PERMITTED': Proportion(13, 16), 'PURPOSE_UNCLEAR': Proportion(0, 4)},
        route_accuracy=Proportion(3, 5),
        recommendation_accuracy=Proportion(2, 5),
        verify_drop_rate=Proportion(2, 6),
        injection_survival=Proportion(0, 1),
        retrieval_recall=Proportion(2, 3),
        p50_latency_seconds=3.0,
        cost_per_submission_usd=cost,
    )


RESULTS = EvalResults(
    measured_on=datetime.date(2026, 11, 20),
    dataset_hash='226df92315047242c93d4e57d559fceae981081fc1f82ac12c91309c886b2d0d',
    case_count=5,
    retrieval_top_k=5,
    judgement_max_tool_calls=6,
    rows=(
        RowResult(
            row=EvalRow(model_key='claude_haiku', label='Claude Haiku (primary)'),
            model=ChatModelSettings(
                provider='bedrock',
                model_id='us.anthropic.claude-haiku-4-5-20251001-v1:0',
                structured_output_method='function_calling',
                temperature=0.0,
            ),
            price=ModelPrice(
                input_per_million_tokens_usd=Decimal('1.00'),
                output_per_million_tokens_usd=Decimal('5.00'),
                priced_on=datetime.date(2026, 11, 2),
            ),
            metrics=metrics(Proportion(1, 4), Decimal('0.0016')),
        ),
        RowResult(
            row=EvalRow(model_key='gemini_pro', label='Gemini Pro'),
            model=ChatModelSettings(
                provider='google_genai',
                model_id='gemini-3.1-pro-preview',
                structured_output_method='json_schema',
                temperature=None,
            ),
            price=None,
            metrics=metrics(Proportion(3, 4), None),
        ),
    ),
    rule_metrics=RuleMetrics(
        precision=Proportion(7, 8),
        recall=Proportion(7, 7),
        per_code={'MISSING_PO': Proportion(2, 2)},
    ),
)


def table_row(readme: str, label: str) -> str:
    return next(line for line in readme.splitlines() if line.startswith(f'| {label} |'))


def test_the_readme_table_gets_one_row_per_model_row_between_the_markers(tmp_path: Path) -> None:
    readme_path = tmp_path / 'README.md'
    readme_path.write_text(README_BEFORE)

    write_results(RESULTS, tmp_path / 'results', readme_path)

    readme = readme_path.read_text()
    haiku_row = table_row(readme, 'Claude Haiku (primary)')
    assert '`us.anthropic.claude-haiku-4-5-20251001-v1:0`' in haiku_row
    assert '1/4 = 0.25 (0.05 to 0.70)' in haiku_row
    assert '1/3 = 0.33 (0.06 to 0.79)' in haiku_row
    assert '$0.0016' in haiku_row
    gemini_row = table_row(readme, 'Gemini Pro')
    assert '3/4 = 0.75 (0.30 to 0.95)' in gemini_row
    assert 'not priced' in gemini_row
    assert 'No results yet.' not in readme
    assert readme.startswith('# A project\n\n## Eval results\n\nFixed wording stays.\n\n')
    assert readme.endswith('## After the results\n\nKept as it was.\n')


def test_rule_check_metrics_appear_once_under_the_table_not_in_any_row(tmp_path: Path) -> None:
    readme_path = tmp_path / 'README.md'
    readme_path.write_text(README_BEFORE)

    write_results(RESULTS, tmp_path / 'results', readme_path)

    readme = readme_path.read_text()
    assert readme.count('7/8 = 0.88') == 1
    assert readme.count('7/7 = 1.00') == 1
    assert '7/8' not in table_row(readme, 'Claude Haiku (primary)')
    assert 'Rule checks, the same for every row: precision 7/8 = 0.88, recall 7/7 = 1.00' in readme
    assert 'Measured on 2026-11-20 against eval-v1 (226df923); test cases: 5.' in readme


def test_the_dated_results_file_states_the_wording_parameters_and_counts(tmp_path: Path) -> None:
    readme_path = tmp_path / 'README.md'
    readme_path.write_text(README_BEFORE)

    results_path = write_results(RESULTS, tmp_path / 'results', readme_path)

    assert results_path == tmp_path / 'results' / 'eval-v1-2026-11-20.md'
    text = results_path.read_text()
    assert 'Overlapping 95% intervals are never claimed as a win.' in text
    assert (
        'Haiku unless the eval shows it worse. With 16 positives the eval can only detect large'
        ' gaps, so this is a cost-led default, not a demonstrated equivalence.'
    ) in text
    assert (
        '| Claude Haiku (primary) | bedrock | `us.anthropic.claude-haiku-4-5-20251001-v1:0`'
        ' | 0.0 | function_calling | $1.00 in, $5.00 out per million tokens, priced on 2026-11-02 |'
    ) in text
    assert (
        '| Gemini Pro | google_genai | `gemini-3.1-pro-preview` | provider default | json_schema'
        ' | not priced |'
    ) in text
    assert 'Retrieval top k: 5. Judgement tool call limit: 6. One run per row.' in text
    assert text.count('| ALCOHOL_NOT_PERMITTED | 13/16 |') == 1
    assert text.count('| ALCOHOL_NOT_PERMITTED | 13/16 | 13/16 |') == 1
    assert text.count('| MISSING_PO | 2/2 |') == 1
    assert text.count('7/8 = 0.88') == 1
