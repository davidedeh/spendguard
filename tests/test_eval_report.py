import dataclasses
import datetime
import json
from decimal import Decimal
from pathlib import Path

import pytest

from spendguard.eval.cases import EvalCase, read_case_directory
from spendguard.eval.metrics import (
    CaseFailure,
    CaseOutcome,
    MemoryMetrics,
    Proportion,
    RowMetrics,
    RuleMetrics,
    judgement_positive_count,
)
from spendguard.eval.report import (
    OVERLAP_WORDING,
    RESULTS_END_MARKER,
    RESULTS_START_MARKER,
    EvalResults,
    MemoryComparison,
    RowResult,
    RowsDirectory,
    production_default_wording,
    write_results,
    write_row_results,
)
from spendguard.eval.runner import REPORTED_SPLIT
from spendguard.policy.loader import POLICY_PATH, load_policy
from spendguard.settings import ChatModelSettings, EvalRow, ModelPrice

REPOSITORY_ROOT = Path(__file__).parent.parent
README_BEFORE = """# A project

## Eval results

Fixed wording stays.

<!-- eval-results:start -->
No results yet.
<!-- eval-results:end -->

## After the results

Kept as it was.
"""


def metrics(
    precision: Proportion, cost: Decimal | None, failed_cases: tuple[CaseFailure, ...] = ()
) -> RowMetrics:
    return RowMetrics(
        judgement_precision=precision,
        judgement_recall=Proportion(1, 3),
        per_code={'ALCOHOL_NOT_PERMITTED': Proportion(13, 16), 'PURPOSE_UNCLEAR': Proportion(0, 4)},
        route_accuracy=Proportion(3, 5),
        recommendation_accuracy=Proportion(2, 5),
        verify_drop_rate=Proportion(2, 6),
        injection_survival=Proportion(0, 1),
        retrieval_recall=Proportion(2, 3),
        p50_latency_seconds=None if cost is None else 3.0,
        cost_per_submission_usd=cost,
        failed_cases=failed_cases,
    )


def failure(case_id: str, reason: str) -> CaseFailure:
    return CaseFailure(
        case_id=case_id, error_type='ValueError', error_message='unused', reason=reason
    )


RESULTS = EvalResults(
    measured_on=datetime.date(2026, 11, 20),
    dataset_hash='226df92315047242c93d4e57d559fceae981081fc1f82ac12c91309c886b2d0d',
    run_identifier='5c0ffee1d2e3f4a5b6c7d8e9f0a1b2c3',
    case_count=5,
    judgement_positive_count=13,
    cases_per_row=None,
    retrieval_top_k=4,
    judgement_max_tool_calls=6,
    rows=(
        RowResult(
            row=EvalRow(
                model_key='claude_haiku',
                label='Claude Haiku (primary)',
                policy_context='retrieved_clauses',
            ),
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
            row=EvalRow(
                model_key='gemini_pro', label='Gemini Pro', policy_context='retrieved_clauses'
            ),
            model=ChatModelSettings(
                provider='google_genai',
                model_id='gemini-3.1-pro-preview',
                structured_output_method='json_schema',
                temperature=None,
            ),
            price=ModelPrice(
                input_per_million_tokens_usd=Decimal('1.25'),
                output_per_million_tokens_usd=Decimal('10.00'),
                priced_on=datetime.date(2026, 11, 3),
            ),
            metrics=metrics(
                Proportion(3, 4),
                None,
                failed_cases=(
                    failure('test-clean', 'malformed reply: max_tokens'),
                    failure('test-alcohol', 'ThrottlingException'),
                ),
            ),
        ),
    ),
    rule_metrics=RuleMetrics(
        precision=Proportion(7, 8),
        recall=Proportion(7, 7),
        per_code={'MISSING_PO': Proportion(2, 2)},
    ),
    memory_comparison=None,
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
    assert gemini_row.endswith(' | not measured | not measured | 2 failed |')
    assert haiku_row.endswith(' | 3.0 s | $0.0016 | 0 failed |')
    assert '| Retrieval recall (top 4 per query) |' in readme
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
    assert 'Measured on 2026-11-20 against eval-v2 (226df923); test cases: 5.' in readme


def test_the_dated_results_file_states_the_wording_parameters_and_counts(tmp_path: Path) -> None:
    readme_path = tmp_path / 'README.md'
    readme_path.write_text(README_BEFORE)

    results_path = write_results(RESULTS, tmp_path / 'results', readme_path)

    assert results_path == tmp_path / 'results' / 'eval-v2-2026-11-20-5c0ffee1.md'
    text = results_path.read_text()
    assert 'Overlapping 95% intervals are never claimed as a win.' in text
    assert (
        'Production default: Haiku unless the eval shows it worse. With 13 positives the eval can'
        ' only detect large gaps, so this is a cost-led default, not a demonstrated equivalence.'
    ) in text
    assert (
        '| Claude Haiku (primary) | bedrock | `us.anthropic.claude-haiku-4-5-20251001-v1:0`'
        ' | 0.0 | function_calling | $1.00 in, $5.00 out per million tokens, priced on 2026-11-02 |'
    ) in text
    assert (
        '| Gemini Pro | google_genai | `gemini-3.1-pro-preview` | provider default | json_schema'
        ' | $1.25 in, $10.00 out per million tokens, priced on 2026-11-03 |'
    ) in text
    assert 'Retrieval top k: 4. Judgement tool call limit: 6. One run per row.' in text
    assert text.count('| ALCOHOL_NOT_PERMITTED | 13/16 |') == 1
    assert text.count('| ALCOHOL_NOT_PERMITTED | 13/16 | 13/16 |') == 1
    assert text.count('| MISSING_PO | 2/2 |') == 1
    assert text.count('7/8 = 0.88') == 1


def test_the_production_default_wording_counts_a_single_positive_in_the_singular() -> None:
    assert 'With 1 positive the eval' in production_default_wording(1)


def test_the_project_readme_states_the_fixed_wording_around_the_results_table() -> None:
    readme = (REPOSITORY_ROOT / 'README.md').read_text()
    reported_cases = [
        case
        for case in (
            EvalCase.model_validate(content)
            for content in read_case_directory(REPOSITORY_ROOT / 'evals' / 'cases').values()
        )
        if case.split == REPORTED_SPLIT
    ]

    assert OVERLAP_WORDING in readme
    assert (
        production_default_wording(
            judgement_positive_count(reported_cases, load_policy(POLICY_PATH))
        )
        in readme
    )
    assert "GPT OSS 120B is OpenAI's open-weight model, not frontier GPT." in readme
    assert readme.count(RESULTS_START_MARKER) == 1
    assert readme.count(RESULTS_END_MARKER) == 1


def test_the_results_file_reports_memory_on_and_off_for_the_primary_model(tmp_path: Path) -> None:
    haiku = RESULTS.rows[0]
    results = dataclasses.replace(
        RESULTS,
        memory_comparison=MemoryComparison(
            row=haiku.row,
            model=haiku.model,
            memory_on=MemoryMetrics(
                raised_labelled_code=Proportion(0, 3),
                raised_any_judgement_code=Proportion(1, 3),
                cleared_with_pair_precedent=Proportion(1, 3),
                forced_review=Proportion(1, 3),
                failed=Proportion(0, 3),
                silent_clearances=Proportion(0, 3),
                controls_raised_labelled_code=Proportion(1, 3),
                controls_cleared_by_precedent=Proportion(2, 3),
                controls_failed=Proportion(0, 3),
                recommendation_agreement=Proportion(5, 6),
                other_findings=(('test-memory-follow-up-01', ('PURPOSE_UNCLEAR',)),),
            ),
            memory_off=MemoryMetrics(
                raised_labelled_code=Proportion(2, 3),
                raised_any_judgement_code=Proportion(2, 3),
                cleared_with_pair_precedent=Proportion(0, 3),
                forced_review=Proportion(0, 3),
                failed=Proportion(1, 3),
                silent_clearances=Proportion(0, 3),
                controls_raised_labelled_code=Proportion(2, 3),
                controls_cleared_by_precedent=Proportion(0, 3),
                controls_failed=Proportion(1, 3),
                recommendation_agreement=Proportion(1, 6),
                other_findings=(),
            ),
        ),
    )
    readme_path = tmp_path / 'README.md'
    readme_path.write_text(README_BEFORE)

    text = write_results(results, tmp_path / 'results', readme_path).read_text()

    assert '## Memory on versus off' in text
    assert (
        'Claude Haiku (primary), `us.anthropic.claude-haiku-4-5-20251001-v1:0`. Each follow-up'
        " gets its pair's scripted reason as a precedent. Follow-ups whose precedent applies: 3;"
        ' controls, whose precedent does not apply: 3. With memory off no precedent is available,'
        ' so raising the labelled code is the right answer on every follow-up.'
    ) in text
    assert (
        '| Setting | Raised the labelled code | Raised any judgement code'
        " | Cleared with the pair's precedent | Forced review | Failed | Silent clearances |"
        '\n| --- | --- | --- | --- | --- | --- | --- |'
        '\n| Memory on | 0/3 = 0.00 | 1/3 = 0.33 | 1/3 = 0.33 | 1/3 = 0.33 | 0/3 = 0.00'
        ' | 0/3 = 0.00 |'
        '\n| Memory off | 2/3 = 0.67 | 2/3 = 0.67 | 0/3 = 0.00 | 0/3 = 0.00 | 1/3 = 0.33'
        ' | 0/3 = 0.00 |'
    ) in text
    assert (
        '| Setting | Controls that raised the labelled code | Controls cleared by the precedent'
        ' | Controls that failed |'
        '\n| --- | --- | --- | --- |'
        '\n| Memory on | 1/3 = 0.33 | 2/3 = 0.67 | 0/3 = 0.00 |'
        '\n| Memory off | 2/3 = 0.67 | 0/3 = 0.00 | 1/3 = 0.33 |'
    ) in text
    assert (
        'Recommendation matches the label with memory on: 5/6 = 0.83. The labels assume the'
        ' precedent, so memory off is not scored on recommendations.'
    ) in text
    assert '1/6 = 0.17' not in text
    assert (
        'Findings on other codes. Memory on: test-memory-follow-up-01 (PURPOSE_UNCLEAR).'
        ' Memory off: none.'
    ) in text
    assert 'Memory on versus off' not in readme_path.read_text()


def test_the_results_file_lists_each_row_s_failed_cases_with_their_reasons(
    tmp_path: Path,
) -> None:
    readme_path = tmp_path / 'README.md'
    readme_path.write_text(README_BEFORE)

    results_path = write_results(RESULTS, tmp_path / 'results', readme_path)

    section = results_path.read_text().split('## Failed cases\n\n')[1].split('\n\n')[0]
    assert section.splitlines() == [
        '| Row | Case | Reason |',
        '| --- | --- | --- |',
        '| Gemini Pro | test-clean | malformed reply: max_tokens |',
        '| Gemini Pro | test-alcohol | ThrottlingException |',
    ]


def test_the_results_file_says_when_no_case_failed(tmp_path: Path) -> None:
    rows_without_failures = tuple(
        dataclasses.replace(result, metrics=dataclasses.replace(result.metrics, failed_cases=()))
        for result in RESULTS.rows
    )
    results = dataclasses.replace(RESULTS, rows=rows_without_failures)
    readme_path = tmp_path / 'README.md'
    readme_path.write_text(README_BEFORE)

    results_path = write_results(results, tmp_path / 'results', readme_path)

    assert '## Failed cases\n\nNo failed cases.\n' in results_path.read_text()


def test_the_results_file_says_when_memory_was_not_compared(tmp_path: Path) -> None:
    readme_path = tmp_path / 'README.md'
    readme_path.write_text(README_BEFORE)

    text = write_results(RESULTS, tmp_path / 'results', readme_path).read_text()

    assert (
        '## Memory on versus off\n\nNot compared: no eval row runs the primary model with'
        ' retrieved clauses.'
    ) in text


def test_the_whole_policy_row_shows_retrieval_recall_as_not_applicable(tmp_path: Path) -> None:
    haiku = RESULTS.rows[0]
    whole_policy = RowResult(
        row=EvalRow(
            model_key='claude_haiku',
            label='Claude Haiku (whole policy)',
            policy_context='whole_policy',
        ),
        model=haiku.model,
        price=haiku.price,
        metrics=haiku.metrics,
    )
    results = dataclasses.replace(RESULTS, rows=(haiku, whole_policy))
    readme_path = tmp_path / 'README.md'
    readme_path.write_text(README_BEFORE)

    write_results(results, tmp_path / 'results', readme_path)

    readme = readme_path.read_text()
    whole_policy_cells = table_row(readme, 'Claude Haiku (whole policy)').split(' | ')
    haiku_cells = table_row(readme, 'Claude Haiku (primary)').split(' | ')
    assert haiku_cells[8] == '2/3 = 0.67'
    assert whole_policy_cells[8] == 'not applicable'
    assert whole_policy_cells[:8] == ['| Claude Haiku (whole policy)', *haiku_cells[1:8]]


def test_a_pre_flight_writes_its_own_results_file_and_leaves_the_readme_alone(
    tmp_path: Path,
) -> None:
    readme_path = tmp_path / 'README.md'
    readme_path.write_text(README_BEFORE)

    results_path = write_results(
        dataclasses.replace(RESULTS, cases_per_row=2, case_count=6),
        tmp_path / 'results',
        readme_path,
    )

    assert results_path == tmp_path / 'results' / 'eval-v2-2026-11-20-5c0ffee1-preflight.md'
    assert (
        'Pre-flight: a sample of 6 cases per row, at least one from every slice; not reported.'
        in results_path.read_text()
    )
    assert readme_path.read_text() == README_BEFORE


def test_writing_results_never_replaces_an_existing_results_file_or_the_readme(
    tmp_path: Path,
) -> None:
    readme_path = tmp_path / 'README.md'
    readme_path.write_text(README_BEFORE)
    existing_path = tmp_path / 'results' / 'eval-v2-2026-11-20-5c0ffee1.md'
    existing_path.parent.mkdir()
    existing_path.write_text('An earlier run.\n')

    with pytest.raises(FileExistsError):
        write_results(RESULTS, tmp_path / 'results', readme_path)

    assert existing_path.read_text() == 'An earlier run.\n'
    assert readme_path.read_text() == README_BEFORE


def test_a_finished_row_is_saved_with_its_case_results_and_token_totals(tmp_path: Path) -> None:
    gemini = RESULTS.rows[1]
    completed = CaseOutcome(
        case_id='test-alcohol',
        rule_codes=frozenset({'MISSING_RECEIPT'}),
        judgement_codes=frozenset({'PURPOSE_UNCLEAR', 'ALCOHOL_NOT_PERMITTED'}),
        route='human_review',
        recommendation='none',
        proposed_judgement_results=2,
        dropped_judgement_results=0,
        retrieved_clause_references=frozenset({'§2.2'}),
        clearances=frozenset(),
        unanswered_judgement_codes=frozenset({'WEEKEND_SPEND_UNJUSTIFIED'}),
        latency_seconds=1.5,
        input_tokens=1200,
        output_tokens=80,
    )
    another = dataclasses.replace(
        completed, case_id='test-weekend', input_tokens=300, output_tokens=7
    )
    failure = CaseFailure(
        case_id='test-injection',
        error_type='RuntimeError',
        error_message='provider down',
        reason='RuntimeError',
    )

    path = write_row_results(
        RowsDirectory(tmp_path / 'eval-v2-2026-11-20-5c0ffee1', RESULTS.run_identifier),
        2,
        gemini.row,
        gemini.model,
        [completed, failure, another],
    )

    assert path == tmp_path / 'eval-v2-2026-11-20-5c0ffee1' / 'row-2.json'
    saved = json.loads(path.read_text())
    assert saved['run_identifier'] == '5c0ffee1d2e3f4a5b6c7d8e9f0a1b2c3'
    assert saved['row'] == {
        'label': 'Gemini Pro',
        'model_key': 'gemini_pro',
        'model_id': 'gemini-3.1-pro-preview',
        'policy_context': 'retrieved_clauses',
    }
    assert (saved['input_tokens'], saved['output_tokens']) == (1500, 87)
    assert saved['failed_case_count'] == 1
    assert saved['cases'][1] == {
        'case_id': 'test-injection',
        'failed': True,
        'error_type': 'RuntimeError',
        'error_message': 'provider down',
        'reason': 'RuntimeError',
    }
    assert saved['cases'][0]['judgement_codes'] == ['ALCOHOL_NOT_PERMITTED', 'PURPOSE_UNCLEAR']
    assert saved['cases'][0]['failed'] is False
    assert saved['cases'][0]['latency_seconds'] == 1.5
    assert saved['cases'][2]['case_id'] == 'test-weekend'
