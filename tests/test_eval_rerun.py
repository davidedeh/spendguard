import asyncio
import datetime
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langfuse._client.resource_manager import LangfuseResourceManager
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import Field, SecretStr

import spendguard.eval.runner
import spendguard.settings
from eval_commands import (
    PRICE,
    UNPACED_REQUESTS_PER_MINUTE,
    eval_command_arguments,
    locked_fixture_dataset,
)
from fake_langfuse import FakeLangfuseApi, attribute, tracing_into
from settings_factory import make_test_settings
from spendguard.db.database import async_connection_pool, branch_connection_info, opened_pool
from spendguard.eval.rerun import main as rerun_command
from spendguard.eval.runner import main as run_eval_command
from spendguard.graph.config import thread_config
from spendguard.graph.nodes.plan import RetrievalPlan
from spendguard.graph.nodes.run_judgement_checks import JudgementReport
from spendguard.models.fake import FakeChatModel, structured_response
from spendguard.settings import ChatModelSettings, ModelPrice, Settings

WEEKEND_CASE_TEXT = 'Station to hotel'
CLEAN_INVOICE_TEXT = 'Quarterly printer'


class ServiceUnavailableException(Exception):  # noqa: N818
    pass


class RerunInterruptedError(BaseException):
    pass


class ThrottlingException(Exception):  # noqa: N818
    pass


class ReviewingModel(FakeChatModel):
    unavailable_for: list[str] = Field(default_factory=list)
    throttled_for: list[str] = Field(default_factory=list)
    broken_for: list[str] = Field(default_factory=list)
    interrupted_by: list[str] = Field(default_factory=list)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        if any(text in message.text for message in messages for text in self.interrupted_by):
            raise RerunInterruptedError
        if any(text in message.text for message in messages for text in self.unavailable_for):
            raise ServiceUnavailableException('Bedrock is unable to process your request.')
        if any(text in message.text for message in messages for text in self.throttled_for):
            raise ThrottlingException('Too many requests, please wait before trying again.')
        if any(text in message.text for message in messages for text in self.broken_for):
            raise RuntimeError('the reply ran to the output limit')
        bound_tool_names = [tool['function']['name'] for tool in kwargs.get('tools', [])]
        if bound_tool_names == [RetrievalPlan.__name__]:
            reply = structured_response(
                RetrievalPlan(retrieval_queries=['receipts', 'weekend spend'], check_guidance=[])
            )
        elif bound_tool_names == [JudgementReport.__name__]:
            reply = structured_response(JudgementReport(verdicts=[]))
        else:
            reply = AIMessage(content='Nothing to look up.')
        return ChatResult(generations=[ChatGeneration(message=reply)])


def one_row_settings() -> Settings:
    return make_test_settings(
        eval_rows=[
            {'model_key': 'fake_primary', 'label': 'Primary', 'policy_context': 'retrieved_clauses'}
        ],
        model_prices={'fake_primary': PRICE},
        eval_requests_per_minute={'fake_primary': UNPACED_REQUESTS_PER_MINUTE},
    )


def two_row_settings() -> Settings:
    return make_test_settings(
        chat_models=make_test_settings().chat_models
        | {
            'fake_second': ChatModelSettings(
                provider='fake',
                model_id='fake-second',
                structured_output_method='function_calling',
                temperature=0.0,
            )
        },
        eval_rows=[
            {
                'model_key': 'fake_primary',
                'label': 'Primary',
                'policy_context': 'retrieved_clauses',
            },
            {'model_key': 'fake_second', 'label': 'Second', 'policy_context': 'retrieved_clauses'},
        ],
        model_prices={'fake_primary': PRICE, 'fake_second': PRICE},
        eval_requests_per_minute={
            'fake_primary': UNPACED_REQUESTS_PER_MINUTE,
            'fake_second': UNPACED_REQUESTS_PER_MINUTE,
        },
    )


def use_model(monkeypatch: pytest.MonkeyPatch, model: FakeChatModel) -> None:
    monkeypatch.setattr(spendguard.eval.runner, 'build_chat_model', lambda settings, key: model)


def use_models(monkeypatch: pytest.MonkeyPatch, models: dict[str, FakeChatModel]) -> None:
    monkeypatch.setattr(
        spendguard.eval.runner, 'build_chat_model', lambda settings, key: models[key]
    )


def finished_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_models: dict[str, FakeChatModel],
    pre_flight_arguments: Sequence[str] = (),
    settings: Settings | None = None,
) -> str:
    lock_path = locked_fixture_dataset(tmp_path / 'dataset.lock.json')
    settings = one_row_settings() if settings is None else settings
    monkeypatch.setattr(spendguard.settings, 'load_settings', lambda: settings)
    use_models(monkeypatch, first_models)
    exit_code = run_eval_command(
        [
            *eval_command_arguments(lock_path, tmp_path, tmp_path / 'README.md'),
            *pre_flight_arguments,
        ]
    )
    assert exit_code == 0
    [results_file] = (tmp_path / 'results').glob('*.md')
    return results_file.stem


def rerun_arguments(tmp_path: Path, run_name: str) -> list[str]:
    return [
        *eval_command_arguments(tmp_path / 'dataset.lock.json', tmp_path, tmp_path / 'README.md'),
        '--run',
        run_name,
    ]


def saved_cases(tmp_path: Path, run_name: str, row_number: int = 1) -> dict[str, dict[str, Any]]:
    record = json.loads((tmp_path / 'results' / run_name / f'row-{row_number}.json').read_text())
    return {case['case_id']: case for case in record['cases']}


@pytest.fixture
def readme(tmp_path: Path) -> Path:
    readme_path = tmp_path / 'README.md'
    readme_path.write_text('<!-- eval-results:start -->\n<!-- eval-results:end -->\n')
    return readme_path


@pytest.mark.usefixtures('readme')
def test_a_case_that_failed_because_bedrock_was_unavailable_is_rerun_into_its_row_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_name = finished_run(
        tmp_path,
        monkeypatch,
        {'fake_primary': ReviewingModel(model_id='fake-chat', unavailable_for=[WEEKEND_CASE_TEXT])},
    )
    assert saved_cases(tmp_path, run_name)['test-weekend']['failed'] is True
    use_model(monkeypatch, ReviewingModel(model_id='fake-chat'))

    exit_code = rerun_command(rerun_arguments(tmp_path, run_name))

    assert exit_code == 0
    rerun_case = saved_cases(tmp_path, run_name)['test-weekend']
    assert rerun_case['failed'] is False
    assert rerun_case['route'] == 'human_review'


@pytest.mark.usefixtures('readme')
def test_a_case_the_model_itself_failed_is_not_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_name = finished_run(
        tmp_path,
        monkeypatch,
        {
            'fake_primary': ReviewingModel(
                model_id='fake-chat',
                unavailable_for=[WEEKEND_CASE_TEXT],
                broken_for=[CLEAN_INVOICE_TEXT],
            )
        },
    )
    first_record = saved_cases(tmp_path, run_name)['test-clean']
    use_model(monkeypatch, ReviewingModel(model_id='fake-chat'))

    exit_code = rerun_command(rerun_arguments(tmp_path, run_name))

    assert exit_code == 0
    cases = saved_cases(tmp_path, run_name)
    assert cases['test-weekend']['failed'] is False
    assert cases['test-clean'] == first_record
    assert first_record['error_type'] == 'RuntimeError'


def results_text(tmp_path: Path, run_name: str) -> str:
    return (tmp_path / 'results' / f'{run_name}.md').read_text()


def section(text: str, heading: str) -> str:
    after_heading = text.split(f'## {heading}\n\n', 1)[1]
    return after_heading.split('\n## ', 1)[0].strip()


@pytest.mark.usefixtures('readme')
def test_the_results_file_moves_a_rerun_case_from_failed_cases_to_reruns_with_its_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_name = finished_run(
        tmp_path,
        monkeypatch,
        {
            'fake_primary': ReviewingModel(
                model_id='fake-chat',
                unavailable_for=[WEEKEND_CASE_TEXT],
                broken_for=[CLEAN_INVOICE_TEXT],
            )
        },
    )
    use_model(monkeypatch, ReviewingModel(model_id='fake-chat'))

    exit_code = rerun_command(rerun_arguments(tmp_path, run_name))

    assert exit_code == 0
    text = results_text(tmp_path, run_name)
    failed_rows = section(text, 'Failed cases').splitlines()[2:]
    assert [line.strip('| ').split(' | ')[1:] for line in failed_rows] == [
        ['test-clean', 'RuntimeError'],
        ['test-missing-po', 'ValueError'],
    ]
    assert section(text, 'Reruns').splitlines()[-3:] == [
        '| Row | Case | Rerun because of | Rerun on | Requests a minute | Rerun result |',
        '| --- | --- | --- | --- | --- | --- |',
        f'| Primary | test-weekend | ServiceUnavailableException | {datetime.date.today()} |'
        f' {UNPACED_REQUESTS_PER_MINUTE} | completed |',
    ]


def run_files(tmp_path: Path, run_name: str) -> dict[str, str]:
    paths = [
        tmp_path / 'README.md',
        tmp_path / 'results' / f'{run_name}.md',
        *sorted((tmp_path / 'results' / run_name).iterdir()),
    ]
    return {path.name: path.read_text() for path in paths}


def refuse_model_calls(*arguments: object) -> NoReturn:
    raise AssertionError('the rerun built a model before refusing')


@pytest.mark.usefixtures('readme')
def test_a_rerun_refuses_a_run_its_settings_no_longer_rebuild_and_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_name = finished_run(
        tmp_path,
        monkeypatch,
        {'fake_primary': ReviewingModel(model_id='fake-chat', unavailable_for=[WEEKEND_CASE_TEXT])},
    )
    before = run_files(tmp_path, run_name)
    repriced = one_row_settings().model_copy(
        update={
            'model_prices': {
                'fake_primary': ModelPrice.model_validate(PRICE | {'priced_on': '2026-11-03'})
            }
        }
    )
    monkeypatch.setattr(spendguard.settings, 'load_settings', lambda: repriced)
    monkeypatch.setattr(spendguard.eval.runner, 'build_chat_model', refuse_model_calls)

    exit_code = rerun_command(rerun_arguments(tmp_path, run_name))

    assert exit_code == 1
    assert 'does not match a rebuild from its row files and the current settings' in (
        capsys.readouterr().err
    )
    assert run_files(tmp_path, run_name) == before


@pytest.mark.usefixtures('readme')
def test_a_run_with_no_failure_on_the_provider_s_side_is_refused_and_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_name = finished_run(
        tmp_path,
        monkeypatch,
        {'fake_primary': ReviewingModel(model_id='fake-chat', broken_for=[CLEAN_INVOICE_TEXT])},
    )
    before = run_files(tmp_path, run_name)
    monkeypatch.setattr(spendguard.eval.runner, 'build_chat_model', refuse_model_calls)

    exit_code = rerun_command(rerun_arguments(tmp_path, run_name))

    assert exit_code == 1
    assert f"No case in {run_name} failed on the provider's side" in capsys.readouterr().err
    assert run_files(tmp_path, run_name) == before


@pytest.mark.usefixtures('readme')
def test_a_pre_flight_run_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_name = finished_run(
        tmp_path,
        monkeypatch,
        {'fake_primary': ReviewingModel(model_id='fake-chat', unavailable_for=[WEEKEND_CASE_TEXT])},
        pre_flight_arguments=['--cases-per-row', '2'],
    )
    before = run_files(tmp_path, run_name)
    monkeypatch.setattr(spendguard.eval.runner, 'build_chat_model', refuse_model_calls)

    exit_code = rerun_command(rerun_arguments(tmp_path, run_name))

    assert exit_code == 1
    assert f'{run_name} is not a full eval-v2 run' in capsys.readouterr().err
    assert run_files(tmp_path, run_name) == before


def test_the_readme_table_is_rebuilt_from_the_merged_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, readme: Path
) -> None:
    run_name = finished_run(
        tmp_path,
        monkeypatch,
        {
            'fake_primary': ReviewingModel(
                model_id='fake-chat',
                unavailable_for=[WEEKEND_CASE_TEXT],
                broken_for=[CLEAN_INVOICE_TEXT],
            )
        },
    )
    assert '| 3 failed |' in readme.read_text()
    use_model(monkeypatch, ReviewingModel(model_id='fake-chat'))

    exit_code = rerun_command(rerun_arguments(tmp_path, run_name))

    assert exit_code == 0
    readme_text = readme.read_text()
    assert '| 2 failed |' in readme_text
    assert section(results_text(tmp_path, run_name), 'Metrics') in readme_text


@pytest.mark.usefixtures('readme')
def test_a_case_still_failing_after_a_rerun_is_rerun_again_with_the_error_it_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_name = finished_run(
        tmp_path,
        monkeypatch,
        {'fake_primary': ReviewingModel(model_id='fake-chat', unavailable_for=[WEEKEND_CASE_TEXT])},
    )
    use_model(monkeypatch, ReviewingModel(model_id='fake-chat', throttled_for=[WEEKEND_CASE_TEXT]))
    first_exit_code = rerun_command(rerun_arguments(tmp_path, run_name))
    use_model(monkeypatch, ReviewingModel(model_id='fake-chat'))

    second_exit_code = rerun_command(rerun_arguments(tmp_path, run_name))

    assert (first_exit_code, second_exit_code) == (0, 0)
    assert saved_cases(tmp_path, run_name)['test-weekend']['failed'] is False
    reruns = json.loads((tmp_path / 'results' / run_name / 'reruns.json').read_text())
    assert [(rerun['replaced_error_type'], rerun['result']) for rerun in reruns] == [
        ('ServiceUnavailableException', 'ThrottlingException'),
        ('ThrottlingException', 'completed'),
    ]
    assert reruns[0]['rerun_identifier'] != reruns[1]['rerun_identifier']
    assert [
        line.strip('| ').split(' | ')[2]
        for line in section(results_text(tmp_path, run_name), 'Reruns').splitlines()[-2:]
    ] == ['ServiceUnavailableException', 'ThrottlingException']


FOLLOW_UP_TEXT = 'Soup and sandwich'


async def saved_review_state(thread_key: str) -> dict[str, Any]:
    async with opened_pool(
        async_connection_pool(branch_connection_info(one_row_settings()))
    ) as connection_pool:
        checkpoint = await AsyncPostgresSaver(connection_pool).aget_tuple(thread_config(thread_key))
    assert checkpoint is not None
    values: dict[str, Any] = checkpoint.checkpoint['channel_values']
    return values


def available_precedent_reasons(thread_key: str) -> list[str]:
    state = asyncio.run(saved_review_state(thread_key))
    return [precedent['reason'] for precedent in state['current_precedents']]


@pytest.mark.usefixtures('readme')
def test_a_rerun_memory_follow_up_has_its_pair_s_seeded_precedent_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_name = finished_run(
        tmp_path,
        monkeypatch,
        {'fake_primary': ReviewingModel(model_id='fake-chat', unavailable_for=[FOLLOW_UP_TEXT])},
    )
    use_model(monkeypatch, ReviewingModel(model_id='fake-chat'))

    exit_code = rerun_command(rerun_arguments(tmp_path, run_name))

    assert exit_code == 0
    [rerun] = json.loads((tmp_path / 'results' / run_name / 'reruns.json').read_text())
    assert rerun['case_id'] == 'test-memory-follow-up'
    assert available_precedent_reasons(
        f'eval:{rerun["rerun_identifier"]}:test-memory-follow-up'
    ) == ['Standing weekly client catch-up agreed with the manager.']


RUN_DAY = datetime.date(2026, 11, 2)


class RunDay(datetime.date):
    @classmethod
    def today(cls) -> 'RunDay':
        return cls(RUN_DAY.year, RUN_DAY.month, RUN_DAY.day)


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    yield InMemorySpanExporter()
    LangfuseResourceManager.reset()


@pytest.mark.usefixtures('readme')
def test_a_rerun_is_its_own_langfuse_dataset_run_holding_only_the_rerun_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exporter: InMemorySpanExporter
) -> None:
    settings = one_row_settings().model_copy(
        update={
            'langfuse_public_key': SecretStr('pk-lf-test'),
            'langfuse_secret_key': SecretStr('sk-lf-test'),
        }
    )
    tracing = tracing_into(exporter, FakeLangfuseApi(), settings)
    monkeypatch.setattr(spendguard.eval.runner, 'create_tracing', lambda loaded_settings: tracing)
    with monkeypatch.context() as on_the_run_day:
        on_the_run_day.setattr(spendguard.eval.runner, 'datetime', SimpleNamespace(date=RunDay))
        run_name = finished_run(
            tmp_path,
            monkeypatch,
            {
                'fake_primary': ReviewingModel(
                    model_id='fake-chat', unavailable_for=[WEEKEND_CASE_TEXT]
                )
            },
            settings=settings,
        )
    use_model(monkeypatch, ReviewingModel(model_id='fake-chat'))

    exit_code = rerun_command(rerun_arguments(tmp_path, run_name))

    assert exit_code == 0
    tracing.client.flush()
    [rerun] = json.loads((tmp_path / 'results' / run_name / 'reruns.json').read_text())
    rerun_item_names = [
        attribute(span, 'langfuse.experiment.name')
        for span in exporter.get_finished_spans()
        if span.name == 'experiment-item-run'
        and ':rerun-' in attribute(span, 'langfuse.experiment.name')
    ]
    assert rerun['rerun_on'] != RUN_DAY.isoformat()
    assert rerun_item_names == [f'{RUN_DAY} Primary {rerun["rerun_identifier"]}']


@pytest.mark.usefixtures('readme')
def test_a_failure_in_the_second_row_is_rerun_with_that_row_s_model_and_leaves_the_first_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_name = finished_run(
        tmp_path,
        monkeypatch,
        {
            'fake_primary': ReviewingModel(model_id='fake-chat'),
            'fake_second': ReviewingModel(
                model_id='fake-second', unavailable_for=[WEEKEND_CASE_TEXT]
            ),
        },
        settings=two_row_settings(),
    )
    first_row_file = (tmp_path / 'results' / run_name / 'row-1.json').read_text()
    primary_rerun_model = ReviewingModel(model_id='fake-chat')
    second_rerun_model = ReviewingModel(model_id='fake-second')
    use_models(
        monkeypatch, {'fake_primary': primary_rerun_model, 'fake_second': second_rerun_model}
    )

    exit_code = rerun_command(rerun_arguments(tmp_path, run_name))

    assert exit_code == 0
    assert (tmp_path / 'results' / run_name / 'row-1.json').read_text() == first_row_file
    assert saved_cases(tmp_path, run_name, row_number=2)['test-weekend']['failed'] is False
    assert primary_rerun_model.received == []
    assert second_rerun_model.received != []
    [rerun_row] = section(results_text(tmp_path, run_name), 'Reruns').splitlines()[-1:]
    assert rerun_row.startswith('| Second | test-weekend |')


@pytest.mark.usefixtures('readme')
def test_a_rerun_after_an_interrupted_one_reviews_each_case_on_a_fresh_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_name = finished_run(
        tmp_path,
        monkeypatch,
        {
            'fake_primary': ReviewingModel(
                model_id='fake-chat', unavailable_for=[WEEKEND_CASE_TEXT, FOLLOW_UP_TEXT]
            )
        },
    )
    use_model(monkeypatch, ReviewingModel(model_id='fake-chat', interrupted_by=[FOLLOW_UP_TEXT]))
    with pytest.raises(RerunInterruptedError):
        rerun_command(rerun_arguments(tmp_path, run_name))
    use_model(monkeypatch, ReviewingModel(model_id='fake-chat'))

    exit_code = rerun_command(rerun_arguments(tmp_path, run_name))

    assert exit_code == 0
    reruns = json.loads((tmp_path / 'results' / run_name / 'reruns.json').read_text())
    [weekend_rerun] = [rerun for rerun in reruns if rerun['case_id'] == 'test-weekend']
    state = asyncio.run(
        saved_review_state(f'eval:{weekend_rerun["rerun_identifier"]}:test-weekend')
    )
    assert [entry['event'] for entry in state['audit']].count('policy_retrieved') == 1


@pytest.mark.usefixtures('readme')
def test_a_rerun_interrupted_in_a_later_row_keeps_the_rows_it_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_name = finished_run(
        tmp_path,
        monkeypatch,
        {
            'fake_primary': ReviewingModel(
                model_id='fake-chat', unavailable_for=[WEEKEND_CASE_TEXT]
            ),
            'fake_second': ReviewingModel(
                model_id='fake-second', unavailable_for=[WEEKEND_CASE_TEXT, CLEAN_INVOICE_TEXT]
            ),
        },
        settings=two_row_settings(),
    )
    use_models(
        monkeypatch,
        {
            'fake_primary': ReviewingModel(model_id='fake-chat'),
            'fake_second': ReviewingModel(
                model_id='fake-second', interrupted_by=[CLEAN_INVOICE_TEXT]
            ),
        },
    )
    with pytest.raises(RerunInterruptedError):
        rerun_command(rerun_arguments(tmp_path, run_name))

    assert saved_cases(tmp_path, run_name, row_number=1)['test-weekend']['failed'] is False
    assert saved_cases(tmp_path, run_name, row_number=2)['test-weekend']['failed'] is True
    assert [
        line.strip('| ').split(' | ')[:2]
        for line in section(results_text(tmp_path, run_name), 'Reruns').splitlines()[-1:]
    ] == [['Primary', 'test-weekend']]
    use_models(
        monkeypatch,
        {
            'fake_primary': ReviewingModel(model_id='fake-chat'),
            'fake_second': ReviewingModel(model_id='fake-second'),
        },
    )

    exit_code = rerun_command(rerun_arguments(tmp_path, run_name))

    assert exit_code == 0
    reruns = json.loads((tmp_path / 'results' / run_name / 'reruns.json').read_text())
    assert [(rerun['row_label'], rerun['case_id']) for rerun in reruns] == [
        ('Primary', 'test-weekend'),
        ('Second', 'test-clean'),
        ('Second', 'test-weekend'),
    ]
