from typing import NoReturn

import pytest

import spendguard.settings

skipped_test_ids_key = pytest.StashKey[list[str]]()


def pytest_configure(config: pytest.Config) -> None:
    config.stash[skipped_test_ids_key] = []


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.skipped:
        report.config.stash[skipped_test_ids_key].append(report.nodeid)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if session.config.stash[skipped_test_ids_key] and exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    skipped_test_ids = terminalreporter.config.stash[skipped_test_ids_key]
    if skipped_test_ids:
        terminalreporter.section('skipped tests fail this run')
        for test_id in skipped_test_ids:
            terminalreporter.line(test_id)


def _refuse_to_load_settings(*arguments: object, **keyword_arguments: object) -> NoReturn:
    raise RuntimeError('Tests build settings with make_test_settings, never from the environment')


@pytest.fixture(autouse=True)
def _settings_never_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spendguard.settings, 'load_settings', _refuse_to_load_settings)
