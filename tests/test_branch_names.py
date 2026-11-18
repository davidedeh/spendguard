import re

import pytest

from spendguard.branch_names import (
    current_branch_name,
    database_name_for_branch,
    dead_letter_queue_name_for,
    queue_name_for_branch,
)


def test_two_branches_that_sanitise_alike_get_different_database_and_queue_names() -> None:
    assert database_name_for_branch('feature/a') != database_name_for_branch('feature-a')
    assert queue_name_for_branch('feature/a') != queue_name_for_branch('feature-a')


def test_a_long_branch_name_gives_names_that_postgres_and_sqs_accept() -> None:
    branch_name = 'Feature/' + 'Very-Long.Branch_Name ' * 10

    database_name = database_name_for_branch(branch_name)
    queue_name = queue_name_for_branch(branch_name)

    assert re.fullmatch(r'[a-z_][a-z0-9_]{0,62}', database_name)
    assert re.fullmatch(r'[A-Za-z0-9_-]{1,75}\.fifo', queue_name)
    assert database_name.startswith('spendguard_feature_very_long_branch_name')
    assert queue_name.startswith('spendguard-feature-very-long-branch-name')


def test_a_long_branch_name_gives_a_dead_letter_queue_name_that_sqs_accepts() -> None:
    branch_name = 'Feature/' + 'Very-Long.Branch_Name ' * 10

    dead_letter_queue_name = dead_letter_queue_name_for(queue_name_for_branch(branch_name))

    assert re.fullmatch(r'[A-Za-z0-9_-]{1,75}\.fifo', dead_letter_queue_name)
    assert dead_letter_queue_name.endswith('-dead-letter.fifo')
    assert dead_letter_queue_name != queue_name_for_branch(branch_name)


def test_a_branch_name_in_the_environment_is_used_where_there_is_no_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv('GITHUB_HEAD_REF', raising=False)
    monkeypatch.delenv('GITHUB_REF_NAME', raising=False)
    monkeypatch.setenv('BRANCH_NAME', 'deployed-branch')
    monkeypatch.setenv('PATH', '')

    assert current_branch_name() == 'deployed-branch'
