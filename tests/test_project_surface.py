import re
from pathlib import Path

import pytest
import yaml

from spendguard.settings import MissingEnvFileError, Settings
from spendguard.settings import load_settings as load_settings_from_files

REPOSITORY_ROOT = Path(__file__).parent.parent
ENV_EXAMPLE = REPOSITORY_ROOT / '.env.example'

TARGETS_THE_SPECS_NAME = {
    'seed',
    'smoke-providers',
    'draft-cases',
    'freeze',
    'issue-token',
    'bootstrap',
    'secrets',
    'deploy',
    'destroy',
    'destroy-bootstrap',
}

env_key_pattern = re.compile(r'^([A-Z][A-Z0-9_]*)=', re.MULTILINE)
make_target_pattern = re.compile(r'^([a-z][a-z0-9-]*):', re.MULTILINE)


def env_example_keys() -> list[str]:
    return env_key_pattern.findall(ENV_EXAMPLE.read_text())


def test_every_makefile_target_the_specs_name_exists() -> None:
    targets = set(make_target_pattern.findall((REPOSITORY_ROOT / 'Makefile').read_text()))

    assert TARGETS_THE_SPECS_NAME - targets == set()


def test_every_env_example_key_is_a_setting_and_every_setting_has_a_key() -> None:
    top_level_keys = {key.split('__')[0].lower() for key in env_example_keys()}

    assert top_level_keys == set(Settings.model_fields) - {'branch_name'}
    assert len(env_example_keys()) == len(set(env_example_keys()))


def test_the_env_example_file_loads_into_valid_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in env_example_keys():
        monkeypatch.delenv(key, raising=False)

    settings = load_settings_from_files(ENV_EXAMPLE)

    assert settings.app_environment == 'local'
    assert settings.google_api_key is None
    assert settings.chat_models['gpt_oss'].provider == 'bedrock'
    assert settings.allowed_providers == ('bedrock', 'google_genai')
    assert settings.primary_model_key in settings.chat_models
    assert set(settings.drafting_model_keys) <= set(settings.chat_models)


def test_a_missing_env_file_is_named_instead_of_listing_every_missing_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for key in env_example_keys():
        monkeypatch.delenv(key, raising=False)
    missing_env_file = tmp_path / '.env'

    with pytest.raises(MissingEnvFileError) as raised:
        load_settings_from_files(missing_env_file)

    assert str(missing_env_file) in str(raised.value)
    assert '.env.example' in str(raised.value)


def test_the_eval_workflow_runs_only_when_dispatched_by_hand() -> None:
    workflow = yaml.load(
        (REPOSITORY_ROOT / '.github' / 'workflows' / 'eval.yml').read_text(), Loader=yaml.BaseLoader
    )

    assert workflow['on'] == {'workflow_dispatch': ''}


def test_the_readme_has_no_em_dashes() -> None:
    readme = (REPOSITORY_ROOT / 'README.md').read_text()

    assert '\N{EM DASH}' not in readme


def test_every_relative_link_in_the_readme_points_to_a_file_in_the_repository() -> None:
    readme = (REPOSITORY_ROOT / 'README.md').read_text()
    link_targets = re.findall(r'\]\(([^)#\s]+)', readme)

    relative_targets = [target for target in link_targets if '://' not in target]
    assert relative_targets
    assert [target for target in relative_targets if not (REPOSITORY_ROOT / target).exists()] == []
