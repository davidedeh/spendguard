import datetime
from decimal import Decimal
from pathlib import Path

import pytest

import spendguard.settings
from settings_factory import make_test_settings


def test_test_settings_ignore_the_environment_and_any_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / '.env').write_text('POSTGRES_HOST=from-env-file\nGOOGLE_API_KEY=from-env-file\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('POSTGRES_HOST', 'from-environment')
    monkeypatch.setenv('GOOGLE_API_KEY', 'from-environment')

    settings = make_test_settings(branch_name='a-branch')

    assert settings.postgres_host == 'localhost'
    assert settings.google_api_key is None
    assert settings.allowed_providers == ('fake',)


def test_loading_settings_from_the_environment_is_refused_inside_tests() -> None:
    with pytest.raises(RuntimeError, match='make_test_settings'):
        spendguard.settings.load_settings()


def test_a_model_price_records_the_date_it_was_taken() -> None:
    settings = make_test_settings(
        model_prices={
            'fake_primary': {
                'input_per_million_tokens_usd': '1.00',
                'output_per_million_tokens_usd': '5.00',
                'priced_on': '2026-11-16',
            }
        }
    )

    assert settings.model_prices['fake_primary'].priced_on == datetime.date(2026, 11, 16)
    assert settings.model_prices['fake_primary'].output_per_million_tokens_usd == Decimal('5.00')


def test_a_visibility_timeout_under_six_worker_timeouts_is_rejected() -> None:
    with pytest.raises(ValueError, match=r'at least 6 times WORKER_TIMEOUT_SECONDS \(720\)'):
        make_test_settings(worker_timeout_seconds=120, queue_visibility_timeout_seconds=719)


def test_a_visibility_timeout_of_exactly_six_worker_timeouts_is_accepted() -> None:
    settings = make_test_settings(worker_timeout_seconds=50, queue_visibility_timeout_seconds=300)

    assert settings.queue_visibility_timeout_seconds == 300


def test_an_eval_row_naming_an_unconfigured_chat_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="Eval row 'Claude Opus' uses chat model 'claude_opus'"):
        make_test_settings(eval_rows=[{'model_key': 'claude_opus', 'label': 'Claude Opus'}])
