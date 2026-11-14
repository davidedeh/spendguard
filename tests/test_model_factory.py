import pytest
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import ValidationError

from settings_factory import make_test_settings
from spendguard.models.factory import build_chat_model, build_embedder
from spendguard.models.fake import FakeChatModel
from spendguard.settings import ModelSettings


@pytest.fixture(autouse=True)
def _aws_credentials_are_dummies(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ('AWS_PROFILE', 'AWS_SESSION_TOKEN', 'AWS_BEARER_TOKEN_BEDROCK'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'test-access-key-id')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'test-secret-access-key')
    monkeypatch.setenv('AWS_EC2_METADATA_DISABLED', 'true')


def test_a_fake_model_key_builds_the_fake_chat_model_with_its_configured_id() -> None:
    settings = make_test_settings(
        chat_models={
            'fake_primary': ModelSettings(provider='fake', model_id='fake-chat'),
            'fake_drafter': ModelSettings(provider='fake', model_id='fake-drafter-2'),
        },
    )

    model = build_chat_model(settings, 'fake_drafter')

    assert isinstance(model, FakeChatModel)
    assert model.model_id == 'fake-drafter-2'


def test_a_bedrock_model_key_builds_a_converse_model_with_its_configured_id_and_region() -> None:
    settings = make_test_settings(
        aws_region='eu-central-1',
        allowed_providers=('fake', 'bedrock'),
        chat_models={
            'fake_primary': ModelSettings(provider='fake', model_id='fake-chat'),
            'gpt_oss': ModelSettings(provider='bedrock', model_id='openai.a-gpt-oss-id'),
        },
    )

    model = build_chat_model(settings, 'gpt_oss')

    assert isinstance(model, ChatBedrockConverse)
    assert model.model_id == 'openai.a-gpt-oss-id'
    assert model.region_name == 'eu-central-1'


def test_a_google_genai_model_key_builds_gemini_with_its_configured_id_and_api_key() -> None:
    settings = make_test_settings(
        allowed_providers=('fake', 'google_genai'),
        chat_models={
            'fake_primary': ModelSettings(provider='fake', model_id='fake-chat'),
            'gemini_pro': ModelSettings(provider='google_genai', model_id='a-gemini-id'),
        },
        google_api_key='test-google-api-key',
    )

    model = build_chat_model(settings, 'gemini_pro')

    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model == 'a-gemini-id'
    assert model.google_api_key is not None
    assert model.google_api_key.get_secret_value() == 'test-google-api-key'


def test_the_test_configuration_builds_a_fake_embedder_with_stable_titan_sized_vectors() -> None:
    embedder = build_embedder(make_test_settings())

    first, second, first_again = embedder.embed_documents(
        ['a clause', 'another clause', 'a clause']
    )

    assert len(first) == 1024
    assert first == first_again
    assert first != second
    assert embedder.embed_query('a clause') == first


def test_a_bedrock_embedding_model_builds_titan_with_its_configured_id_region_and_size() -> None:
    settings = make_test_settings(
        aws_region='eu-central-1',
        allowed_providers=('fake', 'bedrock'),
        embedding_model=ModelSettings(provider='bedrock', model_id='amazon.a-titan-id'),
    )

    embedder = build_embedder(settings)

    assert isinstance(embedder, BedrockEmbeddings)
    assert embedder.model_id == 'amazon.a-titan-id'
    assert embedder.region_name == 'eu-central-1'
    assert embedder.dimensions == 1024


def test_a_chat_model_whose_provider_is_not_allowed_fails_settings_validation() -> None:
    with pytest.raises(ValidationError, match="'gemini_pro' uses provider 'google_genai'"):
        make_test_settings(
            allowed_providers=('fake', 'bedrock'),
            chat_models={
                'fake_primary': ModelSettings(provider='fake', model_id='fake-chat'),
                'claude_haiku': ModelSettings(provider='bedrock', model_id='a-claude-id'),
                'gemini_pro': ModelSettings(provider='google_genai', model_id='a-gemini-id'),
            },
        )


def test_an_embedding_model_whose_provider_is_not_allowed_fails_settings_validation() -> None:
    with pytest.raises(ValidationError, match="embedding model uses provider 'bedrock'"):
        make_test_settings(
            embedding_model=ModelSettings(provider='bedrock', model_id='a-titan-id'),
        )


def test_the_deployed_environment_refuses_any_allowed_provider_other_than_bedrock() -> None:
    bedrock_model = ModelSettings(provider='bedrock', model_id='a-claude-id')

    with pytest.raises(ValidationError, match='deployed environment allows bedrock only'):
        make_test_settings(
            app_environment='deployed',
            allowed_providers=('bedrock', 'google_genai'),
            chat_models={'claude_haiku': bedrock_model},
            primary_model_key='claude_haiku',
            drafting_model_keys=('claude_haiku',),
            embedding_model=bedrock_model,
        )
