import datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from spendguard.branch_names import current_branch_name

Provider = Literal['bedrock', 'google_genai', 'fake']
AppEnvironment = Literal['local', 'test', 'deployed']
StructuredOutputMethod = Literal['function_calling', 'json_schema']
VISIBILITY_TIMEOUTS_PER_WORKER_TIMEOUT = 6


class ModelSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Provider
    model_id: str


class ChatModelSettings(ModelSettings):
    structured_output_method: StructuredOutputMethod


class ModelPrice(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_per_million_tokens_usd: Decimal
    output_per_million_tokens_usd: Decimal
    priced_on: datetime.date


class Settings(BaseSettings):
    model_config = SettingsConfigDict(frozen=True, env_nested_delimiter='__')

    app_environment: AppEnvironment
    log_level: str
    branch_name: str

    postgres_host: str
    postgres_port: int
    postgres_user: str
    postgres_password: SecretStr
    database_url: SecretStr | None

    aws_region: str
    sqs_endpoint_url: str | None
    queue_url: str | None
    ssm_parameter_prefix: str | None

    allowed_providers: tuple[Provider, ...]
    chat_models: dict[str, ChatModelSettings]
    primary_model_key: str
    drafting_model_keys: tuple[str, ...]
    embedding_model: ModelSettings
    model_prices: dict[str, ModelPrice]
    google_api_key: SecretStr | None

    langfuse_public_key: SecretStr | None
    langfuse_secret_key: SecretStr | None
    langfuse_host: str

    webhook_signing_key: SecretStr | None
    decision_event_signing_key: SecretStr | None
    decision_event_url: str
    signature_tolerance_seconds: int

    force_human_review: bool
    judgement_max_tool_calls: int
    retrieval_top_k: int
    worker_timeout_seconds: int
    queue_visibility_timeout_seconds: int
    sweeper_stuck_after_seconds: int

    @model_validator(mode='after')
    def _every_configured_provider_is_allowed(self) -> Self:
        if self.app_environment == 'deployed' and self.allowed_providers != ('bedrock',):
            raise ValueError(
                'The deployed environment allows bedrock only, '
                f'but ALLOWED_PROVIDERS is {list(self.allowed_providers)}'
            )
        for model_key, model in self.chat_models.items():
            if model.provider not in self.allowed_providers:
                raise ValueError(
                    f'Chat model {model_key!r} uses provider {model.provider!r}, '
                    f'which is not in ALLOWED_PROVIDERS {list(self.allowed_providers)}'
                )
        if self.embedding_model.provider not in self.allowed_providers:
            raise ValueError(
                f'The embedding model uses provider {self.embedding_model.provider!r}, '
                f'which is not in ALLOWED_PROVIDERS {list(self.allowed_providers)}'
            )
        return self

    @model_validator(mode='after')
    def _visibility_timeout_outlasts_the_worker(self) -> Self:
        minimum = VISIBILITY_TIMEOUTS_PER_WORKER_TIMEOUT * self.worker_timeout_seconds
        if self.queue_visibility_timeout_seconds < minimum:
            raise ValueError(
                'QUEUE_VISIBILITY_TIMEOUT_SECONDS must be at least'
                f' {VISIBILITY_TIMEOUTS_PER_WORKER_TIMEOUT} times'
                f' WORKER_TIMEOUT_SECONDS ({minimum}),'
                f' but is {self.queue_visibility_timeout_seconds}'
            )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings,)


class _EnvironmentSettings(Settings):
    model_config = SettingsConfigDict(env_parse_none_str='')

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, dotenv_settings)


class MissingEnvFileError(RuntimeError):
    def __init__(self, env_file: Path) -> None:
        super().__init__(f'{env_file} does not exist: copy .env.example to it and fill it in')


def load_settings(env_file: Path | None = Path('.env')) -> Settings:
    try:
        environment_settings = _EnvironmentSettings(  # type: ignore[call-arg]
            _env_file=env_file,
            branch_name=current_branch_name(),
        )
    except ValidationError as error:
        if env_file is not None and not env_file.exists():
            raise MissingEnvFileError(env_file) from error
        raise
    return Settings.model_validate(environment_settings.model_dump())
