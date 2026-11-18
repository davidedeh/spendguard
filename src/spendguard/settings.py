import datetime
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

import boto3
from pydantic import BaseModel, ConfigDict, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from spendguard.branch_names import current_branch_name

if TYPE_CHECKING:
    from mypy_boto3_ssm import SSMClient

Provider = Literal['bedrock', 'google_genai', 'fake']
AppEnvironment = Literal['local', 'test', 'deployed']
StructuredOutputMethod = Literal['function_calling', 'json_schema']
PolicyContext = Literal['retrieved_clauses', 'whole_policy']
VISIBILITY_TIMEOUTS_PER_WORKER_TIMEOUT = 6


class ModelSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Provider
    model_id: str


class ChatModelSettings(ModelSettings):
    structured_output_method: StructuredOutputMethod
    temperature: float | None


class EvalRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_key: str
    label: str
    policy_context: PolicyContext


class ModelPrice(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_per_million_tokens_usd: Decimal
    output_per_million_tokens_usd: Decimal
    priced_on: datetime.date


@dataclass(frozen=True)
class LocalPostgres:
    host: str
    port: int
    user: str
    password: SecretStr


class Settings(BaseSettings):
    model_config = SettingsConfigDict(frozen=True, env_nested_delimiter='__')

    app_environment: AppEnvironment
    log_level: str
    branch_name: str

    postgres_host: str | None
    postgres_port: int | None
    postgres_user: str | None
    postgres_password: SecretStr | None
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
    eval_rows: tuple[EvalRow, ...]
    google_api_key: SecretStr | None

    langfuse_public_key: SecretStr | None
    langfuse_secret_key: SecretStr | None
    langfuse_host: str

    webhook_signing_key: SecretStr | None
    decision_event_signing_key: SecretStr | None
    decision_event_url: str | None
    signature_tolerance_seconds: int

    force_human_review: bool
    maximum_field_length: int
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
    def _every_eval_row_uses_a_configured_chat_model(self) -> Self:
        for row in self.eval_rows:
            if row.model_key not in self.chat_models:
                raise ValueError(
                    f'Eval row {row.label!r} uses chat model {row.model_key!r},'
                    f' which is not in CHAT_MODELS {sorted(self.chat_models)}'
                )
        return self

    @model_validator(mode='after')
    def _a_database_is_configured(self) -> Self:
        if self.database_url is None and self.local_postgres() is None:
            raise ValueError('Set DATABASE_URL or every POSTGRES_ setting')
        return self

    def local_postgres(self) -> 'LocalPostgres | None':
        if (
            self.postgres_host is None
            or self.postgres_port is None
            or self.postgres_user is None
            or self.postgres_password is None
        ):
            return None
        return LocalPostgres(
            host=self.postgres_host,
            port=self.postgres_port,
            user=self.postgres_user,
            password=self.postgres_password,
        )

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

    @model_validator(mode='after')
    def _sweeper_waits_out_the_visibility_timeout(self) -> Self:
        if self.sweeper_stuck_after_seconds <= self.queue_visibility_timeout_seconds:
            raise ValueError(
                'SWEEPER_STUCK_AFTER_SECONDS must be greater than'
                f' QUEUE_VISIBILITY_TIMEOUT_SECONDS ({self.queue_visibility_timeout_seconds}),'
                f' but is {self.sweeper_stuck_after_seconds}'
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


def parameter_store_values(client: 'SSMClient', prefix: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for page in client.get_paginator('get_parameters_by_path').paginate(
        Path=prefix, WithDecryption=True, Recursive=False
    ):
        for parameter in page['Parameters']:
            values[parameter['Name'].rsplit('/', 1)[-1].lower()] = parameter['Value']
    return values


class _ParameterStoreSource(PydanticBaseSettingsSource):
    def __init__(
        self,
        settings_cls: type[BaseSettings],
        configuration_sources: tuple[PydanticBaseSettingsSource, ...],
    ) -> None:
        super().__init__(settings_cls)
        self.configuration_sources = configuration_sources

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        configuration: dict[str, Any] = {}
        for source in reversed(self.configuration_sources):
            configuration |= {key: value for key, value in source().items() if value is not None}
        prefix = configuration.get('ssm_parameter_prefix')
        if not prefix:
            return {}
        client = boto3.client('ssm', region_name=configuration['aws_region'])
        return dict(parameter_store_values(client, prefix))


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
        return (
            init_settings,
            _ParameterStoreSource(settings_cls, (env_settings, dotenv_settings)),
            env_settings,
            dotenv_settings,
        )


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
        if (
            env_file is not None
            and not env_file.exists()
            and not os.environ.get('SSM_PARAMETER_PREFIX')
        ):
            raise MissingEnvFileError(env_file) from error
        raise
    return Settings.model_validate(environment_settings.model_dump())
