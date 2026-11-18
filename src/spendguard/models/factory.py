from typing import assert_never

from botocore.config import Config
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from spendguard.models.fake import FakeChatModel, KeywordFakeEmbedder
from spendguard.settings import Settings

EMBEDDING_DIMENSIONS = 1024
# Bedrock compiles a new output schema for up to about 90 seconds on first use.
BEDROCK_READ_TIMEOUT_SECONDS = 120
# Bedrock token quotas refill each minute; nine retries back off for up to about 110 seconds.
BEDROCK_MAX_ATTEMPTS = 10


def build_chat_model(settings: Settings, model_key: str) -> BaseChatModel:
    model = settings.chat_models[model_key]
    match model.provider:
        case 'bedrock':
            return ChatBedrockConverse(
                model=model.model_id,
                region_name=settings.aws_region,
                temperature=model.temperature,
                config=Config(
                    read_timeout=BEDROCK_READ_TIMEOUT_SECONDS,
                    retries={'mode': 'adaptive', 'total_max_attempts': BEDROCK_MAX_ATTEMPTS},
                ),
            )
        case 'google_genai':
            return ChatGoogleGenerativeAI(
                model=model.model_id,
                google_api_key=settings.google_api_key,
                temperature=model.temperature,
            )
        case 'fake':
            return FakeChatModel(model_id=model.model_id)
        case _:
            assert_never(model.provider)


def build_embedder(settings: Settings) -> Embeddings:
    model = settings.embedding_model
    match model.provider:
        case 'bedrock':
            return BedrockEmbeddings(
                model_id=model.model_id,
                region_name=settings.aws_region,
                dimensions=EMBEDDING_DIMENSIONS,
            )
        case 'google_genai':
            raise ValueError('Embeddings are served by bedrock, or fake in tests, not google_genai')
        case 'fake':
            return KeywordFakeEmbedder(size=EMBEDDING_DIMENSIONS)
        case _:
            assert_never(model.provider)
