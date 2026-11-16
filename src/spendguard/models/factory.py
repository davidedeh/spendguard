from typing import assert_never

from botocore.config import Config
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from spendguard.models.fake import FakeChatModel
from spendguard.settings import Settings

EMBEDDING_DIMENSIONS = 1024
# Bedrock compiles a new output schema for up to about 90 seconds on first use.
BEDROCK_READ_TIMEOUT_SECONDS = 120


def build_chat_model(settings: Settings, model_key: str) -> BaseChatModel:
    model = settings.chat_models[model_key]
    match model.provider:
        case 'bedrock':
            return ChatBedrockConverse(
                model=model.model_id,
                region_name=settings.aws_region,
                config=Config(read_timeout=BEDROCK_READ_TIMEOUT_SECONDS),
            )
        case 'google_genai':
            return ChatGoogleGenerativeAI(
                model=model.model_id, google_api_key=settings.google_api_key
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
            return DeterministicFakeEmbedding(size=EMBEDDING_DIMENSIONS)
        case _:
            assert_never(model.provider)
