from spendguard.settings import Settings


def build_chat_model(settings: Settings, model_key: str) -> object:
    raise NotImplementedError


def build_embedder(settings: Settings) -> object:
    raise NotImplementedError
