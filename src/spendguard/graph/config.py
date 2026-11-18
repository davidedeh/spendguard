from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver

from spendguard.db.database import ConnectionPool
from spendguard.graph.nodes.execute import DecisionEventSender
from spendguard.policy.loader import Policy
from spendguard.seed.schema import SeedData
from spendguard.settings import PolicyContext, Settings


@dataclass(frozen=True)
class GraphConfig:
    settings: Settings
    policy: Policy
    seed_data: SeedData
    chat_model: BaseChatModel
    embedder: Embeddings
    connection_pool: ConnectionPool
    checkpointer: BaseCheckpointSaver[str]
    decision_event_sender: DecisionEventSender
    precedent_memory: bool
    record_precedents: bool
    policy_context: PolicyContext


def thread_config(idempotency_key: str) -> RunnableConfig:
    return {'configurable': {'thread_id': idempotency_key}}
