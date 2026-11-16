import time
from dataclasses import dataclass
from typing import cast

from fastapi import HTTPException, Request, status
from pydantic import SecretStr

from spendguard.db.database import ConnectionPool
from spendguard.policy.loader import Policy
from spendguard.queue.messages import MessageQueue
from spendguard.security.signatures import SIGNATURE_HEADER, verify_signature
from spendguard.seed.schema import SeedData
from spendguard.settings import Settings


@dataclass(frozen=True)
class ApiDependencies:
    settings: Settings
    policy: Policy
    seed_data: SeedData
    connection_pool: ConnectionPool
    queue: MessageQueue


def api_dependencies(request: Request) -> ApiDependencies:
    return cast(ApiDependencies, request.app.state.dependencies)


async def signed_body(request: Request, key: SecretStr | None, tolerance_seconds: int) -> bytes:
    raw_body = await request.body()
    if key is None or not verify_signature(
        key.get_secret_value().encode(),
        request.headers.get(SIGNATURE_HEADER),
        raw_body,
        int(time.time()),
        tolerance_seconds,
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, 'invalid signature')
    return raw_body
