import time
from dataclasses import dataclass
from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import SecretStr

from spendguard.db.database import ConnectionPool
from spendguard.queue.messages import MessageQueue
from spendguard.security.approver_tokens import token_hash
from spendguard.security.signatures import SIGNATURE_HEADER, verify_signature
from spendguard.settings import Settings
from spendguard.store import approvers
from spendguard.store.approvers import AuthenticatedApprover

_BEARER = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class ApiDependencies:
    settings: Settings
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


async def authenticated_approver(
    dependencies: Annotated[ApiDependencies, Depends(api_dependencies)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER)],
) -> AuthenticatedApprover:
    approver = (
        None
        if credentials is None
        else await approvers.find_by_token_hash(
            dependencies.connection_pool, token_hash(credentials.credentials)
        )
    )
    if approver is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            'invalid approver token',
            headers={'WWW-Authenticate': 'Bearer'},
        )
    return approver
