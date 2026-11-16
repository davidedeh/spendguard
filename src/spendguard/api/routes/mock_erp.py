from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import ValidationError

from spendguard.api.dependencies import ApiDependencies, api_dependencies, signed_body
from spendguard.graph.nodes.execute import DecisionEvent
from spendguard.store import decision_events

router = APIRouter()


@router.post('/mock-erp/decision-events')
async def receive_decision_event(
    request: Request,
    response: Response,
    dependencies: Annotated[ApiDependencies, Depends(api_dependencies)],
) -> dict[str, Any]:
    settings = dependencies.settings
    raw_body = await signed_body(
        request, settings.decision_event_signing_key, settings.signature_tolerance_seconds
    )
    try:
        event = DecisionEvent.model_validate_json(raw_body)
    except ValidationError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, 'malformed event') from error
    stored, created = await decision_events.record_event_once(
        dependencies.connection_pool, event.idempotency_key, event.model_dump(mode='json')
    )
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return stored
