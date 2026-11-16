import httpx
from fastapi import FastAPI

from spendguard.deployment.smoke import smoke_failures


async def test_the_smoke_check_reports_a_webhook_that_accepts_an_unsigned_delivery() -> None:
    careless_app = FastAPI()

    @careless_app.get('/health')
    async def health() -> dict[str, str]:
        return {'status': 'ok'}

    @careless_app.post('/webhooks/expense-claims', status_code=202)
    async def accept_anything() -> dict[str, str]:
        return {'submission_id': 'accepted'}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=careless_app), base_url='http://deployed'
    ) as client:
        failures = await smoke_failures(client)

    assert failures == ['POST /webhooks/expense-claims unsigned returned 202, expected 401']
