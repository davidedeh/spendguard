import asyncio
import sys

import httpx

UNSIGNED_WEBHOOK_PATH = '/webhooks/expense-claims'
REQUEST_TIMEOUT_SECONDS = 30


async def smoke_failures(client: httpx.AsyncClient) -> list[str]:
    failures: list[str] = []
    health = await client.get('/health')
    if health.status_code != 200:
        failures.append(f'GET /health returned {health.status_code}, expected 200')
    unsigned = await client.post(UNSIGNED_WEBHOOK_PATH, content=b'{}')
    if unsigned.status_code != 401:
        failures.append(
            f'POST {UNSIGNED_WEBHOOK_PATH} unsigned returned {unsigned.status_code}, expected 401'
        )
    return failures


async def _smoke(base_url: str) -> list[str]:
    async with httpx.AsyncClient(base_url=base_url, timeout=REQUEST_TIMEOUT_SECONDS) as client:
        return await smoke_failures(client)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit('usage: python -m spendguard.deployment.smoke <api base URL>')
    failures = asyncio.run(_smoke(sys.argv[1]))
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        sys.exit(1)
    print('smoke check passed')


if __name__ == '__main__':
    main()
