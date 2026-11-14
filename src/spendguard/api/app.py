from fastapi import FastAPI

from spendguard.api.routes import health, mock_erp, submissions, webhooks


def create_app() -> FastAPI:
    app = FastAPI(title='SpendGuard')
    for module in (health, webhooks, submissions, mock_erp):
        app.include_router(module.router)
    return app
