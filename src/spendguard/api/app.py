from fastapi import FastAPI

from spendguard.api.dependencies import ApiDependencies
from spendguard.api.routes import health, mock_erp, submissions, webhooks


def create_app(dependencies: ApiDependencies) -> FastAPI:
    app = FastAPI(title='SpendGuard')
    app.state.dependencies = dependencies
    for module in (health, webhooks, submissions, mock_erp):
        app.include_router(module.router)
    return app
