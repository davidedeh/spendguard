from typing import Any

from spendguard.branch_names import current_branch_name
from spendguard.settings import ChatModelSettings, ModelSettings, Settings


def make_test_settings(**overrides: Any) -> Settings:
    fields: dict[str, Any] = {
        'app_environment': 'test',
        'log_level': 'INFO',
        'postgres_host': 'localhost',
        'postgres_port': 5432,
        'postgres_user': 'spendguard',
        'postgres_password': 'spendguard',
        'database_url': None,
        'aws_region': 'eu-west-2',
        'sqs_endpoint_url': 'http://localhost:9324',
        'queue_url': None,
        'ssm_parameter_prefix': None,
        'allowed_providers': ('fake',),
        'chat_models': {
            'fake_primary': ChatModelSettings(
                provider='fake', model_id='fake-chat', structured_output_method='function_calling'
            ),
        },
        'primary_model_key': 'fake_primary',
        'drafting_model_keys': ('fake_primary',),
        'embedding_model': ModelSettings(provider='fake', model_id='fake-embedder'),
        'model_prices': {},
        'google_api_key': None,
        'langfuse_public_key': None,
        'langfuse_secret_key': None,
        'langfuse_host': 'http://langfuse.invalid',
        'webhook_signing_key': 'test-webhook-signing-key',
        'decision_event_signing_key': 'test-decision-event-signing-key',
        'decision_event_url': 'http://testserver/mock-erp/decision-events',
        'signature_tolerance_seconds': 300,
        'force_human_review': False,
        'judgement_max_tool_calls': 6,
        'retrieval_top_k': 5,
        'worker_timeout_seconds': 120,
        'queue_visibility_timeout_seconds': 720,
        'sweeper_stuck_after_seconds': 900,
    }
    if 'branch_name' not in overrides:
        fields['branch_name'] = current_branch_name()
    return Settings(**(fields | overrides))
