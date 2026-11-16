from typing import Any

from pydantic import TypeAdapter

from spendguard.graph.config import GraphConfig
from spendguard.graph.state import AuditEntry, ReviewState
from spendguard.privacy.hygiene import clean_field
from spendguard.privacy.redaction import redact
from spendguard.rules.checks import base_currency_amount
from spendguard.submission import ExpenseClaim, Submission

SUBMISSION_ADAPTER: TypeAdapter[Submission] = TypeAdapter(Submission)


def _cleaned(
    submission: dict[str, Any], maximum_length: int
) -> tuple[dict[str, Any], list[AuditEntry]]:
    cleaned_submission: dict[str, Any] = {}
    audit: list[AuditEntry] = []
    for field, value in submission.items():
        if not isinstance(value, str):
            cleaned_submission[field] = value
            continue
        cleaned_value = clean_field(value, maximum_length)
        cleaned_submission[field] = cleaned_value
        if cleaned_value != value:
            audit.append(
                {
                    'node': 'intake',
                    'event': 'field_cleaned',
                    'detail': {
                        'field': field,
                        'original_length': len(value),
                        'cleaned_length': len(cleaned_value),
                    },
                }
            )
    return cleaned_submission, audit


def _redacted(submission: dict[str, Any]) -> tuple[dict[str, Any], list[AuditEntry]]:
    redacted_submission: dict[str, Any] = {}
    audit: list[AuditEntry] = []
    for field, value in submission.items():
        redacted_value = redact(value) if isinstance(value, str) else value
        redacted_submission[field] = redacted_value
        if redacted_value != value:
            audit.append({'node': 'intake', 'event': 'field_redacted', 'detail': {'field': field}})
    return redacted_submission, audit


def intake(state: ReviewState, graph_config: GraphConfig) -> ReviewState:
    cleaned_submission, cleaning_audit = _cleaned(
        state['submission'], graph_config.settings.maximum_field_length
    )
    redacted_submission, redaction_audit = _redacted(cleaned_submission)
    submission = SUBMISSION_ADAPTER.validate_python(cleaned_submission)
    amount = base_currency_amount(submission, graph_config.seed_data, graph_config.policy)
    return {
        'submission': cleaned_submission,
        'redacted_submission': redacted_submission,
        'requester': (
            submission.claimant
            if isinstance(submission, ExpenseClaim)
            else graph_config.seed_data.invoice_requester(submission)
        ),
        'base_currency_amount': str(amount),
        'audit': cleaning_audit + redaction_audit,
    }
