from pydantic import TypeAdapter

from spendguard.graph.config import GraphConfig
from spendguard.graph.state import ReviewState
from spendguard.rules.checks import base_currency_amount
from spendguard.submission import ExpenseClaim, Submission

SUBMISSION_ADAPTER: TypeAdapter[Submission] = TypeAdapter(Submission)


def intake(state: ReviewState, graph_config: GraphConfig) -> ReviewState:
    submission = SUBMISSION_ADAPTER.validate_python(state['submission'])
    amount = base_currency_amount(submission, graph_config.seed_data, graph_config.policy)
    return {
        'requester': (
            submission.claimant
            if isinstance(submission, ExpenseClaim)
            else graph_config.seed_data.invoice_requester(submission)
        ),
        'base_currency_amount': str(amount),
    }
