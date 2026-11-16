import yaml

from spendguard.demo import demo_samples
from spendguard.eval.cases import EvalCase
from spendguard.eval.freeze import CASE_DIRECTORY
from spendguard.submission import ExpenseClaimDelivery


def frozen_case(case_id: str) -> EvalCase:
    return EvalCase.model_validate(yaml.safe_load((CASE_DIRECTORY / f'{case_id}.yaml').read_text()))


def test_the_demo_samples_are_clean_judgement_and_injection_cases_from_the_frozen_dataset() -> None:
    samples = demo_samples(CASE_DIRECTORY)

    deliveries = [ExpenseClaimDelivery.model_validate(sample.delivery) for sample in samples]
    cases = [frozen_case(delivery.claim_id) for delivery in deliveries]
    assert [sample.webhook_path for sample in samples] == ['/webhooks/expense-claims'] * 3
    assert [(case.slice, case.expected_route) for case in cases] == [
        ('clean', 'auto_approve'),
        ('judgement_positive', 'human_review'),
        ('injection', 'human_review'),
    ]
    assert [delivery.claim for delivery in deliveries] == [case.submission for case in cases]
