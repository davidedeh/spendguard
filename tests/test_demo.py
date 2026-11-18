from spendguard.demo import demo_samples, frozen_case
from spendguard.demo_presentation import clause_excerpt, quoted
from spendguard.eval.freeze import CASE_DIRECTORY
from spendguard.submission import ExpenseClaimDelivery


def test_the_demo_samples_are_clean_judgement_and_injection_cases_from_the_frozen_dataset() -> None:
    samples = demo_samples(CASE_DIRECTORY)

    deliveries = [ExpenseClaimDelivery.model_validate(sample.delivery) for sample in samples]
    cases = [frozen_case(CASE_DIRECTORY, delivery.claim_id) for delivery in deliveries]
    assert [sample.webhook_path for sample in samples] == ['/webhooks/expense-claims'] * 3
    assert [(case.slice, case.expected_route) for case in cases] == [
        ('clean', 'auto_approve'),
        ('judgement_positive', 'human_review'),
        ('injection', 'human_review'),
    ]
    assert [delivery.claim for delivery in deliveries] == [case.submission for case in cases]


def test_the_demo_approves_the_judgement_sample_and_rejects_the_injection_sample() -> None:
    assert [sample.decision for sample in demo_samples(CASE_DIRECTORY)] == [
        None,
        'approve',
        'reject',
    ]


def test_the_clause_excerpt_is_the_sentence_sharing_most_words_with_the_quotes() -> None:
    clause_text = (
        'Every submission states its business purpose. The purpose names the organisation\n'
        'the spend was for; a role such as "a client" is not enough. Names are not required.'
    )

    assert (
        clause_excerpt(clause_text, ['Lunch with client contacts.'])
        == 'The purpose names the organisation the spend was for; a role such as "a client" is not'
        ' enough.'
    )
    assert clause_excerpt(clause_text, ['Receipt required']) == 'Names are not required.'
    assert clause_excerpt(clause_text, []) == 'Every submission states its business purpose.'


def test_a_quote_spanning_lines_is_shown_on_one_line_with_its_breaks_marked() -> None:
    assert (
        quoted('Meal 57.77\nMeal 57.77\n  TOTAL 115.54')
        == '"Meal 57.77 / Meal 57.77 / TOTAL 115.54"'
    )
    assert quoted('Lunch with client contacts.') == '"Lunch with client contacts."'
