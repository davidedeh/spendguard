import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from spendguard.models.fake import FakeChatModel, FakeResponsesExhaustedError, structured_response


class Verdict(BaseModel):
    code: str
    clause_count: int


def test_structured_output_returns_the_scripted_responses_in_order() -> None:
    model = FakeChatModel(
        model_id='fake-chat',
        responses=[
            structured_response(Verdict(code='PURPOSE_UNCLEAR', clause_count=2)),
            structured_response(Verdict(code='RECEIPT_MISMATCH', clause_count=1)),
        ],
    )
    structured_model = model.with_structured_output(Verdict)

    assert structured_model.invoke('first') == Verdict(code='PURPOSE_UNCLEAR', clause_count=2)
    assert structured_model.invoke('second') == Verdict(code='RECEIPT_MISMATCH', clause_count=1)


def test_every_prompt_the_model_receives_is_recorded_in_order() -> None:
    model = FakeChatModel(
        model_id='fake-chat',
        responses=[AIMessage(content='one'), AIMessage(content='two')],
    )

    model.invoke([SystemMessage(content='the policy'), HumanMessage(content='draft a case')])
    model.invoke('draft another')

    assert [[message.content for message in prompt] for prompt in model.received] == [
        ['the policy', 'draft a case'],
        ['draft another'],
    ]


def test_a_call_beyond_the_scripted_responses_names_the_model_and_the_call() -> None:
    model = FakeChatModel(model_id='fake-drafter', responses=[AIMessage(content='only one')])
    model.invoke('first')

    with pytest.raises(
        FakeResponsesExhaustedError, match="'fake-drafter' has no response for call 2"
    ):
        model.invoke('second')
