import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from spendguard.models.fake import FakeChatModel
from spendguard.models.structured_output import (
    StructuredOutputError,
    output_schema,
    structured_reply,
)


class Verdict(BaseModel):
    code: str
    outcome: str


def test_a_reply_cut_off_mid_object_raises_with_the_provider_s_stop_reason() -> None:
    model = FakeChatModel(
        model_id='fake-chat',
        responses=[
            AIMessage(
                content='PUR{ "code": "PURPOSE_UNCLEAR",\n   \n\n ',
                response_metadata={'stopReason': 'max_tokens'},
            )
        ],
    )

    with pytest.raises(StructuredOutputError) as raised:
        structured_reply(
            model, 'json_schema', output_schema(Verdict), [HumanMessage(content='Give a verdict.')]
        )

    assert raised.value.stop_reason == 'max_tokens'
