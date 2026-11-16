from langchain_core.messages import AIMessage

from spendguard.models.fake import FakeChatModel
from spendguard.models.smoke import check_structured_output
from spendguard.settings import ChatModelSettings


def test_the_structured_output_check_uses_the_method_the_model_is_configured_with() -> None:
    model = FakeChatModel(
        model_id='fake-locked',
        responses=[AIMessage(content='{"clause_reference": "3.4", "severity": "medium"}')],
    )
    model_settings = ChatModelSettings(
        provider='fake', model_id='fake-locked', structured_output_method='json_schema'
    )

    detail = check_structured_output(model, model_settings)

    assert detail.startswith('parsed 3.4/medium')
    assert [method for method, _ in model.structured_output_calls] == ['json_schema']
