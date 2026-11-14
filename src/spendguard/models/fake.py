from collections.abc import Callable, Sequence
from operator import itemgetter
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field

from spendguard.settings import StructuredOutputMethod


class FakeResponsesExhaustedError(Exception):
    pass


class FakeChatModel(BaseChatModel):
    model_id: str
    responses: list[AIMessage] = Field(default_factory=list)
    received: list[list[BaseMessage]] = Field(default_factory=list)
    structured_output_calls: list[tuple[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return 'spendguard-fake'

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        if len(self.received) > len(self.responses):
            raise FakeResponsesExhaustedError(
                f'Fake model {self.model_id!r} has no response for call {len(self.received)}'
            )
        response = self.responses[len(self.received) - 1]
        return ChatResult(generations=[ChatGeneration(message=response)])

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        method: StructuredOutputMethod = 'function_calling',
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict[str, Any] | BaseModel]:
        self.structured_output_calls.append((method, schema))
        if method == 'function_calling':
            return super().with_structured_output(schema, include_raw=include_raw, **kwargs)
        output_parser: JsonOutputParser | PydanticOutputParser[BaseModel] = (
            PydanticOutputParser(pydantic_object=schema)
            if isinstance(schema, type) and issubclass(schema, BaseModel)
            else JsonOutputParser()
        )
        if not include_raw:
            return self | output_parser
        parse = RunnablePassthrough.assign(
            parsed=itemgetter('raw') | output_parser, parsing_error=lambda _: None
        )
        leave_unparsed = RunnablePassthrough.assign(parsed=lambda _: None)
        return RunnableMap(raw=self) | parse.with_fallbacks(
            [leave_unparsed], exception_key='parsing_error'
        )

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)


def structured_response(value: BaseModel) -> AIMessage:
    return AIMessage(
        content='',
        tool_calls=[
            {
                'name': type(value).__name__,
                'args': value.model_dump(mode='json'),
                'id': f'call_{type(value).__name__}',
            }
        ],
    )
