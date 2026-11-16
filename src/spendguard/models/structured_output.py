import json
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.utils.json_schema import dereference_refs
from pydantic import BaseModel

from spendguard.settings import StructuredOutputMethod

UNLOCKABLE_KEYWORDS = frozenset(
    {'minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum', 'multipleOf'}
    | {'minLength', 'maxLength'}
)


class StructuredOutputError(ValueError):
    pass


def output_schema(model_class: type[BaseModel]) -> dict[str, Any]:
    schema = dereference_refs(_with_enum_tags(model_class.model_json_schema()))
    return {key: value for key, value in schema.items() if key != '$defs'}


def schema_for_method(schema: dict[str, Any], method: StructuredOutputMethod) -> dict[str, Any]:
    match method:
        case 'function_calling':
            return schema
        case 'json_schema':
            lockable: dict[str, Any] = _lockable(schema)
            return lockable


def structured_reply(
    model: BaseChatModel,
    method: StructuredOutputMethod,
    schema: dict[str, Any],
    messages: Sequence[BaseMessage],
) -> dict[str, Any]:
    result = model.with_structured_output(
        schema_for_method(schema, method), method=method, include_raw=True
    ).invoke(list(messages))
    if not isinstance(result, dict):
        raise StructuredOutputError(f'expected a reply, the model returned {type(result).__name__}')
    output = result['parsed']
    if output is None:
        output = _json_object_in(result['raw'])
    if not isinstance(output, dict):
        raise StructuredOutputError(str(result['parsing_error'] or 'the reply holds no object'))
    filled: dict[str, Any] = _with_nulls_filled_in(output, schema)
    return filled


def _with_enum_tags(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_with_enum_tags(value) for value in schema]
    if not isinstance(schema, dict):
        return schema
    converted: dict[str, Any] = {}
    for key, value in schema.items():
        if key == 'const':
            converted['enum'] = [value]
        elif key == 'oneOf':
            converted['anyOf'] = _with_enum_tags(value)
        elif key != 'discriminator':
            converted[key] = _with_enum_tags(value)
    return converted


def _lockable(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_lockable(value) for value in schema]
    if not isinstance(schema, dict):
        return schema
    lockable = {
        key: _lockable(value)
        for key, value in schema.items()
        if key not in UNLOCKABLE_KEYWORDS and not (key == 'minItems' and value > 1)
    }
    if 'properties' in lockable:
        lockable['additionalProperties'] = False
    return lockable


def _with_nulls_filled_in(value: Any, schema: dict[str, Any]) -> Any:
    if isinstance(value, list):
        return [_with_nulls_filled_in(item, schema.get('items', {})) for item in value]
    if not isinstance(value, dict):
        return value
    properties = _object_schema_for(value, schema).get('properties', {})
    filled = {
        key: None
        if item == '' and _allows_null(properties.get(key, {}))
        else _with_nulls_filled_in(item, properties.get(key, {}))
        for key, item in value.items()
    }
    for key, property_schema in properties.items():
        if key not in filled and _allows_null(property_schema):
            filled[key] = None
    return filled


def _object_schema_for(value: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    options: list[dict[str, Any]] = [
        option for option in schema.get('anyOf', [schema]) if 'properties' in option
    ]
    for option in options:
        if value.get('type') in option['properties'].get('type', {}).get('enum', []):
            return option
    return options[0] if options else {}


def _allows_null(schema: dict[str, Any]) -> bool:
    return any(option.get('type') == 'null' for option in schema.get('anyOf', [schema]))


def _json_object_in(message: object) -> object:
    if not isinstance(message, BaseMessage):
        return None
    start, end = message.text.find('{'), message.text.rfind('}')
    if start == -1 or end < start:
        return None
    try:
        return json.loads(message.text[start : end + 1])
    except json.JSONDecodeError:
        return None
