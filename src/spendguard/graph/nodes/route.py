from collections.abc import Mapping


def route(state: Mapping[str, object]) -> dict[str, object]:
    raise NotImplementedError
