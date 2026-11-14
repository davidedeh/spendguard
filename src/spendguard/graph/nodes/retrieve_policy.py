from collections.abc import Mapping


def retrieve_policy(state: Mapping[str, object]) -> dict[str, object]:
    raise NotImplementedError
