from collections.abc import Mapping


def finalize(state: Mapping[str, object]) -> dict[str, object]:
    raise NotImplementedError
