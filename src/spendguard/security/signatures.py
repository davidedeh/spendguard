def sign(key: bytes, timestamp: int, raw_body: bytes) -> str:
    raise NotImplementedError


def verify_signature(key: bytes, header: str, raw_body: bytes, now: int) -> bool:
    raise NotImplementedError
